"""
Test Multi-Task Learning across multiple metapaths.

Key insight: Different metapaths share the same underlying degree-count
relationship. Training on all metapaths simultaneously provides 24x more
data and learns shared patterns.

Architecture:
- Shared encoder: learns degree features
- Metapath embedding: encodes which metapath
- Combined predictor: predicts count

Training data: 10K pairs × N metapaths = N×10K examples

Usage:
    python test_src/test_multitask_metapaths.py

References:
    - Baseline linear (single metapath): r=0.787
    - Current best: RF r=0.778
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys

repo_dir = Path(__file__).parent.parent
sys.path.insert(0, str(repo_dir))

from test_src.validate_mean_variance_prediction import (
    load_permuted_edge_matrices,
    sample_pairs,
    extract_degree_features,
    compute_pathway_counts,
    evaluate_z_scores
)


# Define 2-hop metapaths to train on
METAPATHS = [
    ('CbG', 'GpPW'),   # Compound-binds-Gene-participates-Pathway
    ('CbG', 'GiG'),    # Compound-binds-Gene-interacts-Gene
    ('CtD', 'DaG'),    # Compound-treats-Disease-associates-Gene
    ('CtD', 'DuG'),    # Compound-treats-Disease-upregulates-Gene
    ('CtD', 'DdG'),    # Compound-treats-Disease-downregulates-Gene
    ('GiG', 'GpPW'),   # Gene-interacts-Gene-participates-Pathway
]


class MultiTaskPathPredictor(nn.Module):
    """
    Multi-task neural network for pathway count prediction.

    Learns shared degree-count relationship across all metapaths.
    """

    def __init__(self, n_metapaths, hidden_dim=128, dropout=0.2):
        super(MultiTaskPathPredictor, self).__init__()

        # Shared encoder for degree features
        self.shared_encoder = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Metapath embedding
        self.metapath_embed = nn.Embedding(n_metapaths, 16)

        # Combined predictor
        self.predictor = nn.Sequential(
            nn.Linear(64 + 16, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, degree_features, metapath_ids):
        """
        Args:
            degree_features: (batch_size, 5)
            metapath_ids: (batch_size,) integers

        Returns:
            predictions: (batch_size,)
        """
        degree_encoded = self.shared_encoder(degree_features)
        metapath_encoded = self.metapath_embed(metapath_ids)

        combined = torch.cat([degree_encoded, metapath_encoded], dim=1)
        pred = self.predictor(combined).squeeze()

        return pred


def prepare_multitask_data(data_dir, metapaths, train_perms, n_samples=10000, random_state=42):
    """
    Prepare training data from multiple metapaths.

    Args:
        data_dir: Data directory
        metapaths: List of (edge1_type, edge2_type) tuples
        train_perms: Training permutation indices
        n_samples: Number of pairs per metapath
        random_state: Random seed

    Returns:
        X_all: Degree features (n_metapaths * n_samples, 5)
        y_all: Target counts (n_metapaths * n_samples,)
        metapath_ids_all: Metapath IDs (n_metapaths * n_samples,)
        metapath_data: List of dicts with pairs and features per metapath
    """
    X_all = []
    y_all = []
    metapath_ids_all = []
    metapath_data = []

    for metapath_idx, (edge1_type, edge2_type) in enumerate(metapaths):
        print(f"  Loading {edge1_type} + {edge2_type}...")

        try:
            edge1_perm0, edge2_perm0 = load_permuted_edge_matrices(
                edge1_type, edge2_type, 0, data_dir
            )

            pairs = sample_pairs(edge1_perm0, edge2_perm0, n_samples=n_samples,
                                random_state=random_state + metapath_idx)

            X = extract_degree_features(pairs, edge1_perm0, edge2_perm0)

            counts_list = []
            for perm in train_perms:
                edge1, edge2 = load_permuted_edge_matrices(
                    edge1_type, edge2_type, perm, data_dir
                )
                counts = compute_pathway_counts(pairs, edge1, edge2)
                counts_list.append(counts)

            counts_matrix = np.column_stack(counts_list)
            y = counts_matrix.mean(axis=1)

            X_all.append(X)
            y_all.append(y)
            metapath_ids_all.append(np.full(len(pairs), metapath_idx))

            metapath_data.append({
                'edge1_type': edge1_type,
                'edge2_type': edge2_type,
                'pairs': pairs,
                'X': X
            })

            print(f"    {len(pairs)} pairs, mean count: {y.mean():.3f}")

        except Exception as e:
            print(f"    Skipped: {e}")
            continue

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)
    metapath_ids_all = np.concatenate(metapath_ids_all)

    return X_all, y_all, metapath_ids_all, metapath_data


def train_multitask_model(X, y, metapath_ids, n_metapaths):
    """
    Train multi-task model.

    Args:
        X: Degree features (n_samples, 5)
        y: Target counts (n_samples,)
        metapath_ids: Metapath IDs (n_samples,)
        n_metapaths: Number of metapaths

    Returns:
        model: Trained model
    """
    X_t = torch.FloatTensor(X)
    y_t = torch.FloatTensor(y)
    metapath_ids_t = torch.LongTensor(metapath_ids)

    model = MultiTaskPathPredictor(n_metapaths)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()

    print("Training multi-task model...")
    print(f"  Total training examples: {len(X)}")
    print(f"  Metapaths: {n_metapaths}")

    batch_size = 1024
    n_epochs = 200
    best_loss = np.inf
    patience = 20
    patience_counter = 0

    for epoch in range(n_epochs):
        model.train()

        indices = torch.randperm(len(X))
        epoch_loss = 0
        n_batches = 0

        for i in range(0, len(X), batch_size):
            batch_indices = indices[i:i+batch_size]
            X_batch = X_t[batch_indices]
            y_batch = y_t[batch_indices]
            metapath_batch = metapath_ids_t[batch_indices]

            optimizer.zero_grad()
            pred = model(X_batch, metapath_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    print(f"  Final loss: {avg_loss:.4f}")

    return model


def evaluate_multitask(model, metapath_data, data_dir, test_perms):
    """
    Evaluate multi-task model on test permutations.

    Args:
        model: Trained model
        metapath_data: List of dicts with metapath info
        data_dir: Data directory
        test_perms: Test permutation indices

    Returns:
        results_df: Results for each metapath
    """
    model.eval()
    all_results = []

    for metapath_idx, data in enumerate(metapath_data):
        edge1_type = data['edge1_type']
        edge2_type = data['edge2_type']
        pairs = data['pairs']
        X = data['X']

        X_t = torch.FloatTensor(X)
        metapath_ids_t = torch.LongTensor([metapath_idx] * len(X))

        with torch.no_grad():
            pred = model(X_t, metapath_ids_t).numpy()

        counts_test = []
        for perm in test_perms:
            edge1, edge2 = load_permuted_edge_matrices(
                edge1_type, edge2_type, perm, data_dir
            )
            counts = compute_pathway_counts(pairs, edge1, edge2)
            counts_test.append(counts)
        counts_test = np.column_stack(counts_test)

        for idx, perm in enumerate(test_perms):
            counts = counts_test[:, idx]
            r_mean = np.corrcoef(counts, pred)[0, 1]

            all_results.append({
                'metapath': f'{edge1_type}+{edge2_type}',
                'test_perm': perm,
                'r_mean': r_mean
            })

    results_df = pd.DataFrame(all_results)

    for metapath in results_df['metapath'].unique():
        metapath_results = results_df[results_df['metapath'] == metapath]
        mean_r = metapath_results['r_mean'].mean()
        print(f"  {metapath}: r={mean_r:.4f}")

    overall_mean_r = results_df['r_mean'].mean()
    print(f"\nOverall mean r: {overall_mean_r:.4f}")
    print(f"Comparison to baseline: r=0.787")

    return results_df


def main():
    data_dir = repo_dir / 'data'
    output_dir = repo_dir / 'results' / 'multitask_metapaths'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Multi-Task Learning Across Metapaths")
    print("="*70)

    train_perms = list(range(5))
    test_perms = list(range(15, 21))

    print("\nPreparing multi-task training data...")
    X, y, metapath_ids, metapath_data = prepare_multitask_data(
        data_dir, METAPATHS, train_perms, n_samples=10000, random_state=42
    )

    print(f"\nTotal training data: {len(X)} examples from {len(metapath_data)} metapaths")

    model = train_multitask_model(X, y, metapath_ids, len(metapath_data))

    print("\nEvaluating on test permutations...")
    results_df = evaluate_multitask(model, metapath_data, data_dir, test_perms)

    output_file = output_dir / 'multitask_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nSaved results: {output_file}")


if __name__ == '__main__':
    main()
