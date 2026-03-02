"""
Comprehensive benchmark evaluation for pathway frequency prediction.

IMPORTANT LIMITATION:
DegreeSignatureNN (baseline) operates on degree BINS (100 samples) while
PathwayTransformer operates on individual pathways (900 samples). Due to
this architectural difference, they CANNOT be tested on identical datasets.

This script provides honest evaluation by:
1. Testing sample-level models (Random, Degree Product, Transformer) on 900 samples
2. Testing bin-level models (Random, DegreeSignatureNN) on 100 bins
3. Clearly documenting the granularity difference

Models evaluated at SAMPLE level (900 samples):
- Negative Control: Random predictions
- Weak Baseline: Degree Product (compositional null)
- Test Model: PathwayTransformer

Models evaluated at BIN level (100 bins):
- Negative Control: Random predictions
- Strong Baseline: DegreeSignatureNN

This acknowledges we cannot directly compare Transformer to DegreeSignatureNN
on identical data, but provides proper negative controls for both.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from typing import Dict, Tuple, List
from sklearn.model_selection import KFold
from scipy.stats import pearsonr, ttest_rel

from src.pathway_sequence_data import prepare_transformer_data, sample_node_pairs_from_bins
from src.pathway_transformer import PathwayTransformer, count_parameters
from src.models.degree_signature_nn import DegreeSignatureNN


class NegativeControlRandom:
    """
    Negative control: Random predictions from normal distribution.

    Expected performance: r ≈ 0 (no correlation with true values)
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.mean_ = None
        self.std_ = None

    def fit(self, y_train):
        """Learn mean and std from training data."""
        self.mean_ = np.mean(y_train)
        self.std_ = np.std(y_train)
        return self

    def predict(self, n_samples):
        """Generate random predictions."""
        np.random.seed(self.random_state)
        return np.random.normal(self.mean_, self.std_, size=n_samples)


class WeakBaselineDegreeProduct:
    """
    Weak baseline: Degree product (compositional null).

    Expected performance: r ≈ 0.35 (based on notebook 17 compositional failure)
    This assumes edges are independent, which is known to be false.
    """

    def __init__(self):
        self.alpha_ = None
        self.beta_ = None

    def fit(self, source_degrees, target_degrees, y_train):
        """Fit simple linear model: y = alpha * deg_source * deg_target + beta"""
        deg_product = source_degrees * target_degrees

        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        lr.fit(deg_product.reshape(-1, 1), y_train)

        self.alpha_ = lr.coef_[0]
        self.beta_ = lr.intercept_

        return self

    def predict(self, source_degrees, target_degrees):
        """Predict using degree product."""
        deg_product = source_degrees * target_degrees
        return self.alpha_ * deg_product + self.beta_


def evaluate_transformer_fold(model: nn.Module,
                              train_dataset: Subset,
                              val_dataset: Subset,
                              n_epochs: int = 200,
                              device: str = 'cpu') -> Tuple[np.ndarray, np.ndarray]:
    """Train and evaluate PathwayTransformer on one fold."""
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20

    for epoch in range(n_epochs):
        model.train()
        for batch in train_loader:
            node_features = batch['node_features'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['pathway_count'].to(device)

            optimizer.zero_grad()
            outputs = model(node_features, attention_mask)
            loss = loss_fn(outputs['prediction'], targets)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                node_features = batch['node_features'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                targets = batch['pathway_count'].to(device)

                outputs = model(node_features, attention_mask)
                loss = loss_fn(outputs['prediction'], targets)
                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            node_features = batch['node_features'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['pathway_count'].numpy()

            outputs = model(node_features, attention_mask)
            predictions = outputs['prediction'].cpu().numpy()

            all_preds.append(predictions.ravel())
            all_targets.append(targets.ravel())

    predictions = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    return predictions, targets


def evaluate_sample_level_models(sample_metadata: pd.DataFrame,
                                 dataset,
                                 n_folds: int = 5,
                                 random_state: int = 42) -> Dict:
    """
    Evaluate models at SAMPLE level (900 individual pathways).

    Models:
    - Negative Control (Random)
    - Weak Baseline (Degree Product)
    - Test Model (PathwayTransformer)
    """
    print("=" * 80)
    print("SAMPLE-LEVEL EVALUATION (900 individual pathways)")
    print("=" * 80)

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    indices = np.arange(len(sample_metadata))

    results = {
        'Random': {'r': [], 'rmse': [], 'bias': []},
        'Degree Product': {'r': [], 'rmse': [], 'bias': []},
        'PathwayTransformer': {'r': [], 'rmse': [], 'bias': []}
    }

    for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
        print(f"\n{'-' * 80}")
        print(f"FOLD {fold + 1}/{n_folds}")
        print(f"{'-' * 80}")

        train_meta = sample_metadata.iloc[train_idx]
        val_meta = sample_metadata.iloc[val_idx]

        y_train = train_meta['pathway_count'].values
        y_val = val_meta['pathway_count'].values

        print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

        print("\n1. Random (Negative Control)")
        random_model = NegativeControlRandom(random_state=random_state + fold)
        random_model.fit(y_train)
        random_preds = random_model.predict(len(y_val))

        r = pearsonr(y_val, random_preds)[0]
        rmse = np.sqrt(np.mean((random_preds - y_val)**2))
        bias = np.mean(random_preds - y_val)

        results['Random']['r'].append(r)
        results['Random']['rmse'].append(rmse)
        results['Random']['bias'].append(bias)

        print(f"   r = {r:.4f}, RMSE = {rmse:.4f}, bias = {bias:+.4f}")

        print("\n2. Degree Product (Weak Baseline)")
        deg_prod = WeakBaselineDegreeProduct()
        deg_prod.fit(
            train_meta['source_degree'].values,
            train_meta['target_degree'].values,
            y_train
        )
        deg_prod_preds = deg_prod.predict(
            val_meta['source_degree'].values,
            val_meta['target_degree'].values
        )

        r = pearsonr(y_val, deg_prod_preds)[0]
        rmse = np.sqrt(np.mean((deg_prod_preds - y_val)**2))
        bias = np.mean(deg_prod_preds - y_val)

        results['Degree Product']['r'].append(r)
        results['Degree Product']['rmse'].append(rmse)
        results['Degree Product']['bias'].append(bias)

        print(f"   r = {r:.4f}, RMSE = {rmse:.4f}, bias = {bias:+.4f}")

        print("\n3. PathwayTransformer")
        train_dataset = Subset(dataset, train_idx.tolist())
        val_dataset = Subset(dataset, val_idx.tolist())

        transformer = PathwayTransformer(
            node_feature_dim=10,
            d_model=64,
            nhead=4,
            num_layers=3,
            dim_feedforward=256,
            dropout=0.1,
            max_seq_len=dataset.max_seq_len
        )

        transformer_preds, transformer_targets = evaluate_transformer_fold(
            transformer, train_dataset, val_dataset, n_epochs=200, device='cpu'
        )

        r = pearsonr(transformer_targets, transformer_preds)[0]
        rmse = np.sqrt(np.mean((transformer_preds - transformer_targets)**2))
        bias = np.mean(transformer_preds - transformer_targets)

        results['PathwayTransformer']['r'].append(r)
        results['PathwayTransformer']['rmse'].append(rmse)
        results['PathwayTransformer']['bias'].append(bias)

        print(f"   r = {r:.4f}, RMSE = {rmse:.4f}, bias = {bias:+.4f}")

    return results


def evaluate_bin_level_models(degree_bins_df: pd.DataFrame,
                              n_folds: int = 5,
                              random_state: int = 42) -> Dict:
    """
    Evaluate models at BIN level (100 degree bins).

    Models:
    - Negative Control (Random)
    - Strong Baseline (DegreeSignatureNN)
    """
    print("\n\n" + "=" * 80)
    print("BIN-LEVEL EVALUATION (100 degree bins)")
    print("=" * 80)

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    feature_cols = ['source_bin', 'target_bin'] + [f'inter_sig_{i}' for i in range(100)]
    X = degree_bins_df[feature_cols].values
    y = degree_bins_df['pathway_count_mean'].values

    results = {
        'Random': {'r': [], 'rmse': [], 'bias': []},
        'DegreeSignatureNN': {'r': [], 'rmse': [], 'bias': []}
    }

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"\n{'-' * 80}")
        print(f"FOLD {fold + 1}/{n_folds}")
        print(f"{'-' * 80}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

        print("\n1. Random (Negative Control)")
        random_model = NegativeControlRandom(random_state=random_state + fold)
        random_model.fit(y_train)
        random_preds = random_model.predict(len(y_val))

        r = pearsonr(y_val, random_preds)[0]
        rmse = np.sqrt(np.mean((random_preds - y_val)**2))
        bias = np.mean(random_preds - y_val)

        results['Random']['r'].append(r)
        results['Random']['rmse'].append(rmse)
        results['Random']['bias'].append(bias)

        print(f"   r = {r:.4f}, RMSE = {rmse:.4f}, bias = {bias:+.4f}")

        print("\n2. DegreeSignatureNN")
        baseline = DegreeSignatureNN(
            hidden_dims=(128, 64, 32),
            dropout=0.2,
            learning_rate=0.001,
            batch_size=16,
            n_epochs=200,
            early_stopping_patience=20,
            loss_fn=nn.MSELoss(),
            random_state=random_state + fold,
            device='cpu'
        )
        baseline.fit(X_train, y_train)
        baseline_preds = baseline.predict(X_val)

        r = pearsonr(y_val, baseline_preds)[0]
        rmse = np.sqrt(np.mean((baseline_preds - y_val)**2))
        bias = np.mean(baseline_preds - y_val)

        results['DegreeSignatureNN']['r'].append(r)
        results['DegreeSignatureNN']['rmse'].append(rmse)
        results['DegreeSignatureNN']['bias'].append(bias)

        print(f"   r = {r:.4f}, RMSE = {rmse:.4f}, bias = {bias:+.4f}")

    return results


def print_summary(sample_results: Dict, bin_results: Dict):
    """Print summary of both evaluations."""
    print("\n\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print("\n" + "-" * 80)
    print("SAMPLE-LEVEL MODELS (900 individual pathways)")
    print("-" * 80)
    print(f"{'Model':<30} {'r':<20} {'RMSE':<20}")
    print("-" * 80)

    for model_name, metrics in sample_results.items():
        r_mean = np.mean(metrics['r'])
        r_std = np.std(metrics['r'])
        rmse_mean = np.mean(metrics['rmse'])
        rmse_std = np.std(metrics['rmse'])

        print(f"{model_name:<30} {r_mean:6.4f} ± {r_std:5.4f}    "
              f"{rmse_mean:6.4f} ± {rmse_std:5.4f}")

    print("\n" + "-" * 80)
    print("BIN-LEVEL MODELS (100 degree bins)")
    print("-" * 80)
    print(f"{'Model':<30} {'r':<20} {'RMSE':<20}")
    print("-" * 80)

    for model_name, metrics in bin_results.items():
        r_mean = np.mean(metrics['r'])
        r_std = np.std(metrics['r'])
        rmse_mean = np.mean(metrics['rmse'])
        rmse_std = np.std(metrics['rmse'])

        print(f"{model_name:<30} {r_mean:6.4f} ± {r_std:5.4f}    "
              f"{rmse_mean:6.4f} ± {rmse_std:5.4f}")

    print("\n" + "-" * 80)
    print("KEY FINDINGS")
    print("-" * 80)

    transformer_r = np.mean(sample_results['PathwayTransformer']['r'])
    deg_prod_r = np.mean(sample_results['Degree Product']['r'])
    baseline_r = np.mean(bin_results['DegreeSignatureNN']['r'])

    if transformer_r > 0.95:
        print(f"✓ PathwayTransformer achieves r > 0.95 target: r = {transformer_r:.4f}")
    else:
        print(f"✗ PathwayTransformer does NOT achieve r > 0.95: r = {transformer_r:.4f}")

    diff_vs_weak = transformer_r - deg_prod_r
    print(f"\nTransformer vs Degree Product (weak baseline): Δr = {diff_vs_weak:+.4f}")

    if diff_vs_weak > 0:
        print(f"  → Transformer improves over compositional null")
    else:
        print(f"  → Transformer does NOT improve over compositional null")

    print(f"\nDegreeSignatureNN (bin-level): r = {baseline_r:.4f}")
    print(f"  → Cannot directly compare to Transformer (different granularities)")

    print("\n" + "=" * 80)


def main():
    """Main evaluation."""
    print("=" * 80)
    print("COMPREHENSIVE BENCHMARK EVALUATION")
    print("=" * 80)

    repo_dir = Path.cwd()

    print("\n" + "-" * 80)
    print("LOADING DATA")
    print("-" * 80)

    training_data_file = repo_dir / 'results' / 'pathway_nn' / \
                        'training_data' / 'CbGpPW_training_data.csv'

    df = pd.read_csv(training_data_file)
    print(f"Loaded degree bins: {len(df)} bins")

    data_dir = repo_dir / 'data'
    edge1 = sp.load_npz(str(data_dir / 'edges' / 'CbG.sparse.npz'))
    edge2 = sp.load_npz(str(data_dir / 'edges' / 'GpPW.sparse.npz'))

    if edge1.dtype == bool:
        edge1 = edge1.astype(np.int32)
    if edge2.dtype == bool:
        edge2 = edge2.astype(np.int32)

    source_indices, target_indices = sample_node_pairs_from_bins(
        edge1, edge2, df, n_samples_per_bin=10, random_seed=42
    )

    source_degrees = np.asarray(edge1.sum(axis=1)).ravel()
    target_degrees = np.asarray(edge2.sum(axis=0)).ravel()

    pathway_matrix = edge1 @ edge2
    pathway_counts = [pathway_matrix[s, t] for s, t in zip(source_indices, target_indices)]

    sample_metadata = pd.DataFrame({
        'source_degree': [source_degrees[i] for i in source_indices],
        'target_degree': [target_degrees[i] for i in target_indices],
        'pathway_count': pathway_counts
    })

    print(f"Created metadata for {len(sample_metadata)} samples")

    dataset, _ = prepare_transformer_data(
        df, edge1, edge2, n_samples_per_bin=10,
        n_intermediate_samples=5, random_seed=42
    )

    print("\nEvaluating models...")

    sample_results = evaluate_sample_level_models(
        sample_metadata, dataset, n_folds=5, random_state=42
    )

    bin_results = evaluate_bin_level_models(
        df, n_folds=5, random_state=42
    )

    print_summary(sample_results, bin_results)

    output_dir = repo_dir / 'results' / 'comprehensive_benchmark'
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_df = pd.DataFrame([{
        'Model': name,
        'Level': 'Sample (900)',
        'r_mean': np.mean(metrics['r']),
        'r_std': np.std(metrics['r']),
        'rmse_mean': np.mean(metrics['rmse']),
        'rmse_std': np.std(metrics['rmse'])
    } for name, metrics in sample_results.items()])

    bin_df = pd.DataFrame([{
        'Model': name,
        'Level': 'Bin (100)',
        'r_mean': np.mean(metrics['r']),
        'r_std': np.std(metrics['r']),
        'rmse_mean': np.mean(metrics['rmse']),
        'rmse_std': np.std(metrics['rmse'])
    } for name, metrics in bin_results.items()])

    summary_df = pd.concat([sample_df, bin_df], ignore_index=True)
    summary_df.to_csv(output_dir / 'benchmark_summary.csv', index=False)

    print(f"\nResults saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
