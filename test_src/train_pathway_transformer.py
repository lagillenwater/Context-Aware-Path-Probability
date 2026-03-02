"""
Train and evaluate PathwayTransformer on real pathway data.

This script:
1. Prepares sequential training data from Hetionet
2. Trains PathwayTransformer with MSE loss
3. Evaluates against baseline DegreeSignatureNN
4. Tests for r > 0.95 target achievement

Usage:
    python train_pathway_transformer.py
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from typing import Dict, Tuple

from src.pathway_sequence_data import prepare_transformer_data
from src.pathway_transformer import PathwayTransformer, count_parameters
from src.models.degree_signature_nn import DegreeSignatureNN
from src.baseline_framework import EvaluationMetrics
from scipy.stats import pearsonr


def train_transformer(model: nn.Module,
                     train_loader: DataLoader,
                     val_loader: DataLoader,
                     loss_fn: nn.Module,
                     n_epochs: int = 500,
                     learning_rate: float = 0.001,
                     early_stopping_patience: int = 50,
                     device: str = 'cpu') -> Dict:
    """
    Train PathwayTransformer.

    Parameters
    ----------
    model : nn.Module
        PathwayTransformer model
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader
        Validation data loader
    loss_fn : nn.Module
        Loss function
    n_epochs : int
        Maximum epochs
    learning_rate : float
        Learning rate
    early_stopping_patience : int
        Early stopping patience
    device : str
        Device to train on

    Returns
    -------
    history : dict
        Training history
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    print("Training PathwayTransformer...")

    for epoch in range(n_epochs):
        model.train()
        train_losses = []

        for batch in train_loader:
            node_features = batch['node_features'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['pathway_count'].to(device)

            optimizer.zero_grad()

            outputs = model(node_features, attention_mask)
            predictions = outputs['prediction']

            loss = loss_fn(predictions, targets)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                node_features = batch['node_features'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                targets = batch['pathway_count'].to(device)

                outputs = model(node_features, attention_mask)
                predictions = outputs['prediction']

                loss = loss_fn(predictions, targets)
                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{n_epochs}: "
                  f"train_loss={avg_train_loss:.4f}, "
                  f"val_loss={avg_val_loss:.4f}")

    return history


def evaluate_transformer(model: nn.Module,
                        data_loader: DataLoader,
                        device: str = 'cpu') -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate PathwayTransformer.

    Parameters
    ----------
    model : nn.Module
        Trained model
    data_loader : DataLoader
        Data loader
    device : str
        Device

    Returns
    -------
    predictions : np.ndarray
        Predicted pathway counts
    targets : np.ndarray
        True pathway counts
    """
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in data_loader:
            node_features = batch['node_features'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['pathway_count'].numpy()

            outputs = model(node_features, attention_mask)
            predictions = outputs['prediction'].cpu().numpy()

            all_predictions.append(predictions.ravel())
            all_targets.append(targets.ravel())

    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)

    return predictions, targets


def main():
    """Main training and evaluation."""
    print("=" * 80)
    print("PATHWAYTRANSFORMER TRAINING AND EVALUATION")
    print("=" * 80)
    print("\nGoal: Achieve r > 0.95 robustly on real pathway data")
    print("Target: Improve upon baseline DegreeSignatureNN (r = 0.9538)")

    repo_dir = Path.cwd()

    print("\n" + "-" * 80)
    print("LOADING DATA")
    print("-" * 80)

    training_data_file = repo_dir / 'results' / 'pathway_nn' / \
                        'training_data' / 'CbGpPW_training_data.csv'

    if not training_data_file.exists():
        print(f"Training data not found: {training_data_file}")
        print("Run prepare_and_test_pathway_data.py first")
        return

    df = pd.read_csv(training_data_file)
    print(f"Loaded degree bins: {len(df)} bins")

    data_dir = repo_dir / 'data'
    edge1_file = data_dir / 'edges' / 'CbG.sparse.npz'
    edge2_file = data_dir / 'edges' / 'GpPW.sparse.npz'

    edge1 = sp.load_npz(str(edge1_file))
    edge2 = sp.load_npz(str(edge2_file))

    if edge1.dtype == bool:
        edge1 = edge1.astype(np.int32)
    if edge2.dtype == bool:
        edge2 = edge2.astype(np.int32)

    print("\n" + "-" * 80)
    print("PREPARING SEQUENTIAL DATA")
    print("-" * 80)

    dataset, targets = prepare_transformer_data(
        df, edge1, edge2,
        n_samples_per_bin=10,
        n_intermediate_samples=5,
        random_seed=42
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"\nDataset split:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False
    )

    print("\n" + "-" * 80)
    print("INITIALIZING MODELS")
    print("-" * 80)

    transformer_model = PathwayTransformer(
        node_feature_dim=10,
        d_model=64,
        nhead=4,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.1,
        max_seq_len=dataset.max_seq_len
    )

    print(f"\nPathwayTransformer:")
    print(f"  Parameters: {count_parameters(transformer_model):,}")
    print(f"  Architecture: d_model=64, heads=4, layers=3")

    print("\n" + "-" * 80)
    print("TRAINING TRANSFORMER")
    print("-" * 80)

    history = train_transformer(
        transformer_model,
        train_loader,
        val_loader,
        loss_fn=nn.MSELoss(),
        n_epochs=500,
        learning_rate=0.001,
        early_stopping_patience=50,
        device='cpu'
    )

    print("\n" + "-" * 80)
    print("EVALUATING ON VALIDATION SET")
    print("-" * 80)

    val_preds, val_targets = evaluate_transformer(
        transformer_model, val_loader, device='cpu'
    )

    metrics = EvaluationMetrics.compute_all(val_targets, val_preds)

    print(f"\nPathwayTransformer Performance:")
    print(f"  Pearson r: {metrics['pearson_r']:.4f}")
    print(f"  Spearman r: {metrics['spearman_r']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  Bias: {metrics['bias']:+.4f}")
    print(f"  R²: {metrics['r2']:.4f}")

    print("\n" + "-" * 80)
    print("BASELINE COMPARISON")
    print("-" * 80)

    print("\nTraining DegreeSignatureNN baseline...")

    feature_cols = ['source_bin', 'target_bin'] + \
                   [f'inter_sig_{i}' for i in range(100)]
    X = df[feature_cols].values
    y = df['pathway_count_mean'].values

    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    baseline_model = DegreeSignatureNN(
        hidden_dims=(128, 64, 32),
        dropout=0.2,
        learning_rate=0.001,
        batch_size=32,
        n_epochs=500,
        early_stopping_patience=50,
        loss_fn=nn.MSELoss(),
        random_state=42,
        device='cpu'
    )

    baseline_model.fit(X_train, y_train)
    baseline_preds = baseline_model.predict(X_val)

    baseline_r = pearsonr(y_val, baseline_preds)[0]
    baseline_rmse = np.sqrt(np.mean((baseline_preds - y_val)**2))
    baseline_bias = np.mean(baseline_preds - y_val)

    print(f"\nDegreeSignatureNN Performance:")
    print(f"  Pearson r: {baseline_r:.4f}")
    print(f"  RMSE: {baseline_rmse:.4f}")
    print(f"  Bias: {baseline_bias:+.4f}")

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    print(f"\n{'Model':<25} {'Correlation r':<15} {'RMSE':<15} {'Bias':<15}")
    print("-" * 70)
    print(f"{'PathwayTransformer':<25} {metrics['pearson_r']:<15.4f} "
          f"{metrics['rmse']:<15.4f} {metrics['bias']:<+15.4f}")
    print(f"{'DegreeSignatureNN':<25} {baseline_r:<15.4f} "
          f"{baseline_rmse:<15.4f} {baseline_bias:<+15.4f}")

    improvement = metrics['pearson_r'] - baseline_r

    print(f"\nImprovement: {improvement:+.4f}")

    print("\n" + "-" * 80)
    print("TARGET ASSESSMENT")
    print("-" * 80)

    print(f"\nTarget: r > 0.95")
    print(f"PathwayTransformer: r = {metrics['pearson_r']:.4f}")

    if metrics['pearson_r'] > 0.95:
        print(f"STATUS: SUCCESS - Target achieved!")
        print(f"Exceeded target by: {metrics['pearson_r'] - 0.95:.4f}")
    else:
        print(f"STATUS: NOT MET")
        print(f"Gap to target: {0.95 - metrics['pearson_r']:.4f}")

    if improvement > 0:
        print(f"\nTransformer improved over baseline by {improvement:+.4f}")
    else:
        print(f"\nTransformer did NOT improve over baseline ({improvement:+.4f})")

    output_dir = repo_dir / 'results' / 'pathway_transformer'
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame([
        {
            'model': 'PathwayTransformer',
            'pearson_r': metrics['pearson_r'],
            'spearman_r': metrics['spearman_r'],
            'rmse': metrics['rmse'],
            'bias': metrics['bias'],
            'r2': metrics['r2'],
            'n_parameters': count_parameters(transformer_model)
        },
        {
            'model': 'DegreeSignatureNN',
            'pearson_r': baseline_r,
            'spearman_r': np.nan,
            'rmse': baseline_rmse,
            'bias': baseline_bias,
            'r2': np.nan,
            'n_parameters': np.nan
        }
    ])

    results_df.to_csv(output_dir / 'transformer_vs_baseline.csv', index=False)

    torch.save(transformer_model.state_dict(),
              output_dir / 'pathway_transformer.pt')

    print(f"\n\nResults saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
