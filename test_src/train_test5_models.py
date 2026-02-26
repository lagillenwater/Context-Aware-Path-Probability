"""
Train Test 5 models: StandardScaler WITHOUT class weighting

This script trains models to optimize for probability calibration rather than classification accuracy.

Configuration:
- StandardScaler: YES (feature normalization)
- Class weighting: NO (no pos_weight, no class_weight parameter)
- Goal: Match notebook 04's LogReg calibration performance
"""

import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp

# Add src to path
sys.path.insert(0, 'src')
sys.path.insert(0, '../src')

# Import project modules
from model_comparison import prepare_edge_features_and_labels, filter_zero_degree_nodes, SimpleNN
from simple_models import SingleLayerNN

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

# Setup paths
repo_dir = Path.cwd() if (Path.cwd() / 'data').exists() else Path.cwd().parent
data_dir = repo_dir / 'data'
results_dir = repo_dir / 'results' / 'nn_optimizer_comparison'
results_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Test 5: StandardScaler WITHOUT Class Weighting")
print("=" * 80)

# Load data
edge_type = 'CbG'
edge_file_path = data_dir / 'permutations' / '000.hetmat' / 'edges' / f'{edge_type}.sparse.npz'

print(f"\n1. Loading edge data from: {edge_file_path}")
edge_matrix = sp.load_npz(str(edge_file_path))
print(f"   Original edge matrix: {edge_matrix.shape} with {edge_matrix.nnz} edges")

# Filter zero-degree nodes
print("\n2. Filtering zero-degree nodes...")
filtered_edge_matrix, source_mapping, target_mapping = filter_zero_degree_nodes(edge_matrix)
print(f"   Filtered edge matrix: {filtered_edge_matrix.shape} with {filtered_edge_matrix.nnz} edges")

# Save filtered matrix temporarily
filtered_edge_path = data_dir / 'permutations' / '000.hetmat' / 'edges' / f'filtered_{edge_type}_temp.sparse.npz'
sp.save_npz(str(filtered_edge_path), filtered_edge_matrix)

# Prepare features and labels
print("\n3. Preparing features and labels...")
X, y = prepare_edge_features_and_labels(
    str(filtered_edge_path),
    sample_ratio=0.01,
    adaptive_sampling=True,
    enhanced_features=False
)
filtered_edge_path.unlink()  # Clean up temp file

print(f"   Loaded {X.shape[0]} samples with {X.shape[1]} features")
print(f"   Positive samples: {y.sum()}, Negative samples: {(1-y).sum()}")
print(f"   Class balance: {y.sum()/len(y):.3f}")

# Train/test split
print("\n4. Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Training set: {X_train.shape[0]} samples")
print(f"   Test set: {X_test.shape[0]} samples")

# Apply StandardScaler
print("\n5. Applying StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)

print("\n6. Training models...")
print("=" * 80)

# Model 1: Single-layer NN with Adam (NO pos_weight)
print("\nModel 1: Single-layer NN (Adam) - NO class weighting")
model_adam = SingleLayerNN()
optimizer_adam = torch.optim.Adam(model_adam.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()  # NO pos_weight

# Add OneCycleLR scheduler
scheduler_adam = torch.optim.lr_scheduler.OneCycleLR(
    optimizer_adam,
    max_lr=0.01,
    steps_per_epoch=1,
    epochs=100,
    pct_start=0.3,
    anneal_strategy='cos'
)

# Training loop
model_adam.train()
for epoch in range(100):
    optimizer_adam.zero_grad()
    outputs = model_adam(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer_adam.step()
    scheduler_adam.step()

    if (epoch + 1) % 20 == 0:
        print(f"   Epoch {epoch+1}/100, Loss: {loss.item():.4f}")

# Evaluate
model_adam.eval()
with torch.no_grad():
    test_pred = torch.sigmoid(model_adam(X_test_tensor)).numpy().flatten()
    print(f"   Test predictions range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")

# Save
model_path = results_dir / 'test5_single_layer_nn_adam_unweighted_scaled.pkl'
with open(model_path, 'wb') as f:
    pickle.dump({'model': model_adam, 'scaler': scaler}, f)
print(f"   Saved to: {model_path.name}")

# Model 2: Single-layer NN with L-BFGS (NO pos_weight)
print("\nModel 2: Single-layer NN (L-BFGS) - NO class weighting")
model_lbfgs = SingleLayerNN()
optimizer_lbfgs = torch.optim.LBFGS(model_lbfgs.parameters(), lr=0.1, max_iter=20)

def closure():
    optimizer_lbfgs.zero_grad()
    outputs = model_lbfgs(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    return loss

model_lbfgs.train()
for epoch in range(100):
    loss = optimizer_lbfgs.step(closure)
    if (epoch + 1) % 20 == 0:
        print(f"   Epoch {epoch+1}/100, Loss: {loss.item():.4f}")

# Evaluate
model_lbfgs.eval()
with torch.no_grad():
    test_pred = torch.sigmoid(model_lbfgs(X_test_tensor)).numpy().flatten()
    print(f"   Test predictions range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")

# Save
model_path = results_dir / 'test5_single_layer_nn_lbfgs_unweighted_scaled.pkl'
with open(model_path, 'wb') as f:
    pickle.dump({'model': model_lbfgs, 'scaler': scaler}, f)
print(f"   Saved to: {model_path.name}")

# Model 3: LogisticRegression (NO class_weight) - Exact notebook 04 match
print("\nModel 3: Logistic Regression - NO class weighting (matches notebook 04)")
logreg = LogisticRegression(random_state=42, max_iter=1000)  # NO class_weight parameter
logreg.fit(X_train_scaled, y_train)

# Evaluate
test_pred = logreg.predict_proba(X_test_scaled)[:, 1]
print(f"   Test predictions range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")

# Save
model_path = results_dir / 'test5_logreg_unweighted_scaled.pkl'
with open(model_path, 'wb') as f:
    pickle.dump({'model': logreg, 'scaler': scaler}, f)
print(f"   Saved to: {model_path.name}")

# Model 4: SimpleNN (use_class_weights=False, NO pos_weight)
print("\nModel 4: SimpleNN - NO class weighting (experimental)")
simple_nn = SimpleNN(
    input_dim=2,
    hidden_dims=(128, 64, 32),
    dropout_rate=0.3,
    use_class_weights=False  # Key difference from Test 4
)

# Train
optimizer_simple = torch.optim.Adam(simple_nn.parameters(), lr=0.001)
simple_nn.train()
for epoch in range(100):
    optimizer_simple.zero_grad()
    outputs = simple_nn(X_train_tensor)
    # Flatten y_train_tensor to match outputs shape
    loss = criterion(outputs, y_train_tensor.squeeze())
    loss.backward()
    optimizer_simple.step()

    if (epoch + 1) % 20 == 0:
        print(f"   Epoch {epoch+1}/100, Loss: {loss.item():.4f}")

# Evaluate
simple_nn.eval()
with torch.no_grad():
    test_pred = torch.sigmoid(simple_nn(X_test_tensor)).numpy().flatten()
    print(f"   Test predictions range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")

# Save
model_path = results_dir / 'test5_simple_nn_unweighted_scaled.pkl'
with open(model_path, 'wb') as f:
    pickle.dump({'model': simple_nn, 'scaler': scaler}, f)
print(f"   Saved to: {model_path.name}")

print("\n" + "=" * 80)
print("Test 5 training complete!")
print("=" * 80)
print(f"\nAll models saved to: {results_dir}")
print("\nNext step: Run notebook 22 with Test 5 models to validate correlation with empirical frequencies")
