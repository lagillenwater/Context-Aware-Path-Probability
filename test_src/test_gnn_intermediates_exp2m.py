"""
Experiment 2M: Graph Neural Network on Intermediate Subgraphs

Test whether modeling connectivity between intermediates can break r=0.80 ceiling.

Instead of aggregating intermediates independently, build subgraph of
intermediate genes and learn from their GiG interactions using GNN.

Training: Perm 1 (or mean of perms 1-K)
Validation: Mean of perms 11-20

Date: 2025-11-05
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
import warnings
warnings.filterwarnings('ignore')

repo_dir = Path(__file__).parent.parent
sys.path.append(str(repo_dir / 'src'))
data_dir = repo_dir / 'data'
results_dir = repo_dir / 'results' / 'hierarchical_prediction'
results_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("EXPERIMENT 2M: GNN ON INTERMEDIATE SUBGRAPHS")
print("Model intermediate connectivity for pathway prediction")
print("="*80)


def load_edge_matrix(edge_abbrev, perm_num):
    edge_file = data_dir / 'permutations' / f'{perm_num:03d}.hetmat' / 'edges' / f'{edge_abbrev}.sparse.npz'
    if edge_file.exists():
        return sp.load_npz(str(edge_file)).astype(np.int32)
    return None


def sample_pairs_stratified(pathway_matrix, n_samples=10000, random_state=42):
    np.random.seed(random_state)
    sources_nonzero, targets_nonzero = pathway_matrix.nonzero()
    n_nonzero = len(sources_nonzero)

    n_nonzero_sample = min(int(n_samples * 0.5), n_nonzero)
    if n_nonzero > 0:
        idx_nonzero = np.random.choice(n_nonzero, n_nonzero_sample, replace=False)
        sampled_sources = list(sources_nonzero[idx_nonzero])
        sampled_targets = list(targets_nonzero[idx_nonzero])
    else:
        sampled_sources = []
        sampled_targets = []

    n_random = n_samples - len(sampled_sources)
    random_sources = np.random.randint(0, pathway_matrix.shape[0], n_random)
    random_targets = np.random.randint(0, pathway_matrix.shape[1], n_random)

    sampled_sources.extend(random_sources)
    sampled_targets.extend(random_targets)
    return list(zip(sampled_sources, sampled_targets))


print("\n" + "="*80)
print("PHASE 1: Load Data and Train 2-Hop Models")
print("="*80)

print("\nLoading perm 0 topology...")
CbG_0 = load_edge_matrix('CbG', 0)
GiG_0 = load_edge_matrix('GiG', 0)
GpPW_0 = load_edge_matrix('GpPW', 0)

CbGiG_0 = CbG_0 @ GiG_0
CbGiGpPW_0 = CbGiG_0 @ GpPW_0

compound_degrees = np.array(CbG_0.sum(axis=1)).flatten()
gene_degrees = np.array(GiG_0.sum(axis=1)).flatten()
pathway_degrees = np.array(GpPW_0.sum(axis=0)).flatten()

print("\nSampling pairs...")
pairs = sample_pairs_stratified(CbGiGpPW_0, n_samples=10000, random_state=42)
print(f"  Sampled {len(pairs)} pairs")

print("\nTraining 2-hop models on perm 1...")
CbG_1 = load_edge_matrix('CbG', 1)
GiG_1 = load_edge_matrix('GiG', 1)
GpPW_1 = load_edge_matrix('GpPW', 1)

CbGiG_1 = CbG_1 @ GiG_1
GiGpPW_1 = GiG_1 @ GpPW_1

# Train CbGiG model
print("  Training CbGiG model...")
CbGiG_1_coo = CbGiG_1.tocoo()
X_CbGiG = np.array([[compound_degrees[i], gene_degrees[j]]
                     for i, j in zip(CbGiG_1_coo.row, CbGiG_1_coo.col)])
y_CbGiG = np.array(CbGiG_1_coo.data, dtype=float)

model_CbGiG = LinearRegression()
model_CbGiG.fit(X_CbGiG, y_CbGiG)
print(f"    Trained on {len(X_CbGiG)} edges")

# Train GiGpPW model
print("  Training GiGpPW model...")
GiGpPW_1_coo = GiGpPW_1.tocoo()
X_GiGpPW = np.array([[gene_degrees[i], pathway_degrees[j]]
                      for i, j in zip(GiGpPW_1_coo.row, GiGpPW_1_coo.col)])
y_GiGpPW = np.array(GiGpPW_1_coo.data, dtype=float)

model_GiGpPW = LinearRegression()
model_GiGpPW.fit(X_GiGpPW, y_GiGpPW)
print(f"    Trained on {len(X_GiGpPW)} edges")


print("\n" + "="*80)
print("PHASE 2: Build Intermediate Subgraphs")
print("="*80)

CbGiG_0_lil = CbGiG_0.tolil()
GpPW_0_csr = GpPW_0.tocsr()
GiG_0_lil = GiG_0.tolil()

print("\nBuilding subgraphs for each pair...")
graphs = []

for idx, (C, PW) in enumerate(pairs):
    if idx % 2000 == 0:
        print(f"  Processed {idx}/{len(pairs)} pairs...")

    deg_C = compound_degrees[C]
    deg_PW = pathway_degrees[PW]

    # Identify intermediates
    genes_to_PW = GpPW_0_csr[:, PW].nonzero()[0]
    intermediates = [G2 for G2 in genes_to_PW if CbGiG_0_lil[C, G2] > 0]

    if len(intermediates) == 0:
        # No intermediates - create dummy graph with single node
        x = torch.zeros((1, 5), dtype=torch.float)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        context = torch.tensor([deg_C, deg_PW, 0, 0], dtype=torch.float)
        graphs.append(Data(x=x, edge_index=edge_index, context=context))
        continue

    # Create node-to-index mapping
    node_to_idx = {gene: i for i, gene in enumerate(intermediates)}

    # Extract node features for each intermediate
    node_features = []
    for G2 in intermediates:
        deg_G2 = gene_degrees[G2]

        # Predict 2-hop paths
        pred_CbGiG = model_CbGiG.predict([[deg_C, deg_G2]])[0]
        pred_GiGpPW = model_GiGpPW.predict([[deg_G2, deg_PW]])[0]

        # Count connections to other intermediates
        degree_in_subgraph = sum(1 for other in intermediates
                                 if other != G2 and GiG_0_lil[G2, other] > 0)

        features = [
            deg_G2 / 1000.0,  # Normalize
            degree_in_subgraph / max(len(intermediates), 1),
            pred_CbGiG,
            pred_GiGpPW,
            float(CbGiG_0_lil[C, G2]),
        ]
        node_features.append(features)

    # Extract edges between intermediates
    edges = []
    for i, G1 in enumerate(intermediates):
        for j, G2 in enumerate(intermediates):
            if i != j and GiG_0_lil[G1, G2] > 0:
                edges.append([i, j])

    # Convert to tensors
    x = torch.tensor(node_features, dtype=torch.float)

    if len(edges) > 0:
        edge_index = torch.tensor(edges, dtype=torch.long).t()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    # Global context
    context = torch.tensor([
        deg_C / 1000.0,
        deg_PW / 1000.0,
        len(intermediates) / 100.0,
        np.mean([f[0] for f in node_features])
    ], dtype=torch.float)

    graphs.append(Data(x=x, edge_index=edge_index, context=context))

print(f"\nCreated {len(graphs)} subgraphs")


print("\n" + "="*80)
print("PHASE 3: Compute Targets and Attach to Graphs")
print("="*80)

print("\nComputing training target (perm 1)...")
CbGiGpPW_1 = CbGiG_1 @ GpPW_1
CbGiGpPW_1_lil = CbGiGpPW_1.tolil()
y_train_target = np.array([float(CbGiGpPW_1_lil[C, PW]) for C, PW in pairs])
print(f"  Mean: {y_train_target.mean():.3f}")

print("\nComputing validation target (mean of perms 11-20)...")
val_counts = []
for perm in range(11, 21):
    print(f"  Loading perm {perm}...")
    CbG_p = load_edge_matrix('CbG', perm)
    GiG_p = load_edge_matrix('GiG', perm)
    GpPW_p = load_edge_matrix('GpPW', perm)
    CbGiG_p = CbG_p @ GiG_p
    CbGiGpPW_p = CbGiG_p @ GpPW_p
    CbGiGpPW_p_lil = CbGiGpPW_p.tolil()
    counts = np.array([float(CbGiGpPW_p_lil[C, PW]) for C, PW in pairs])
    val_counts.append(counts)

y_val_target = np.mean(val_counts, axis=0)
print(f"  Mean: {y_val_target.mean():.3f}")

# Attach targets to graphs
for i, graph in enumerate(graphs):
    graph.y_train = torch.tensor([y_train_target[i]], dtype=torch.float)
    graph.y_val = torch.tensor([y_val_target[i]], dtype=torch.float)


print("\n" + "="*80)
print("PHASE 4: Define GNN Model")
print("="*80)


class IntermediateGNN(nn.Module):
    def __init__(self, node_feature_dim, context_dim):
        super().__init__()

        # Node feature encoder
        self.node_encoder = nn.Linear(node_feature_dim, 32)

        # Graph convolution layers
        self.conv1 = GCNConv(32, 64)
        self.conv2 = GCNConv(64, 32)
        self.conv3 = GCNConv(32, 16)

        # Context encoder
        self.context_encoder = nn.Linear(context_dim, 16)

        # Combine graph embedding + context
        self.predictor = nn.Sequential(
            nn.Linear(16 + 16 + 16, 64),  # mean + max + context
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, data):
        x, edge_index, batch, context = data.x, data.edge_index, data.batch, data.context

        # Encode nodes
        x = self.node_encoder(x)
        x = F.relu(x)

        # Message passing
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)

        # Aggregate to graph-level (use both mean and max pooling)
        graph_mean = global_mean_pool(x, batch)
        graph_max = global_max_pool(x, batch)

        # Encode context
        context_embedding = self.context_encoder(context)
        context_embedding = F.relu(context_embedding)

        # Combine and predict
        combined = torch.cat([graph_mean, graph_max, context_embedding], dim=1)
        return self.predictor(combined).squeeze()


print("\nGNN Architecture:")
print("  Node encoder: 5 -> 32")
print("  GCN layers: 32 -> 64 -> 32 -> 16")
print("  Pooling: mean + max")
print("  Context encoder: 4 -> 16")
print("  Predictor: 48 -> 64 -> 32 -> 1")


print("\n" + "="*80)
print("PHASE 5: Train GNN")
print("="*80)

# Split data
train_graphs = [graphs[i] for i in range(len(graphs)) if i % 5 != 0]
test_graphs = [graphs[i] for i in range(len(graphs)) if i % 5 == 0]

print(f"\nTrain: {len(train_graphs)}, Test: {len(test_graphs)}")

# Create data loaders
train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)

# Initialize model
model = IntermediateGNN(node_feature_dim=5, context_dim=4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
criterion = nn.MSELoss()

# Train
print("\nTraining...")
for epoch in range(50):
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()
        pred = model(batch)
        loss = criterion(pred, batch.y_train)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1}/50, Loss: {total_loss/len(train_loader):.4f}")

# Evaluate
print("\nEvaluating...")
model.eval()
predictions_train_target = []
predictions_val_target = []
true_train = []
true_val = []

with torch.no_grad():
    for batch in test_loader:
        pred = model(batch).numpy()
        predictions_train_target.extend(pred)
        predictions_val_target.extend(pred)  # Same predictions
        true_train.extend(batch.y_train.numpy())
        true_val.extend(batch.y_val.numpy())

predictions_train_target = np.array(predictions_train_target)
predictions_val_target = np.array(predictions_val_target)
true_train = np.array(true_train)
true_val = np.array(true_val)

r_train = pearsonr(predictions_train_target, true_train)[0]
r_val = pearsonr(predictions_val_target, true_val)[0]
mae_val = mean_absolute_error(true_val, predictions_val_target)

print(f"\n  r vs perm 1: {r_train:.4f}")
print(f"  r vs mean(perms 11-20): {r_val:.4f}")
print(f"  MAE vs validation: {mae_val:.3f}")


print("\n" + "="*80)
print("PHASE 6: Summary")
print("="*80)

print(f"\nTarget correlation: {pearsonr(y_train_target, y_val_target)[0]:.4f}")
print(f"\nGNN Performance:")
print(f"  r vs validation: {r_val:.4f}")
print(f"  MAE: {mae_val:.3f}")

if r_val > 0.95:
    print("\nSUCCESS: Achieved r > 0.95!")
elif r_val > 0.85:
    print("\nPROMISING: Improvement over baseline (r=0.80)")
    print("Consider testing with multiple permutations")
else:
    print("\nNo significant improvement over linear baseline")

# Save results
results_df = pd.DataFrame({
    'model': ['GNN'],
    'r_validation': [r_val],
    'mae_validation': [mae_val],
    'r_training': [r_train]
})
results_df.to_csv(results_dir / 'experiment2m_results.csv', index=False)
print(f"\nSaved: {results_dir / 'experiment2m_results.csv'}")

print("\n" + "="*80)
print("EXPERIMENT 2M COMPLETE")
print("="*80)
