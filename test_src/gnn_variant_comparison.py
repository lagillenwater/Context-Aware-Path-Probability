"""
Test Graph Neural Network for pathway count prediction.

This proof-of-concept tests whether GNN can improve beyond current r=0.787 ceiling
by capturing graph structure beyond just degree features.

Key differences from previous models:
- Pairs are NOT independent - nodes can appear in multiple pairs
- GNN learns embeddings from graph topology (neighbors, clustering, etc.)
- Message passing aggregates information from local neighborhood

Architecture:
1. Build graph from permuted edges (avoid data leakage)
2. Node features: degree
3. GNN: 2-3 layers of message passing
4. For each query pair (src, tgt): concatenate embeddings and predict count

Usage:
    python test_src/gnn_variant_comparison.py CbGpPW

References:
    - docs/2025-11-11_NONLINEAR_MEAN_MODELS.md (current best: r=0.787)
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
import sys
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats

repo_dir = Path(__file__).parent.parent
cache_dir = repo_dir / '.cache'
mpl_cache_dir = cache_dir / 'matplotlib'
mpl_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('XDG_CACHE_HOME', str(cache_dir))
os.environ.setdefault('MPLCONFIGDIR', str(mpl_cache_dir))
sys.path.insert(0, str(repo_dir))

import matplotlib.pyplot as plt

from test_src.validate_mean_variance_prediction import (
    load_permuted_edge_matrices,
    sample_pairs,
    extract_degree_features,
    compute_pathway_counts,
    evaluate_z_scores
)


def list_available_permutation_ids(data_dir):
    """Return sorted available permutation IDs from data/permutations."""
    perm_dir = data_dir / 'permutations'
    if not perm_dir.exists():
        return []
    perm_ids = []
    for child in perm_dir.glob('*.hetmat'):
        try:
            perm_ids.append(int(child.stem))
        except ValueError:
            continue
    return sorted(set(perm_ids))


def resolve_permutation_splits(available, train_perms=None, val_perms=None, test_perms=None):
    """
    Resolve train/val/test permutation splits.

    If explicit splits are provided, validates them.
    Otherwise uses canonical splits when available; falls back to local IDs.
    """
    if not available:
        raise FileNotFoundError("No permutations found in data/permutations.")

    if train_perms or val_perms or test_perms:
        if not (train_perms and val_perms and test_perms):
            raise ValueError(
                "Provide all of --train-perms, --val-perms, and --test-perms, or none."
            )
        all_requested = [*train_perms, *val_perms, *test_perms]
        missing = [perm for perm in all_requested if perm not in available]
        if missing:
            raise FileNotFoundError(f"Requested permutations missing: {sorted(set(missing))}")
        return train_perms, val_perms, test_perms

    canonical_train = [0, 1, 2, 3, 4]
    canonical_val = [10, 11, 12, 13, 14]
    canonical_test = [15, 16, 17, 18, 19, 20]
    canonical_all = canonical_train + canonical_val + canonical_test
    if all(perm in available for perm in canonical_all):
        return canonical_train, canonical_val, canonical_test

    train_count = min(5, max(1, len(available) - 2))
    train = available[:train_count]
    holdout = available[train_count:]
    if not holdout:
        holdout = available[-1:]
    if len(holdout) == 1:
        val = holdout
        test = holdout
    else:
        split = max(1, len(holdout) // 2)
        val = holdout[:split]
        test = holdout[split:]
        if not test:
            test = val[-1:]
    return train, val, test


class SimpleGCNLayer(nn.Module):
    """
    Simple Graph Convolutional Layer.

    Implements: H^(l+1) = σ(D^(-1/2) A D^(-1/2) H^(l) W^(l))

    Where:
    - A: Adjacency matrix with self-loops
    - D: Degree matrix
    - H: Node features
    - W: Learnable weight matrix
    """

    def __init__(self, in_dim, out_dim):
        super(SimpleGCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index):
        """
        Args:
            x: Node features (n_nodes, in_dim)
            edge_index: Edge list (2, n_edges)

        Returns:
            out: Updated node features (n_nodes, out_dim)
        """
        n_nodes = x.size(0)

        row, col = edge_index
        edge_weight = torch.ones(edge_index.size(1), device=x.device)

        self_loop_edge_index = torch.arange(n_nodes, device=x.device).unsqueeze(0).repeat(2, 1)
        edge_index_with_loops = torch.cat([edge_index, self_loop_edge_index], dim=1)
        edge_weight_with_loops = torch.cat([edge_weight, torch.ones(n_nodes, device=x.device)])

        row_full, col_full = edge_index_with_loops

        degree = torch.zeros(n_nodes, device=x.device)
        degree = degree.scatter_add_(0, row_full, edge_weight_with_loops)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0

        norm = degree_inv_sqrt[row_full] * edge_weight_with_loops * degree_inv_sqrt[col_full]

        support = torch.mm(x, self.weight)

        out = torch.zeros(n_nodes, support.size(1), device=x.device)
        out = out.scatter_add_(0, col_full.unsqueeze(1).expand(-1, support.size(1)),
                               norm.unsqueeze(1) * support[row_full])

        out = out + self.bias

        return out


class PathwayGNN(nn.Module):
    """
    Graph Neural Network for pathway count prediction.

    Uses message passing to learn node embeddings from graph structure,
    then predicts pathway counts from source-target embedding pairs.
    """

    def __init__(self, hidden_dim=64, num_layers=2, dropout=0.2):
        super(PathwayGNN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(SimpleGCNLayer(1, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(SimpleGCNLayer(hidden_dim, hidden_dim))

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        self.dropout = dropout

    def forward(self, x, edge_index, query_pairs):
        """
        Args:
            x: Node features (n_nodes, 1) - degrees
            edge_index: Graph edges (2, n_edges)
            query_pairs: Pairs to predict (n_pairs, 2) - indices into x

        Returns:
            predictions: (n_pairs,) predicted pathway counts
        """
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        src_embed = x[query_pairs[:, 0]]
        tgt_embed = x[query_pairs[:, 1]]
        pair_embed = torch.cat([src_embed, tgt_embed], dim=1)

        return self.predictor(pair_embed).squeeze()


def build_graph_from_edges(edge1_matrix, edge2_matrix):
    """
    Build PyTorch Geometric graph from edge matrices.

    For CbGpPW:
        edge1: Compound-binds-Gene (CbG)
        edge2: Gene-participates-Pathway (GpPW)

    Creates heterogeneous graph with all nodes and edges.

    Args:
        edge1_matrix: Sparse matrix (n_src, n_intermediate)
        edge2_matrix: Sparse matrix (n_intermediate, n_tgt)

    Returns:
        node_features: Tensor (n_nodes, 1) - node degrees
        edge_index: Tensor (2, n_edges) - edge list
        node_id_to_index: Dict mapping (node_type, node_id) to graph index
        index_to_node_id: Dict mapping graph index to (node_type, node_id)
    """
    n_src = edge1_matrix.shape[0]
    n_intermediate = edge1_matrix.shape[1]
    n_tgt = edge2_matrix.shape[1]

    node_id_to_index = {}
    index_to_node_id = {}
    node_degrees = []

    current_index = 0

    for src_id in range(n_src):
        node_id_to_index[('src', src_id)] = current_index
        index_to_node_id[current_index] = ('src', src_id)
        degree = edge1_matrix[src_id].nnz
        node_degrees.append(degree)
        current_index += 1

    for int_id in range(n_intermediate):
        node_id_to_index[('int', int_id)] = current_index
        index_to_node_id[current_index] = ('int', int_id)
        degree = edge1_matrix[:, int_id].nnz + edge2_matrix[int_id, :].nnz
        node_degrees.append(degree)
        current_index += 1

    for tgt_id in range(n_tgt):
        node_id_to_index[('tgt', tgt_id)] = current_index
        index_to_node_id[current_index] = ('tgt', tgt_id)
        degree = edge2_matrix[:, tgt_id].nnz
        node_degrees.append(degree)
        current_index += 1

    edge1_coo = edge1_matrix.tocoo()
    edge2_coo = edge2_matrix.tocoo()

    edge_list = []

    for i in range(edge1_coo.nnz):
        src_idx = node_id_to_index[('src', edge1_coo.row[i])]
        int_idx = node_id_to_index[('int', edge1_coo.col[i])]
        edge_list.append([src_idx, int_idx])
        edge_list.append([int_idx, src_idx])

    for i in range(edge2_coo.nnz):
        int_idx = node_id_to_index[('int', edge2_coo.row[i])]
        tgt_idx = node_id_to_index[('tgt', edge2_coo.col[i])]
        edge_list.append([int_idx, tgt_idx])
        edge_list.append([tgt_idx, int_idx])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    node_features = torch.tensor(node_degrees, dtype=torch.float).unsqueeze(1)

    return node_features, edge_index, node_id_to_index, index_to_node_id


def pairs_to_graph_indices(pairs, node_id_to_index):
    """
    Convert (src_id, tgt_id) pairs to graph node indices.

    Args:
        pairs: Array (n_pairs, 2) of (src_node_id, tgt_node_id)
        node_id_to_index: Dict mapping (node_type, node_id) to graph index

    Returns:
        query_pairs: Tensor (n_pairs, 2) of graph indices
    """
    query_pairs = []
    for src_id, tgt_id in pairs:
        src_idx = node_id_to_index[('src', src_id)]
        tgt_idx = node_id_to_index[('tgt', tgt_id)]
        query_pairs.append([src_idx, tgt_idx])

    return torch.tensor(query_pairs, dtype=torch.long)


def train_gnn(
    pairs,
    edge1_type,
    edge2_type,
    data_dir,
    train_perms,
    val_perms,
    n_samples=10000,
    random_state=42,
):
    """
    Train GNN on permuted graphs to predict pathway counts.

    Args:
        pairs: Sampled node pairs (n_pairs, 2)
        edge1_type: First edge type (e.g., 'CbG')
        edge2_type: Second edge type (e.g., 'GpPW')
        data_dir: Path to data directory
        n_samples: Number of pairs to sample
        random_state: Random seed

    Returns:
        model: Trained GNN
        train_results: Training metrics
    """
    print("Building graphs from permutations...")

    base_perm = train_perms[0]
    print(f"  Training permutations: {train_perms}")
    print(f"  Validation permutations: {val_perms}")
    print(f"  Building graph from perm {base_perm} (shared structure across perms)...")
    edge1_perm0, edge2_perm0 = load_permuted_edge_matrices(edge1_type, edge2_type, base_perm, data_dir)

    node_features, edge_index, node_id_to_index, index_to_node_id = build_graph_from_edges(
        edge1_perm0, edge2_perm0
    )

    query_pairs = pairs_to_graph_indices(pairs, node_id_to_index)

    print(f"  Graph: {node_features.shape[0]} nodes, {edge_index.shape[1]} edges")
    print(f"  Query pairs: {query_pairs.shape[0]}")

    print(f"Computing training targets (mean of perms {train_perms})...")
    counts_train = []
    for perm in train_perms:
        edge1, edge2 = load_permuted_edge_matrices(edge1_type, edge2_type, perm, data_dir)
        counts = compute_pathway_counts(pairs, edge1, edge2)
        counts_train.append(counts)
    counts_train = np.column_stack(counts_train)
    mu_train = np.mean(counts_train, axis=1)
    mu_train_t = torch.FloatTensor(mu_train)

    print(f"Computing validation targets (mean of perms {val_perms})...")
    counts_val = []
    for perm in val_perms:
        edge1, edge2 = load_permuted_edge_matrices(edge1_type, edge2_type, perm, data_dir)
        counts = compute_pathway_counts(pairs, edge1, edge2)
        counts_val.append(counts)
    counts_val = np.column_stack(counts_val)
    mu_val = np.mean(counts_val, axis=1)
    mu_val_t = torch.FloatTensor(mu_val)

    print("Training GNN...")
    model = PathwayGNN(hidden_dim=64, num_layers=2, dropout=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()

    best_val_loss = np.inf
    patience = 30
    patience_counter = 0
    best_state = None

    for epoch in range(500):
        model.train()
        optimizer.zero_grad()

        pred_train = model(node_features, edge_index, query_pairs)
        loss = criterion(pred_train, mu_train_t)

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_val = model(node_features, edge_index, query_pairs)
            val_loss = criterion(pred_val, mu_val_t).item()
            val_r = np.corrcoef(mu_val, pred_val.numpy())[0, 1]

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}: train_loss={loss.item():.4f}, val_loss={val_loss:.4f}, val_r={val_r:.4f}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred_val = model(node_features, edge_index, query_pairs)
    val_r = np.corrcoef(mu_val, pred_val.numpy())[0, 1]
    print(f"  Best validation r={val_r:.4f}")

    return model, {
        'node_features': node_features,
        'edge_index': edge_index,
        'query_pairs': query_pairs,
        'val_r': val_r
    }


def analyze_embeddings(model, graph_data):
    """
    Analyze if embeddings are degree-specific or permutation-specific.

    Args:
        model: Trained GNN
        graph_data: Dict with node_features, edge_index, query_pairs

    Returns:
        embedding_analysis: Dict with analysis results
    """
    node_features = graph_data['node_features']
    edge_index = graph_data['edge_index']

    model.eval()
    with torch.no_grad():
        x = node_features
        for i, layer in enumerate(model.layers):
            x = layer(x, edge_index)
            if i < len(model.layers) - 1:
                x = F.relu(x)

        embeddings = x.numpy()

    degrees = node_features.numpy().flatten()

    degree_bins = [0, 10, 50, 100, 500, np.inf]
    bin_labels = ['0-10', '10-50', '50-100', '100-500', '500+']

    print("\nEmbedding Analysis:")
    print("  Checking if embeddings cluster by degree...")

    degree_bin_indices = np.digitize(degrees, degree_bins[:-1])

    for i, label in enumerate(bin_labels, start=1):
        mask = degree_bin_indices == i
        if mask.sum() > 0:
            bin_embeddings = embeddings[mask]
            avg_norm = np.linalg.norm(bin_embeddings, axis=1).mean()
            print(f"  Degree {label}: {mask.sum()} nodes, avg embedding norm: {avg_norm:.3f}")

    return {
        'embeddings': embeddings,
        'degrees': degrees
    }


def evaluate_gnn(model, graph_data, pairs, edge1_type, edge2_type, data_dir, test_perms):
    """
    Evaluate GNN on test permutations.

    Args:
        model: Trained GNN
        graph_data: Dict with node_features, edge_index, query_pairs
        pairs: Node pairs
        edge1_type: First edge type
        edge2_type: Second edge type
        data_dir: Path to data directory

    Returns:
        results_df: DataFrame with test results
    """
    node_features = graph_data['node_features']
    edge_index = graph_data['edge_index']
    query_pairs = graph_data['query_pairs']

    embedding_analysis = analyze_embeddings(model, graph_data)

    print(f"\nEvaluating on test permutations {test_perms}...")

    model.eval()
    with torch.no_grad():
        mu_pred = model(node_features, edge_index, query_pairs).numpy()

    counts_test = []
    for perm in test_perms:
        edge1, edge2 = load_permuted_edge_matrices(edge1_type, edge2_type, perm, data_dir)
        counts = compute_pathway_counts(pairs, edge1, edge2)
        counts_test.append(counts)
    counts_test = np.column_stack(counts_test)

    results = []
    for idx, perm in enumerate(test_perms):
        counts = counts_test[:, idx]
        r_mean = np.corrcoef(counts, mu_pred)[0, 1]

        results.append({
            'test_perm': perm,
            'model_type': 'gnn_single_perm',
            'r_mean': r_mean
        })

    results_df = pd.DataFrame(results)

    mean_r = results_df['r_mean'].mean()
    print(f"\nTest Results (GNN on single training permutation graph):")
    print(f"  Mean r across test perms: {mean_r:.4f}")
    print(f"  Comparison to baseline linear: 0.787")
    print(f"  Comparison to best (RF/Hetero NN): 0.778")

    return results_df


def train_gnn_multi_perm(
    pairs,
    edge1_type,
    edge2_type,
    data_dir,
    train_perms,
    val_perms,
    n_samples=10000,
    random_state=42,
):
    """
    Train GNN on multiple permutations, aggregating embeddings.

    Instead of using graph from perm 0 only, we:
    1. Build graphs from perms 0-4
    2. For each perm, compute node embeddings
    3. Average embeddings across permutations
    4. Use averaged embeddings for prediction

    This captures degree-specific patterns that are robust across permutations.

    Args:
        pairs: Sampled node pairs
        edge1_type: First edge type
        edge2_type: Second edge type
        data_dir: Path to data directory
        n_samples: Number of pairs
        random_state: Random seed

    Returns:
        model: Trained GNN
        multi_perm_data: Data needed for evaluation
    """
    print(f"Building graphs from training permutations {train_perms}...")

    graphs = []
    node_id_mappings = []

    for perm in train_perms:
        print(f"  Building graph from perm {perm}...")
        edge1, edge2 = load_permuted_edge_matrices(edge1_type, edge2_type, perm, data_dir)

        node_features, edge_index, node_id_to_index, index_to_node_id = build_graph_from_edges(
            edge1, edge2
        )

        graphs.append({
            'node_features': node_features,
            'edge_index': edge_index,
            'perm': perm
        })

        if perm == 0:
            node_id_mappings = (node_id_to_index, index_to_node_id)

    node_id_to_index, index_to_node_id = node_id_mappings

    query_pairs = pairs_to_graph_indices(pairs, node_id_to_index)

    print(f"  Query pairs: {query_pairs.shape[0]}")
    print(f"  Training on {len(graphs)} permutations")

    print(f"Computing training targets (mean of perms {train_perms})...")
    counts_train = []
    for perm in train_perms:
        edge1, edge2 = load_permuted_edge_matrices(edge1_type, edge2_type, perm, data_dir)
        counts = compute_pathway_counts(pairs, edge1, edge2)
        counts_train.append(counts)
    counts_train = np.column_stack(counts_train)
    mu_train = np.mean(counts_train, axis=1)
    mu_train_t = torch.FloatTensor(mu_train)

    print(f"Computing validation targets (mean of perms {val_perms})...")
    counts_val = []
    for perm in val_perms:
        edge1, edge2 = load_permuted_edge_matrices(edge1_type, edge2_type, perm, data_dir)
        counts = compute_pathway_counts(pairs, edge1, edge2)
        counts_val.append(counts)
    counts_val = np.column_stack(counts_val)
    mu_val = np.mean(counts_val, axis=1)
    mu_val_t = torch.FloatTensor(mu_val)

    print("Training GNN with multi-permutation embeddings...")
    model = PathwayGNN(hidden_dim=64, num_layers=2, dropout=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()

    best_val_loss = np.inf
    patience = 30
    patience_counter = 0
    best_state = None

    for epoch in range(500):
        model.train()
        optimizer.zero_grad()

        perm_embeddings = []
        for graph in graphs:
            node_features = graph['node_features']
            edge_index = graph['edge_index']

            with torch.no_grad():
                x = node_features
                for i, layer in enumerate(model.layers):
                    x = layer(x, edge_index)
                    if i < len(model.layers) - 1:
                        x = F.relu(x)
                perm_embeddings.append(x)

        avg_embeddings = torch.stack(perm_embeddings).mean(dim=0)

        src_embed = avg_embeddings[query_pairs[:, 0]]
        tgt_embed = avg_embeddings[query_pairs[:, 1]]
        pair_embed = torch.cat([src_embed, tgt_embed], dim=1)

        pred_train = model.predictor(pair_embed).squeeze()
        loss = criterion(pred_train, mu_train_t)

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            perm_embeddings_val = []
            for graph in graphs:
                node_features = graph['node_features']
                edge_index = graph['edge_index']

                x = node_features
                for i, layer in enumerate(model.layers):
                    x = layer(x, edge_index)
                    if i < len(model.layers) - 1:
                        x = F.relu(x)
                perm_embeddings_val.append(x)

            avg_embeddings_val = torch.stack(perm_embeddings_val).mean(dim=0)

            src_embed_val = avg_embeddings_val[query_pairs[:, 0]]
            tgt_embed_val = avg_embeddings_val[query_pairs[:, 1]]
            pair_embed_val = torch.cat([src_embed_val, tgt_embed_val], dim=1)

            pred_val = model.predictor(pair_embed_val).squeeze()
            val_loss = criterion(pred_val, mu_val_t).item()
            val_r = np.corrcoef(mu_val, pred_val.numpy())[0, 1]

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}: train_loss={loss.item():.4f}, val_loss={val_loss:.4f}, val_r={val_r:.4f}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        perm_embeddings_val = []
        for graph in graphs:
            node_features = graph['node_features']
            edge_index = graph['edge_index']

            x = node_features
            for i, layer in enumerate(model.layers):
                x = layer(x, edge_index)
                if i < len(model.layers) - 1:
                    x = F.relu(x)
            perm_embeddings_val.append(x)

        avg_embeddings_val = torch.stack(perm_embeddings_val).mean(dim=0)

        src_embed_val = avg_embeddings_val[query_pairs[:, 0]]
        tgt_embed_val = avg_embeddings_val[query_pairs[:, 1]]
        pair_embed_val = torch.cat([src_embed_val, tgt_embed_val], dim=1)

        pred_val = model.predictor(pair_embed_val).squeeze()
        val_r = np.corrcoef(mu_val, pred_val.numpy())[0, 1]

    print(f"  Best validation r={val_r:.4f}")

    return model, {
        'graphs': graphs,
        'query_pairs': query_pairs,
        'val_r': val_r
    }


def evaluate_gnn_multi_perm(
    model,
    multi_perm_data,
    pairs,
    edge1_type,
    edge2_type,
    data_dir,
    test_perms,
):
    """
    Evaluate multi-permutation GNN on test set.

    Args:
        model: Trained GNN
        multi_perm_data: Dict with graphs and query_pairs
        pairs: Node pairs
        edge1_type: First edge type
        edge2_type: Second edge type
        data_dir: Path to data directory

    Returns:
        results_df: DataFrame with test results
    """
    graphs = multi_perm_data['graphs']
    query_pairs = multi_perm_data['query_pairs']

    print(f"\nEvaluating multi-perm GNN on test permutations {test_perms}...")

    model.eval()
    with torch.no_grad():
        perm_embeddings = []
        for graph in graphs:
            node_features = graph['node_features']
            edge_index = graph['edge_index']

            x = node_features
            for i, layer in enumerate(model.layers):
                x = layer(x, edge_index)
                if i < len(model.layers) - 1:
                    x = F.relu(x)
            perm_embeddings.append(x)

        avg_embeddings = torch.stack(perm_embeddings).mean(dim=0)

        src_embed = avg_embeddings[query_pairs[:, 0]]
        tgt_embed = avg_embeddings[query_pairs[:, 1]]
        pair_embed = torch.cat([src_embed, tgt_embed], dim=1)

        mu_pred = model.predictor(pair_embed).squeeze().numpy()

    counts_test = []
    for perm in test_perms:
        edge1, edge2 = load_permuted_edge_matrices(edge1_type, edge2_type, perm, data_dir)
        counts = compute_pathway_counts(pairs, edge1, edge2)
        counts_test.append(counts)
    counts_test = np.column_stack(counts_test)

    results = []
    for idx, perm in enumerate(test_perms):
        counts = counts_test[:, idx]
        r_mean = np.corrcoef(counts, mu_pred)[0, 1]

        results.append({
            'test_perm': perm,
            'model_type': 'gnn_multi_perm',
            'r_mean': r_mean
        })

    results_df = pd.DataFrame(results)

    mean_r = results_df['r_mean'].mean()
    print(f"\nTest Results (GNN multi-perm averaged):")
    print(f"  Mean r across test perms: {mean_r:.4f}")
    print(f"  Comparison to baseline linear: 0.787")
    print(f"  Comparison to best (RF/Hetero NN): 0.778")

    return results_df


def main():
    parser = argparse.ArgumentParser(description='Test GNN for pathway count prediction')
    parser.add_argument('metapath', help='Metapath name (e.g., CbGpPW)')
    parser.add_argument('--multi_perm', action='store_true',
                       help='Use multi-permutation embedding averaging')
    parser.add_argument('--n_samples', type=int, default=10000,
                       help='Number of node pairs to sample')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--train-perms', type=int, nargs='+', default=None,
                       help='Explicit train permutation IDs')
    parser.add_argument('--val-perms', type=int, nargs='+', default=None,
                       help='Explicit validation permutation IDs')
    parser.add_argument('--test-perms', type=int, nargs='+', default=None,
                       help='Explicit test permutation IDs')
    parser.add_argument('--smoke', action='store_true',
                       help='Run with lighter settings for quick validation')

    args = parser.parse_args()

    metapath_configs = {
        'CbGpPW': ('CbG', 'GpPW')
    }

    if args.metapath not in metapath_configs:
        print(f"Error: Unknown metapath {args.metapath}")
        print(f"Available: {list(metapath_configs.keys())}")
        sys.exit(1)

    edge1_type, edge2_type = metapath_configs[args.metapath]

    data_dir = repo_dir / 'data'
    output_dir = repo_dir / 'results' / 'gnn_pathway_counts'
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        args.n_samples = min(args.n_samples, 2000)

    available = list_available_permutation_ids(data_dir)
    train_perms, val_perms, test_perms = resolve_permutation_splits(
        available,
        train_perms=args.train_perms,
        val_perms=args.val_perms,
        test_perms=args.test_perms,
    )

    print("="*70)
    print(f"GNN Pathway Count Prediction: {args.metapath}")
    print("="*70)
    print(f"Train perms: {train_perms}")
    print(f"Validation perms: {val_perms}")
    print(f"Test perms: {test_perms}")

    print(f"\nSampling {args.n_samples} node pairs...")
    sample_perm = train_perms[0]
    edge1_perm0, edge2_perm0 = load_permuted_edge_matrices(edge1_type, edge2_type, sample_perm, data_dir)
    pairs = sample_pairs(edge1_perm0, edge2_perm0, n_samples=args.n_samples,
                        random_state=args.random_state)
    print(f"  Sampled pairs: {len(pairs)}")

    if args.multi_perm:
        model, graph_data = train_gnn_multi_perm(
            pairs, edge1_type, edge2_type, data_dir,
            train_perms=train_perms, val_perms=val_perms,
            n_samples=args.n_samples, random_state=args.random_state
        )

        results_df = evaluate_gnn_multi_perm(
            model, graph_data, pairs, edge1_type, edge2_type, data_dir, test_perms=test_perms
        )

        output_file = output_dir / f'{args.metapath}_gnn_multi_perm.csv'
    else:
        model, graph_data = train_gnn(
            pairs, edge1_type, edge2_type, data_dir,
            train_perms=train_perms, val_perms=val_perms,
            n_samples=args.n_samples, random_state=args.random_state
        )

        results_df = evaluate_gnn(
            model, graph_data, pairs, edge1_type, edge2_type, data_dir, test_perms=test_perms
        )

        output_file = output_dir / f'{args.metapath}_gnn.csv'

    results_df.to_csv(output_file, index=False)
    print(f"\nSaved results: {output_file}")


if __name__ == '__main__':
    main()
