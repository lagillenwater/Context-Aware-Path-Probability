"""
Validation and distribution analysis utilities for edge prediction.
"""
import numpy as np
from typing import Tuple, Dict, Any

def compute_adaptive_degree_based_probability_distribution(edge_matrix, source_degrees, target_degrees, n_bins=8, adaptive_binning=True):
    if adaptive_binning:
        source_nonzero = source_degrees[source_degrees > 0]
        target_nonzero = target_degrees[target_degrees > 0]
        if len(source_nonzero) == 0:
            source_bin_edges = np.array([0, 1])
        else:
            source_quantiles = np.linspace(0, 100, n_bins + 1)
            source_bin_edges = np.percentile(source_degrees, source_quantiles)
            source_bin_edges = np.unique(source_bin_edges)
            if len(source_bin_edges) < 3:
                source_bin_edges = np.linspace(source_degrees.min(), source_degrees.max(), 3)
        if len(target_nonzero) == 0:
            target_bin_edges = np.array([0, 1])
        else:
            target_quantiles = np.linspace(0, 100, n_bins + 1)
            target_bin_edges = np.percentile(target_degrees, target_quantiles)
            target_bin_edges = np.unique(target_bin_edges)
            if len(target_bin_edges) < 3:
                target_bin_edges = np.linspace(target_degrees.min(), target_degrees.max(), 3)
    else:
        source_bin_edges = np.linspace(source_degrees.min(), source_degrees.max(), n_bins + 1)
        target_bin_edges = np.linspace(target_degrees.min(), target_degrees.max(), n_bins + 1)
    n_source_bins = len(source_bin_edges) - 1
    n_target_bins = len(target_bin_edges) - 1
    edge_counts = np.zeros((n_source_bins, n_target_bins))
    total_counts = np.zeros((n_source_bins, n_target_bins))
    n_nodes_source, n_nodes_target = edge_matrix.shape
    max_sample_pairs = 200000
    if n_nodes_source * n_nodes_target > max_sample_pairs:
        source_weights = (source_degrees + 1) / (source_degrees + 1).sum()
        target_weights = (target_degrees + 1) / (target_degrees + 1).sum()
        n_samples = int(np.sqrt(max_sample_pairs))
        source_indices = np.random.choice(n_nodes_source, n_samples, p=source_weights, replace=True)
        target_indices = np.random.choice(n_nodes_target, n_samples, p=target_weights, replace=True)
        for i, j in zip(source_indices, target_indices):
            source_bin = np.digitize(source_degrees[i], source_bin_edges) - 1
            target_bin = np.digitize(target_degrees[j], target_bin_edges) - 1
            source_bin = max(0, min(source_bin, n_source_bins - 1))
            target_bin = max(0, min(target_bin, n_target_bins - 1))
            total_counts[source_bin, target_bin] += 1
            if edge_matrix[i, j]:
                edge_counts[source_bin, target_bin] += 1
    else:
        for i in range(n_nodes_source):
            for j in range(n_nodes_target):
                source_bin = np.digitize(source_degrees[i], source_bin_edges) - 1
                target_bin = np.digitize(target_degrees[j], target_bin_edges) - 1
                source_bin = max(0, min(source_bin, n_source_bins - 1))
                target_bin = max(0, min(target_bin, n_target_bins - 1))
                total_counts[source_bin, target_bin] += 1
                if edge_matrix[i, j]:
                    edge_counts[source_bin, target_bin] += 1
    alpha = 1e-6
    prob_matrix = (edge_counts + alpha) / (total_counts + 2 * alpha)
    return prob_matrix, source_bin_edges, target_bin_edges

def compute_enhanced_distribution_difference(observed_dist, predicted_dist):
    obs_flat = observed_dist.flatten()
    pred_flat = predicted_dist.flatten()
    valid_mask = ~(np.isnan(obs_flat) | np.isnan(pred_flat))
    obs_clean = obs_flat[valid_mask]
    pred_clean = pred_flat[valid_mask]
    if len(obs_clean) == 0:
        return {'mse': np.inf, 'mae': np.inf, 'wasserstein': np.inf, 'ks_statistic': 1.0, 'jensen_shannon': 1.0, 'hellinger': 1.0, 'relative_entropy': np.inf}
    mse = np.mean((obs_clean - pred_clean) ** 2)
    mae = np.mean(np.abs(obs_clean - pred_clean))
    from scipy.stats import wasserstein_distance, ks_2samp
    try:
        wasserstein = wasserstein_distance(obs_clean, pred_clean)
        ks_stat = ks_2samp(obs_clean, pred_clean).statistic
    except:
        wasserstein = np.inf
        ks_stat = 1.0
    def jensen_shannon_distance(p, q):
        p = p + 1e-10
        q = q + 1e-10
        m = 0.5 * (p + q)
        return 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
    def hellinger_distance(p, q):
        return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))
    obs_norm = obs_clean / (obs_clean.sum() + 1e-10)
    pred_norm = pred_clean / (pred_clean.sum() + 1e-10)
    try:
        js_distance = jensen_shannon_distance(obs_norm, pred_norm)
        hellinger = hellinger_distance(obs_norm, pred_norm)
    except:
        js_distance = 1.0
        hellinger = 1.0
    try:
        kl_div = np.sum(obs_norm * np.log((obs_norm + 1e-10) / (pred_norm + 1e-10)))
    except:
        kl_div = np.inf
    metrics = {'mse': mse, 'mae': mae, 'wasserstein': wasserstein, 'ks_statistic': ks_stat, 'jensen_shannon': js_distance, 'hellinger': hellinger, 'relative_entropy': kl_div}
    return metrics
