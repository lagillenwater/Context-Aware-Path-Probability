"""
Unique edge sampling and progressive experiment utilities for edge prediction.
"""
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any
import scipy.sparse as sp
from .data_processing import load_permutation_data, extract_improved_edge_features_and_labels
from .models import OptimizedModelTrainer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class UniqueEdgeSampler:
    """
    Implements unique edge sampling across multiple permutations to eliminate data leakage.
    """
    def __init__(self, edge_type: str = 'CtD', random_seed: int = 42):
        self.edge_type = edge_type
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def sample_unique_edges_from_permutations(self, permutation_names: List[str], permutations_dir: Path, samples_per_permutation: int = 10000, negative_ratio: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        print(f"Sampling unique edges from {len(permutation_names)} permutations...")
        unique_positive_edges = set()
        unique_edge_features = []
        unique_edge_targets = []
        edge_to_perm = {}
        all_permutation_data = {}
        total_positive_attempted = 0
        for perm_idx, perm_name in enumerate(permutation_names):
            perm_dir = permutations_dir / perm_name
            edge_matrix, source_degrees, target_degrees = load_permutation_data(perm_dir, self.edge_type)
            all_permutation_data[perm_name] = {
                'edge_matrix': edge_matrix,
                'source_degrees': source_degrees,
                'target_degrees': target_degrees
            }
            rows, cols = edge_matrix.nonzero()
            positive_edges = list(zip(rows, cols))
            total_positive_attempted += len(positive_edges)
            if len(positive_edges) > samples_per_permutation:
                sampled_indices = np.random.choice(len(positive_edges), samples_per_permutation, replace=False)
                positive_edges = [positive_edges[i] for i in sampled_indices]
            for edge in positive_edges:
                if edge not in unique_positive_edges:
                    unique_positive_edges.add(edge)
                    edge_to_perm[edge] = perm_name
        for edge in unique_positive_edges:
            source, target = edge
            perm_name = edge_to_perm[edge]
            perm_data = all_permutation_data[perm_name]
            source_degree = perm_data['source_degrees'][source]
            target_degree = perm_data['target_degrees'][target]
            features = np.array([source_degree, target_degree])
            unique_edge_features.append(features)
            unique_edge_targets.append(1.0)
        negative_edges_needed = int(len(unique_positive_edges) * negative_ratio)
        negative_edges_generated = 0
        for perm_name in permutation_names:
            if negative_edges_generated >= negative_edges_needed:
                break
            perm_data = all_permutation_data[perm_name]
            edge_matrix = perm_data['edge_matrix']
            source_degrees = perm_data['source_degrees']
            target_degrees = perm_data['target_degrees']
            n_sources = edge_matrix.shape[0]
            n_targets = edge_matrix.shape[1]
            perm_negative_needed = min(negative_edges_needed - negative_edges_generated, samples_per_permutation)
            attempts = 0
            max_attempts = perm_negative_needed * 10
            while negative_edges_generated < negative_edges_needed and attempts < max_attempts:
                source = np.random.randint(0, n_sources)
                target = np.random.randint(0, n_targets)
                edge_exists = False
                for other_perm_data in all_permutation_data.values():
                    if other_perm_data['edge_matrix'][source, target] > 0:
                        edge_exists = True
                        break
                if not edge_exists:
                    features = np.array([source_degrees[source], target_degrees[target]])
                    unique_edge_features.append(features)
                    unique_edge_targets.append(0.0)
                    negative_edges_generated += 1
                attempts += 1
        combined_features = np.vstack(unique_edge_features)
        combined_targets = np.array(unique_edge_targets)
        return combined_features, combined_targets

    def progressive_experiment(self, training_perms: List[str], permutations_dir: Path, max_permutations: int = 10) -> Dict[str, Any]:
        results = {
            'permutation_counts': [],
            'training_metrics': [],
            'validation_metrics': [],
            'unique_edge_counts': [],
            'positive_edge_counts': []
        }
        validation_perm = training_perms[-1]
        validation_dir = permutations_dir / validation_perm
        val_edge_matrix, val_source_degrees, val_target_degrees = load_permutation_data(validation_dir, self.edge_type)
        val_features, val_targets = extract_improved_edge_features_and_labels(val_edge_matrix, val_source_degrees, val_target_degrees, 1.0, False, True)
        training_perms_subset = training_perms[:-1]
        for n_perms in range(1, min(len(training_perms_subset), max_permutations) + 1):
            current_perms = training_perms_subset[:n_perms]
            train_features, train_targets = self.sample_unique_edges_from_permutations(current_perms, permutations_dir, samples_per_permutation=8000)
            trainer = OptimizedModelTrainer('NN', self.random_seed, True, True)
            training_metrics = trainer.train(train_features, train_targets, test_size=0.2)
            val_predictions = trainer.predict_probabilities(val_features)
            val_mse = mean_squared_error(val_targets, val_predictions)
            val_mae = mean_absolute_error(val_targets, val_predictions)
            val_r2 = r2_score(val_targets, val_predictions)
            validation_metrics = {'mse': val_mse, 'mae': val_mae, 'r2': val_r2}
            results['permutation_counts'].append(n_perms)
            results['training_metrics'].append(training_metrics['metrics'])
            results['validation_metrics'].append(validation_metrics)
            results['unique_edge_counts'].append(len(train_features))
            results['positive_edge_counts'].append(np.sum(train_targets == 1.0))
        return results
