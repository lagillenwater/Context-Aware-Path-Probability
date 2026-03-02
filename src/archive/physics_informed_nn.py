"""
Physics-Informed Neural Network for Edge Probability Prediction.

This module implements a neural network trained using ONLY theoretical
constraints from XSwap Markov chain theory. No empirical frequencies are
used in training.

The network learns to predict null distribution edge probabilities by
respecting:
1. Conservation: Total predicted edges = actual edge count
2. Marginal consistency: Marginal distributions match observed
3. Detailed balance: XSwap equilibrium constraints
4. Monotonicity: Higher degree product implies higher probability
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, Optional
import scipy.sparse as sp
from pathlib import Path


class PhysicsInformedNN(nn.Module):
    """
    Neural network for edge probability prediction with physics constraints.

    Architecture: Simple feedforward network with skip connections
    Input: (source_degree, target_degree)
    Output: edge probability P(u,v)
    """

    def __init__(self, hidden_dims=[64, 32, 16], dropout=0.1):
        """
        Initialize physics-informed neural network.

        Parameters
        ----------
        hidden_dims : list
            Hidden layer dimensions
        dropout : float
            Dropout probability for regularization
        """
        super().__init__()

        layers = []
        in_dim = 2

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        u : tensor
            Source degrees (batch_size,)
        v : tensor
            Target degrees (batch_size,)

        Returns
        -------
        probs : tensor
            Predicted edge probabilities (batch_size,)
        """
        x = torch.stack([u, v], dim=1)
        x = x.float()
        return self.network(x).squeeze()


class PhysicsInformedLoss:
    """
    Loss function combining multiple physics constraints.

    Uses only observable graph features, no empirical frequencies.
    """

    def __init__(self,
                 edge_matrix: sp.spmatrix,
                 w_conservation: float = 1.0,
                 w_marginal: float = 1.0,
                 w_detailed_balance: float = 0.5,
                 w_monotonicity: float = 0.5):
        """
        Initialize physics-informed loss.

        Parameters
        ----------
        edge_matrix : sparse matrix
            Original graph adjacency matrix
        w_conservation : float
            Weight for conservation loss
        w_marginal : float
            Weight for marginal loss
        w_detailed_balance : float
            Weight for detailed balance loss
        w_monotonicity : float
            Weight for monotonicity loss
        """
        self.w_conservation = w_conservation
        self.w_marginal = w_marginal
        self.w_detailed_balance = w_detailed_balance
        self.w_monotonicity = w_monotonicity

        source_degrees = np.array(edge_matrix.sum(axis=1)).flatten()
        target_degrees = np.array(edge_matrix.sum(axis=0)).flatten()

        self.n_sources = edge_matrix.shape[0]
        self.n_targets = edge_matrix.shape[1]
        self.m_edges = edge_matrix.nnz

        self.source_degrees = torch.tensor(source_degrees, dtype=torch.float32)
        self.target_degrees = torch.tensor(target_degrees, dtype=torch.float32)

        self.mean_source_degree = self.source_degrees.mean()
        self.mean_target_degree = self.target_degrees.mean()

    def conservation_loss(self,
                         u: torch.Tensor,
                         v: torch.Tensor,
                         predicted_probs: torch.Tensor,
                         node_pair_counts: torch.Tensor) -> torch.Tensor:
        """
        Conservation constraint: Total predicted edges = actual edge count.

        For each degree pair (u,v), we have n_pairs with those degrees.
        Expected edges = sum over all pairs of (n_pairs * predicted_prob).

        Parameters
        ----------
        u, v : tensor
            Unique degree pairs
        predicted_probs : tensor
            Predicted probabilities for each degree pair
        node_pair_counts : tensor
            Number of node pairs with each degree combination

        Returns
        -------
        loss : tensor
            Conservation loss
        """
        predicted_edges = torch.sum(predicted_probs * node_pair_counts)

        actual_edges = float(self.m_edges)

        loss = ((predicted_edges - actual_edges) / actual_edges) ** 2

        return loss

    def marginal_loss(self,
                     u: torch.Tensor,
                     v: torch.Tensor,
                     predicted_probs: torch.Tensor,
                     node_pair_counts: torch.Tensor) -> torch.Tensor:
        """
        Marginal consistency: Marginal distributions match observed degrees.

        Simplified version: Instead of checking every unique degree value,
        we sample a subset to make computation tractable.

        Parameters
        ----------
        u, v : tensor
            Unique degree pairs
        predicted_probs : tensor
            Predicted probabilities for each degree pair
        node_pair_counts : tensor
            Number of node pairs with each degree combination

        Returns
        -------
        loss : tensor
            Marginal consistency loss (simplified)
        """
        unique_u = torch.unique(u)
        unique_v = torch.unique(v)

        n_samples_u = min(20, len(unique_u))
        n_samples_v = min(20, len(unique_v))

        sampled_u_indices = torch.randperm(len(unique_u))[:n_samples_u]
        sampled_v_indices = torch.randperm(len(unique_v))[:n_samples_v]

        sampled_u = unique_u[sampled_u_indices]
        sampled_v = unique_v[sampled_v_indices]

        source_loss = 0.0
        for u_val in sampled_u:
            mask = (u == u_val)
            if not mask.any():
                continue

            v_for_u = v[mask]
            probs_for_u = predicted_probs[mask]

            n_targets_per_v = torch.tensor([
                (self.target_degrees == v_val.item()).sum().float()
                for v_val in v_for_u
            ], dtype=torch.float32, device=probs_for_u.device)

            predicted_degree = torch.sum(probs_for_u * n_targets_per_v)
            expected_degree = u_val

            source_loss += ((predicted_degree - expected_degree) / (expected_degree + 1e-8)) ** 2

        source_loss = source_loss / n_samples_u

        target_loss = 0.0
        for v_val in sampled_v:
            mask = (v == v_val)
            if not mask.any():
                continue

            u_for_v = u[mask]
            probs_for_v = predicted_probs[mask]

            n_sources_per_u = torch.tensor([
                (self.source_degrees == u_val.item()).sum().float()
                for u_val in u_for_v
            ], dtype=torch.float32, device=probs_for_v.device)

            predicted_degree = torch.sum(probs_for_v * n_sources_per_u)
            expected_degree = v_val

            target_loss += ((predicted_degree - expected_degree) / (expected_degree + 1e-8)) ** 2

        target_loss = target_loss / n_samples_v

        return source_loss + target_loss

    def detailed_balance_loss(self,
                             u: torch.Tensor,
                             v: torch.Tensor,
                             predicted_probs: torch.Tensor) -> torch.Tensor:
        """
        Detailed balance constraint from XSwap Markov chain theory.

        The XSwap process satisfies detailed balance at equilibrium.
        For edge swap (u1,v1), (u2,v2) -> (u1,v2), (u2,v1):

        P(u1,v1) * P(u2,v2) * swap_prob = P(u1,v2) * P(u2,v1) * swap_prob_reverse

        Under symmetric proposal distribution:
        P(u1,v1) * P(u2,v2) ≈ P(u1,v2) * P(u2,v1)

        We enforce this by sampling pairs and minimizing the violation.

        Parameters
        ----------
        u, v : tensor
            Degree pairs
        predicted_probs : tensor
            Predicted probabilities

        Returns
        -------
        loss : tensor
            Detailed balance loss
        """
        n_samples = min(100, len(u) // 2)

        if len(u) < 4:
            return torch.tensor(0.0)

        indices = torch.randperm(len(u))[:n_samples*2]

        u1 = u[indices[:n_samples]]
        v1 = v[indices[:n_samples]]
        u2 = u[indices[n_samples:2*n_samples]]
        v2 = v[indices[n_samples:2*n_samples]]

        p_u1v1_idx = indices[:n_samples]
        p_u2v2_idx = indices[n_samples:2*n_samples]

        p_u1v1 = predicted_probs[p_u1v1_idx]
        p_u2v2 = predicted_probs[p_u2v2_idx]

        u1v2_found = []
        u2v1_found = []

        for i in range(n_samples):
            mask_u1v2 = (u == u1[i]) & (v == v2[i])
            if mask_u1v2.any():
                u1v2_found.append(predicted_probs[mask_u1v2].mean())
            else:
                u1v2_found.append(p_u1v1[i])

            mask_u2v1 = (u == u2[i]) & (v == v1[i])
            if mask_u2v1.any():
                u2v1_found.append(predicted_probs[mask_u2v1].mean())
            else:
                u2v1_found.append(p_u2v2[i])

        p_u1v2 = torch.stack(u1v2_found)
        p_u2v1 = torch.stack(u2v1_found)

        lhs = torch.log(p_u1v1 + 1e-8) + torch.log(p_u2v2 + 1e-8)
        rhs = torch.log(p_u1v2 + 1e-8) + torch.log(p_u2v1 + 1e-8)

        loss = torch.mean((lhs - rhs) ** 2)

        return loss

    def monotonicity_loss(self,
                         u: torch.Tensor,
                         v: torch.Tensor,
                         predicted_probs: torch.Tensor) -> torch.Tensor:
        """
        Monotonicity constraint: Higher degree product implies higher probability.

        This is a soft constraint that encourages the model to respect the
        general trend that high-degree nodes are more likely to connect.

        Parameters
        ----------
        u, v : tensor
            Degree pairs
        predicted_probs : tensor
            Predicted probabilities

        Returns
        -------
        loss : tensor
            Monotonicity violation loss
        """
        degree_products = u * v

        sorted_indices = torch.argsort(degree_products)
        sorted_probs = predicted_probs[sorted_indices]

        violations = torch.relu(sorted_probs[:-1] - sorted_probs[1:])

        loss = torch.mean(violations ** 2)

        return loss

    def __call__(self,
                 u: torch.Tensor,
                 v: torch.Tensor,
                 predicted_probs: torch.Tensor,
                 node_pair_counts: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute total physics-informed loss.

        Parameters
        ----------
        u, v : tensor
            Degree pairs
        predicted_probs : tensor
            Predicted probabilities
        node_pair_counts : tensor
            Number of node pairs with each degree combination

        Returns
        -------
        total_loss : tensor
            Weighted sum of all losses
        loss_dict : dict
            Individual loss components
        """
        conservation = self.conservation_loss(u, v, predicted_probs, node_pair_counts)
        marginal = self.marginal_loss(u, v, predicted_probs, node_pair_counts)
        detailed_balance = self.detailed_balance_loss(u, v, predicted_probs)
        monotonicity = self.monotonicity_loss(u, v, predicted_probs)

        total_loss = (self.w_conservation * conservation +
                     self.w_marginal * marginal +
                     self.w_detailed_balance * detailed_balance +
                     self.w_monotonicity * monotonicity)

        loss_dict = {
            'total': total_loss.item(),
            'conservation': conservation.item(),
            'marginal': marginal.item(),
            'detailed_balance': detailed_balance.item(),
            'monotonicity': monotonicity.item()
        }

        return total_loss, loss_dict


def compute_node_pair_counts(edge_matrix: sp.spmatrix,
                             u_degrees: np.ndarray,
                             v_degrees: np.ndarray) -> np.ndarray:
    """
    Compute number of node pairs for each (u,v) degree combination.

    Parameters
    ----------
    edge_matrix : sparse matrix
        Graph adjacency matrix
    u_degrees : array
        Source degrees for each degree pair
    v_degrees : array
        Target degrees for each degree pair

    Returns
    -------
    counts : array
        Number of node pairs with each degree combination
    """
    source_degrees = np.array(edge_matrix.sum(axis=1)).flatten()
    target_degrees = np.array(edge_matrix.sum(axis=0)).flatten()

    counts = []
    for u_val, v_val in zip(u_degrees, v_degrees):
        n_sources_with_u = (source_degrees == u_val).sum()
        n_targets_with_v = (target_degrees == v_val).sum()
        counts.append(n_sources_with_u * n_targets_with_v)

    return np.array(counts)


def train_physics_informed_nn(edge_matrix: sp.spmatrix,
                              degree_pairs: np.ndarray,
                              n_epochs: int = 1000,
                              learning_rate: float = 0.001,
                              hidden_dims=[64, 32, 16],
                              loss_weights: Dict[str, float] = None,
                              verbose: bool = True) -> PhysicsInformedNN:
    """
    Train physics-informed neural network.

    Parameters
    ----------
    edge_matrix : sparse matrix
        Original graph adjacency matrix
    degree_pairs : array
        Array of (source_degree, target_degree) pairs to train on
    n_epochs : int
        Number of training epochs
    learning_rate : float
        Learning rate for Adam optimizer
    hidden_dims : list
        Hidden layer dimensions
    loss_weights : dict
        Weights for loss components (conservation, marginal, detailed_balance, monotonicity)
    verbose : bool
        Print training progress

    Returns
    -------
    model : PhysicsInformedNN
        Trained neural network
    """
    if loss_weights is None:
        loss_weights = {
            'conservation': 1.0,
            'marginal': 1.0,
            'detailed_balance': 0.5,
            'monotonicity': 0.5
        }

    model = PhysicsInformedNN(hidden_dims=hidden_dims)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_fn = PhysicsInformedLoss(
        edge_matrix,
        w_conservation=loss_weights['conservation'],
        w_marginal=loss_weights['marginal'],
        w_detailed_balance=loss_weights['detailed_balance'],
        w_monotonicity=loss_weights['monotonicity']
    )

    u_degrees = degree_pairs[:, 0]
    v_degrees = degree_pairs[:, 1]

    node_pair_counts = compute_node_pair_counts(edge_matrix, u_degrees, v_degrees)

    u_tensor = torch.tensor(u_degrees, dtype=torch.float32)
    v_tensor = torch.tensor(v_degrees, dtype=torch.float32)
    counts_tensor = torch.tensor(node_pair_counts, dtype=torch.float32)

    u_normalized = (u_tensor - u_tensor.mean()) / (u_tensor.std() + 1e-8)
    v_normalized = (v_tensor - v_tensor.mean()) / (v_tensor.std() + 1e-8)

    model.train()

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        predicted_probs = model(u_normalized, v_normalized)

        total_loss, loss_dict = loss_fn(u_tensor, v_tensor, predicted_probs, counts_tensor)

        total_loss.backward()
        optimizer.step()

        if verbose and (epoch % 100 == 0 or epoch == n_epochs - 1):
            print(f"Epoch {epoch:4d} | Total: {loss_dict['total']:.6f} | "
                  f"Conservation: {loss_dict['conservation']:.6f} | "
                  f"Marginal: {loss_dict['marginal']:.6f} | "
                  f"DetailedBalance: {loss_dict['detailed_balance']:.6f} | "
                  f"Monotonicity: {loss_dict['monotonicity']:.6f}")

    model.eval()
    return model
