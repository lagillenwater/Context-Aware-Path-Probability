"""
PathwayTransformer: Attention-based model for pathway frequency prediction.

This module implements a Transformer encoder that:
1. Processes variable-length node sequences (source → intermediates → target)
2. Uses self-attention to learn which nodes in path are important
3. Handles paths of any length (2, 3, 4, 5+ hops) with single model
4. Provides interpretable attention weights

Architecture:
    - Node feature embedding
    - Positional encoding
    - Multi-head self-attention layers
    - Attention pooling or [CLS] token aggregation
    - Regression head for pathway count prediction

Key advantages over DegreeSignatureNN:
- Variable-length paths (no feature explosion)
- Interpretable (attention shows important nodes)
- Captures sequential dependencies
- Better generalization to longer paths
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Tuple, Optional
from pathlib import Path


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for Transformer.

    Adds position information to node embeddings using sine/cosine functions
    of different frequencies.
    """

    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        """
        Initialize positional encoding.

        Parameters
        ----------
        d_model : int
            Embedding dimension
        max_len : int
            Maximum sequence length
        dropout : float
            Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (batch_size, seq_len, d_model)

        Returns
        -------
        output : torch.Tensor
            Input with positional encoding added
        """
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)


class PathwayTransformerEncoder(nn.Module):
    """
    Transformer encoder for pathway sequences.

    Processes node sequences with self-attention to capture relationships
    between nodes in the pathway.
    """

    def __init__(self,
                 node_feature_dim: int = 10,
                 d_model: int = 64,
                 nhead: int = 4,
                 num_layers: int = 3,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1,
                 max_seq_len: int = 100):
        """
        Initialize Transformer encoder.

        Parameters
        ----------
        node_feature_dim : int
            Dimension of input node features
        d_model : int
            Embedding dimension
        nhead : int
            Number of attention heads
        num_layers : int
            Number of Transformer layers
        dim_feedforward : int
            Dimension of feedforward network
        dropout : float
            Dropout probability
        max_seq_len : int
            Maximum sequence length
        """
        super().__init__()

        self.d_model = d_model

        self.node_embedding = nn.Linear(node_feature_dim, d_model)

        self.pos_encoder = PositionalEncoding(
            d_model, max_len=max_seq_len, dropout=dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.attention_pooling = nn.Linear(d_model, 1)

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self,
                node_features: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Transformer.

        Parameters
        ----------
        node_features : torch.Tensor
            Node features (batch_size, seq_len, node_feature_dim)
        attention_mask : torch.Tensor, optional
            Mask for padding (batch_size, seq_len), 1=real, 0=padding

        Returns
        -------
        pooled : torch.Tensor
            Pooled sequence representation (batch_size, d_model)
        attn_weights : torch.Tensor
            Attention weights (batch_size, seq_len)
        """
        batch_size, seq_len, _ = node_features.shape

        x = self.node_embedding(node_features)

        x = self.pos_encoder(x)

        if attention_mask is not None:
            padding_mask = (attention_mask == 0)
        else:
            padding_mask = None

        x = self.transformer(x, src_key_padding_mask=padding_mask)

        x = self.layer_norm(x)

        attn_logits = self.attention_pooling(x).squeeze(-1)

        if attention_mask is not None:
            attn_logits = attn_logits.masked_fill(attention_mask == 0, -1e9)

        attn_weights = F.softmax(attn_logits, dim=1)

        pooled = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)

        return pooled, attn_weights


class PathwayTransformer(nn.Module):
    """
    Complete Pathway Transformer model.

    Combines Transformer encoder with regression head for pathway
    count prediction.
    """

    def __init__(self,
                 node_feature_dim: int = 10,
                 d_model: int = 64,
                 nhead: int = 4,
                 num_layers: int = 3,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1,
                 max_seq_len: int = 100):
        """
        Initialize PathwayTransformer.

        Parameters
        ----------
        node_feature_dim : int
            Dimension of input node features
        d_model : int
            Embedding dimension
        nhead : int
            Number of attention heads
        num_layers : int
            Number of Transformer layers
        dim_feedforward : int
            Dimension of feedforward network
        dropout : float
            Dropout probability
        max_seq_len : int
            Maximum sequence length
        """
        super().__init__()

        self.encoder = PathwayTransformerEncoder(
            node_feature_dim=node_feature_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=max_seq_len
        )

        self.predictor = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, dim_feedforward // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 4, 1),
            nn.Softplus()
        )

    def forward(self,
                node_features: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        node_features : torch.Tensor
            Node features (batch_size, seq_len, node_feature_dim)
        attention_mask : torch.Tensor, optional
            Mask for padding (batch_size, seq_len)

        Returns
        -------
        outputs : dict
            Dictionary with:
            - 'prediction': Predicted pathway counts (batch_size, 1)
            - 'attention_weights': Attention weights (batch_size, seq_len)
            - 'encoding': Pooled sequence encoding (batch_size, d_model)
        """
        pooled, attn_weights = self.encoder(node_features, attention_mask)

        prediction = self.predictor(pooled)

        return {
            'prediction': prediction,
            'attention_weights': attn_weights,
            'encoding': pooled
        }

    def get_attention_weights(self,
                             node_features: torch.Tensor,
                             attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get attention weights for interpreting model.

        Parameters
        ----------
        node_features : torch.Tensor
            Node features (batch_size, seq_len, node_feature_dim)
        attention_mask : torch.Tensor, optional
            Mask for padding

        Returns
        -------
        attn_weights : torch.Tensor
            Attention weights (batch_size, seq_len)
        """
        with torch.no_grad():
            _, attn_weights = self.encoder(node_features, attention_mask)
        return attn_weights


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Testing PathwayTransformer...")

    batch_size = 8
    seq_len = 5
    node_feature_dim = 10

    node_features = torch.randn(batch_size, seq_len, node_feature_dim)

    attention_mask = torch.ones(batch_size, seq_len)
    attention_mask[0, 3:] = 0
    attention_mask[1, 4:] = 0

    model = PathwayTransformer(
        node_feature_dim=node_feature_dim,
        d_model=64,
        nhead=4,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.1,
        max_seq_len=20
    )

    print(f"\nModel parameters: {count_parameters(model):,}")

    outputs = model(node_features, attention_mask)

    print(f"\nOutput shapes:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")

    print(f"\nAttention weights (batch 0): {outputs['attention_weights'][0].detach().numpy()}")

    print(f"\nPredictions (batch 0-3): {outputs['prediction'][:4].detach().numpy().ravel()}")

    print("\nPathwayTransformer working correctly!")
