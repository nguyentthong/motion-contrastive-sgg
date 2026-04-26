"""Model components for temporal panoptic scene graph generation.

Implements:
    - TransformerTubeEncoder: self-attention over temporal dimension
    - ConvolutionTubeEncoder: 1D conv layers over temporal dimension
    - RelationClassifier: MLP over concatenated pooled representations (Eq. 2, 3)
    - TPSGGModel: Full relation classification model
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    input_dim: int = 256          # Dimension of input mask tube features
    hidden_dim: int = 256         # Hidden dimension D
    num_relation_classes: int = 57 # Number of relation categories
    num_entity_classes: int = 133  # Number of entity categories
    tube_encoder_type: str = "convolution"  # 'transformer' or 'convolution'
    num_encoder_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    max_tube_length: int = 64     # Max temporal length T


class TransformerTubeEncoder(nn.Module):
    """Encode mask tubes using self-attention over the temporal dimension.

    "IPS+T - Transformer uses ... Transformer-based encoder with self-attention
    layers to encode entity mask tubes"
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
        self.pos_embed = nn.Parameter(
            torch.randn(1, config.max_tube_length, config.hidden_dim) * 0.02
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_encoder_layers
        )
        self.norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode tube representation.

        Args:
            x: [B, T, D_in] mask tube features.

        Returns:
            H: [B, T, D] encoded tube representations.
        """
        B, T, _ = x.shape
        x = self.input_proj(x)
        x = x + self.pos_embed[:, :T, :]
        x = self.encoder(x)
        return self.norm(x)


class ConvolutionTubeEncoder(nn.Module):
    """Encode mask tubes using 1D convolutions over temporal dimension.

    "IPS+T - Convolution uses ... learnable convolutional layers to encode entity mask tubes"
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
        layers = []
        for i in range(config.num_encoder_layers):
            layers.extend([
                nn.Conv1d(config.hidden_dim, config.hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(config.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(config.dropout),
            ])
        self.conv_layers = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D_in] mask tube features.

        Returns:
            H: [B, T, D] encoded tube representations.
        """
        x = self.input_proj(x)                     # [B, T, D]
        x = rearrange(x, "b t d -> b d t")         # [B, D, T] for Conv1d
        x = self.conv_layers(x)
        x = rearrange(x, "b d t -> b t d")         # [B, T, D]
        return self.norm(x)


class RelationClassifier(nn.Module):
    """MLP relation classifier over pooled concatenated representations (Eq. 2, 3).

    h_i = Pooling(H_i)                        (Eq. 2)
    log p(r_{i,j}) = MLP([h_i, h_j])          (Eq. 3)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.num_relation_classes),
        )

    def forward(self, h_sub: torch.Tensor, h_obj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_sub: [B, D] pooled subject representation.
            h_obj: [B, D] pooled object representation.

        Returns:
            logits: [B, num_relation_classes].
        """
        concat = torch.cat([h_sub, h_obj], dim=-1)  # [B, 2D]
        return self.mlp(concat)


def build_tube_encoder(config: ModelConfig) -> nn.Module:
    """Factory for tube encoders."""
    if config.tube_encoder_type == "transformer":
        return TransformerTubeEncoder(config)
    elif config.tube_encoder_type == "convolution":
        return ConvolutionTubeEncoder(config)
    else:
        raise ValueError(f"Unknown encoder type: {config.tube_encoder_type}")


class TPSGGModel(nn.Module):
    """Full temporal panoptic scene graph generation relation model.

    Pipeline (Stage B):
    1. Encode entity mask tubes -> H_i [T, D]
    2. Global pool -> h_i [D]
    3. For all pairs (i, j), predict relation via MLP([h_i, h_j])
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.tube_encoder = build_tube_encoder(config)
        self.relation_classifier = RelationClassifier(config)

    def encode_tubes(self, tube_features: torch.Tensor) -> torch.Tensor:
        """Encode mask tube features.

        Args:
            tube_features: [B, T, D_in] raw tube features.

        Returns:
            H: [B, T, D] encoded representations.
        """
        return self.tube_encoder(tube_features)

    def pool_tubes(self, H: torch.Tensor) -> torch.Tensor:
        """Global average pooling over temporal dimension (Eq. 2).

        Args:
            H: [B, T, D] encoded tube representations.

        Returns:
            h: [B, D] pooled representations.
        """
        return H.mean(dim=1)

    def classify_relations(
        self, h_sub: torch.Tensor, h_obj: torch.Tensor
    ) -> torch.Tensor:
        """Predict relation logits (Eq. 3).

        Args:
            h_sub: [B, D] pooled subject representations.
            h_obj: [B, D] pooled object representations.

        Returns:
            logits: [B, num_relation_classes].
        """
        return self.relation_classifier(h_sub, h_obj)

    def forward(
        self,
        subject_tubes: torch.Tensor,
        object_tubes: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass: encode, pool, classify.

        Args:
            subject_tubes: [B, T, D_in] subject mask tube features.
            object_tubes: [B, T, D_in] object mask tube features.

        Returns:
            Dict with 'logits', 'h_sub', 'h_obj', 'H_sub', 'H_obj'.
        """
        H_sub = self.encode_tubes(subject_tubes)  # [B, T, D]
        H_obj = self.encode_tubes(object_tubes)    # [B, T, D]

        h_sub = self.pool_tubes(H_sub)  # [B, D]
        h_obj = self.pool_tubes(H_obj)  # [B, D]

        logits = self.classify_relations(h_sub, h_obj)  # [B, C]

        return {
            "logits": logits,
            "h_sub": h_sub,
            "h_obj": h_obj,
            "H_sub": H_sub,
            "H_obj": H_obj,
        }
