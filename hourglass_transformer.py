"""
Hourglass Transformer Architecture 
Based on NVIDIA's MeshTron paper: https://arxiv.org/html/2412.09548v1
adjusted for 2D Quad Meshes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math


class ShorteningLayer(nn.Module):
    """
        Shortening layer that reduces sequence length by a factor.
        The Shortening & Upsampling is according to the mesh structure
        coordinates => vertices => faces
        Example 3D Trinagulated Mesh 3 coordinates each vertex, 3 vertices each face => 9 => 3 => 1.
            => shortening_factor is 3x and 3x
        Example 2D Quadrangulated Mesh 2 coordinates each vertex, 4 vertices each face => 8 => 4 => 1.
            => shortening_factor is 2x and 4x
    """

    def __init__(self, d_model: int, shortening_factor: int, method: str = 'linear'):
        """
        Args:
            d_model: Model dimension
            shortening_factor: Factor by which to reduce sequence length
            method: 'average', 'linear', or 'attention'
        """
        super().__init__()
        self.d_model = d_model
        self.shortening_factor = shortening_factor
        self.method = method

        if method == 'linear':
            self.linear = nn.Linear(d_model * shortening_factor, d_model)
        elif method == 'attention':
            self.attention = nn.MultiheadAttention(
                d_model, num_heads=8, batch_first=True)
            self.query_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask

        Returns:
            shortened: Shortened sequence [batch_size, seq_len//factor, d_model]
            indices: Indices of tokens that were processed
        """
        batch_size, seq_len, d_model = x.shape
        factor = self.shortening_factor

        # Ensure sequence length is divisible by shortening factor
        if seq_len % factor != 0:
            padding_len = factor - (seq_len % factor)
            padding = torch.zeros(batch_size, padding_len,
                                  d_model, device=x.device)
            x = torch.cat([x, padding], dim=1)
            seq_len = x.shape[1]

        # Reshape for grouping
        x_grouped = x.view(batch_size, seq_len // factor, factor, d_model)

        if self.method == 'linear':
            # Linear projection
            x_flat = x_grouped.reshape(
                batch_size, seq_len // factor, factor * d_model)
            shortened = self.linear(x_flat)
        elif self.method == 'attention':
            # Attention-based pooling
            # Use the last token of each group as query
            # [batch, seq_len//factor, d_model]
            queries = self.query_proj(x_grouped[:, :, -1, :])
            keys = x_grouped.reshape(
                batch_size * (seq_len // factor), factor, d_model)
            values = keys

            # Reshape queries for attention
            queries = queries.reshape(
                batch_size * (seq_len // factor), 1, d_model)

            attended, _ = self.attention(queries, keys, values)
            shortened = attended.squeeze(1).reshape(
                batch_size, seq_len // factor, d_model)

        # Track which indices were kept (for static routing)
        indices = torch.arange(factor - 1, seq_len, factor, device=x.device)

        return shortened, indices


class UpsamplingLayer(nn.Module):
    """Upsampling layer that increases sequence length by a factor."""

    def __init__(self, d_model: int, upsampling_factor: int, method: str = 'linear'):
        """
        Args:
            d_model: Model dimension
            upsampling_factor: Factor by which to increase sequence length
            method: 'linear', or 'attention'
        """
        super().__init__()
        self.d_model = d_model
        self.upsampling_factor = upsampling_factor
        self.method = method

        if method == 'linear':
            self.linear = nn.Linear(d_model, d_model * upsampling_factor)
        elif method == 'attention':
            self.attention = nn.MultiheadAttention(
                d_model, num_heads=8, batch_first=True)
            self.pos_emb = nn.Parameter(
                torch.randn(1, upsampling_factor, d_model))

    def forward(self, x: torch.Tensor, target_len: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            target_len: Target sequence length after upsampling

        Returns:
            upsampled: Upsampled sequence [batch_size, seq_len*factor, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        factor = self.upsampling_factor

        if self.method == 'linear':
            # Linear projection
            projected = self.linear(x)  # [batch, seq_len, d_model * factor]
            upsampled = projected.reshape(
                batch_size, seq_len * factor, d_model)
        elif self.method == 'attention':
            # Attention-based upsampling
            upsampled_list = []
            for i in range(seq_len):
                # Use positional embeddings as queries
                queries = self.pos_emb.expand(batch_size, -1, -1)
                key = x[:, i:i+1, :].expand(-1, factor, -1)
                value = key

                attended, _ = self.attention(queries, key, value)
                upsampled_list.append(attended)

            upsampled = torch.cat(upsampled_list, dim=1)

        # Apply causal shift to preserve causality
        # Shift by (factor - 1) positions as per the paper
        shift_amount = factor - 1
        if shift_amount > 0:
            padding = torch.zeros(batch_size, shift_amount,
                                  d_model, device=x.device)
            upsampled = torch.cat([padding, upsampled], dim=1)
            if target_len is not None:
                upsampled = upsampled[:, :target_len, :]

        return upsampled


class HourglassTransformerBlock(nn.Module):
    """A single transformer block in the hourglass architecture."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True, is_causal=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual
        attn_out, _ = self.attention(
            x, x, x, attn_mask=mask, need_weights=False)
        x = self.norm1(x + attn_out)

        # Feed-forward with residual
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)

        return x


class CrossAttentionCondition(nn.Module):
    """
        A single transformer block which applis cross attention for a point clound and face count condition.
        Instead of a single input x used as query, value, key, we utilize latent space of the point cloud and face cound as key and vlaue.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual
        attn_out, _ = self.attention(
            query, key, value, attn_mask=mask, need_weights=False)
        x = self.norm1(query + attn_out)

        # Feed-forward with residual
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)

        return x


class HourglassStage(nn.Module):
    """A stage in the hourglass transformer with multiple blocks."""

    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            HourglassTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x


class HourglassTransformer(nn.Module):
    """
    Hourglass Transformer for 2D Quad MeshTron.

    Architecture:
    - Stage 1: Coordinate level (2x shortening for x,y pairs)
    - Stage 2: Vertex level (4x shortening for quad vertices)
    - Stage 3: Face level (operates on faces)
    - Stage 4: Reconsturcted Vertex Level (4x upsampling for x,y pairs)
    - Stage 5: Reconsturcted Coordinate Level (2x upsampling for x,y pairs)
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        stage_layers: Tuple[int, int, int, int, int] = (4, 8, 12, 16, 20),
        d_ff: int = 2048,
        dropout: float = 0.1,
        shortening_method: str = 'linear',
        upsampling_method: str = 'linear',
        use_static_routing: bool = True

    ):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            stage_layers: Number of layers in each stage (coord, vertex, face)
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            shortening_method: Method for shortening ('average', 'linear', 'attention')
            upsampling_method: Method for upsampling ('repeat', 'linear', 'attention')
            use_static_routing: Whether to use static routing mechanism
        """
        super().__init__()
        self.d_model = d_model

        self.use_static_routing = use_static_routing

        # Stage 1: Coordinate level
        self.stage1 = HourglassStage(
            stage_layers[0], d_model, n_heads, d_ff, dropout)
        self.stage1_conditioned = CrossAttentionCondition(
            d_model, n_heads, d_ff, dropout)

        # Shortening from coordinate to vertex level (2x for 2D coords)
        self.shortening1 = ShorteningLayer(d_model, 2, shortening_method)

        # Stage 2: Vertex level
        self.stage2 = HourglassStage(
            stage_layers[1], d_model, n_heads, d_ff, dropout)
        self.stage2_conditioned = CrossAttentionCondition(
            d_model, n_heads, d_ff, dropout)

        # Shortening from vertex to face level (4x for quad vertices)
        self.shortening2 = ShorteningLayer(d_model, 4, shortening_method)

        # Stage 3: Face level
        self.stage3 = HourglassStage(
            stage_layers[2], d_model, n_heads, d_ff, dropout)
        self.stage3_conditioned = CrossAttentionCondition(
            d_model, n_heads, d_ff, dropout)

        # Upsampling layer from face to vertex level (4x for quad vertices)
        self.upsampling2 = UpsamplingLayer(d_model, 4, upsampling_method)

        # Stage 4: Reconsturcted Vertex level
        self.stage4 = HourglassStage(
            stage_layers[3], d_model, n_heads, d_ff, dropout)
        self.stage4_conditioned = CrossAttentionCondition(
            d_model, n_heads, d_ff, dropout)

        # Upsampling layer from vertex coordinate level (2x 2D vertices)
        self.upsampling1 = UpsamplingLayer(d_model, 2, upsampling_method)

        self.stage5 = HourglassStage(
            stage_layers[4], d_model, n_heads, d_ff, dropout)
        self.stage5_conditioned = CrossAttentionCondition(
            d_model, n_heads, d_ff, dropout)

        # Residual projections
        self.residual_proj1 = nn.Linear(d_model, d_model)
        self.residual_proj2 = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        latent_condition: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the hourglass transformer.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask

        Returns:
            output: Output tensor [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        # Stage 1: Coordinate level
        stage1_out = self.stage1(x, mask)
        stage1_conditioned_out = self.stage1_conditioned(
            stage1_out, latent_condition, latent_condition)
        # Store for residual connection
        residual1 = stage1_conditioned_out

        # Shortening 1: Coordinate -> Vertex (2x reduction)
        shortened1, indices1 = self.shortening1(stage1_conditioned_out, mask)

        # Stage 2: Coordinate level
        stage2_out = self.stage2(shortened1)
        stage2_conditioned_out = self.stage2_conditioned(
            stage2_out, latent_condition, latent_condition)

        # Store for residual connection
        residual2 = stage2_conditioned_out

        # Shortening 2: Vertex -> Face (4x reduction)
        shortened2, indices2 = self.shortening2(stage2_conditioned_out)

        # Stage 3: Face level
        stage3_out = self.stage3(shortened2)
        stage3_conditioned_out = self.stage3_conditioned(
            stage3_out, latent_condition, latent_condition)

        # Upsampling 2: Face -> Vertex
        upsampled2 = self.upsampling2(
            stage3_conditioned_out, target_len=residual2.shape[1])

        # Add residual connection
        combined2 = upsampled2 + self.residual_proj2(residual2)

        # Stage 4: Reconsturcted Vertex level
        stage4_out = self.stage4(combined2)
        stage4_conditioned_out = self.stage4_conditioned(
            stage4_out, latent_condition, latent_condition)

        # Upsampling 1: Vertex -> Coordinate
        upsampled1 = self.upsampling1(
            stage4_conditioned_out, target_len=seq_len)

        # Add residual connection
        combined1 = upsampled1 + self.residual_proj1(residual1)

        # Stage 5: Reconsturcted Coordinate level
        stage5_out = self.stage5(combined1)
        output_conditioned = self.stage5_conditioned(
            stage5_out, latent_condition, latent_condition)

        return output_conditioned
