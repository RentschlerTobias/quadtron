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
from attention import MultiHeadAttention


class ShorteningLayer(nn.Module):
    def __init__(self, d_model: int, shortening_factor: int):
        super().__init__()
        self.d_model = d_model
        self.shortening_factor = shortening_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Wählt das letzte Element jedes Blocks.
        Factor 2: [x0,y0,x1,y1,...] → [y0,y1,...] (Indizes 1,3,5,...)
        Factor 4: [v0,v1,v2,v3,v4,...] → [v3,v7,...] (Indizes 3,7,11,...)
        """
        _, seq_len, _ = x.shape
        indices = torch.arange(
            self.shortening_factor - 1,
            seq_len,
            self.shortening_factor,
            device=x.device
        )
        return x[:, indices, :]
#


class UpsamplingLayer(nn.Module):
    def __init__(self, d_model: int, upsampling_factor: int):
        super().__init__()
        self.d_model = d_model
        self.factor = upsampling_factor

        # Projektion um Dimensionen beizubehalten/anzupassen
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        Args:
            x: Input [batch, shortened_len, d_model]
            target_len: Die Länge des Residual-Tensors (z.B. 3407)
        """
        # 1. Lineare Projektion
        x = self.linear(x)

        # 2. Wiederholen der Tokens (Repeat Interleave)
        # Aus [A, B] wird [A, A, A, A, B, B, B, B] (bei factor 4)
        x = x.repeat_interleave(self.factor, dim=1)

        # 3. Auf target_len anpassen: auffüllen wenn seq_len nicht durch factor teilbar
        current_len = x.shape[1]
        if current_len < target_len:
            pad = torch.zeros(x.shape[0], target_len - current_len, x.shape[2], device=x.device)
            x = torch.cat([x, pad], dim=1)
        return x[:, :target_len, :]


class HourglassTransformerBlock(nn.Module):
    """A single transformer block in the hourglass architecture."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.2, max_position: Optional[int] = 1000):
        super().__init__()
        self.attention = MultiHeadAttention(
            d_model, n_heads, dropout=dropout, max_position=max_position)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, is_casual: Optional[bool] = True, position_ids: Optional[torch.tensor] = None) -> torch.Tensor:
        # Self-attention with residual

        if is_casual:
            _, seq_len, _ = x.shape
            mask = self.create_manual_mask(seq_len, x.device)
        else:
            mask = None

        attn_out = self.attention(
            x, x, x, mask=mask, position_ids=position_ids)
        x = self.norm1(x + attn_out)

        # Feed-forward with residual
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)

        return x

    def create_manual_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(
            (seq_len, seq_len), device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        # Erweitere für batch und heads: [1, 1, seq_len, seq_len]
        return mask.unsqueeze(0).unsqueeze(0)
    # def create_manual_mask(self, seq_len, device):
    #     mask = torch.ones((seq_len, seq_len), device=device)
    #     # triu(1) behält das obere Dreieck AB DER 1. DIAGONALEN (also ohne die Mitte)
    #     mask = torch.triu(mask, diagonal=1)
    #     # Ersetze 1en mit -inf und 0en mit 0
    #     mask = mask.masked_fill(mask == 1, float('-inf'))
    #     # Optional: Nullen explizit setzen (oft nicht nötig, da masked_fill reicht)
    #     mask = mask.masked_fill(mask == 0, float(0.0))
    #     return mask


class CrossAttentionCondition(nn.Module):
    """
        A single transformer block which applis cross attention for a point clound and face count condition.
        Instead of a single input x used as query, value, key, we utilize latent space of the point cloud and face cound as key and vlaue.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, max_position: Optional[int] = None):
        super().__init__()
        self.attention = MultiHeadAttention(
            d_model, n_heads, dropout=dropout, is_cross_attention=True, max_position=max_position)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual
        attn_out = self.attention(
            query, key, value, mask=mask, position_ids=position_ids)
        x = self.norm1(query + attn_out)

        # Feed-forward with residual
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)

        return x


class HourglassStage(nn.Module):
    """A stage in the hourglass transformer with multiple blocks."""

    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, max_position: Optional[int] = None):
        super().__init__()
        self.layers = nn.ModuleList([
            HourglassTransformerBlock(
                d_model, n_heads, d_ff, dropout, max_position)
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, is_casual, position_ids) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, is_casual, position_ids)
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
        max_position: Optional[int] = None

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
        """
        super().__init__()
        self.d_model = d_model

        # Stage 1: Coordinate level
        self.stage1 = HourglassStage(
            stage_layers[0], d_model, n_heads, d_ff, dropout, max_position)
        self.stage1_conditioned = CrossAttentionCondition(
            d_model, n_heads, d_ff, dropout, max_position)

        # Shortening from coordinate to vertex level (2x for 2D coords)
        self.factor1 = 2
        self.shortening1 = ShorteningLayer(d_model, self.factor1)

        # Stage 2: Vertex level
        self.stage2 = HourglassStage(
            stage_layers[1], d_model, n_heads, d_ff, dropout, max_position)
        self.stage2_conditioned = CrossAttentionCondition(
            d_model, n_heads, d_ff, dropout, max_position)

        # Shortening from vertex to face level (4x for quad vertices)
        self.factor2 = 4
        self.shortening2 = ShorteningLayer(d_model, self.factor2)

        # Stage 3: Face level
        self.stage3 = HourglassStage(
            stage_layers[2], d_model, n_heads, d_ff, dropout, max_position)
        self.stage3_conditioned = CrossAttentionCondition(
            d_model, n_heads, d_ff, dropout, max_position)

        # Upsampling layer from face to vertex level (4x for quad vertices)
        self.upsampling2 = UpsamplingLayer(d_model, self.factor2)

        # Stage 4: Reconsturcted Vertex level
        self.stage4 = HourglassStage(
            stage_layers[3], d_model, n_heads, d_ff, dropout, max_position)
        self.stage4_conditioned = CrossAttentionCondition(
            d_model, n_heads, d_ff, dropout, max_position)

        # Upsampling layer from vertex coordinate level (2x 2D vertices)
        self.upsampling1 = UpsamplingLayer(d_model, self.factor1)

        self.stage5 = HourglassStage(
            stage_layers[4], d_model, n_heads, d_ff, dropout, max_position)
        self.stage5_conditioned = CrossAttentionCondition(
            d_model, n_heads, d_ff, dropout, max_position)


    def forward(
        self,
        x: torch.Tensor,
        latent_condition: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        is_casual: Optional[bool] = True
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

        if position_ids is None:
            position_ids = torch.arange(
                seq_len,
                dtype=torch.long,
                device=x.device
            ).unsqueeze(0).expand(batch_size, -1)

        # Stage 1: Coordinate level
        stage1_out = self.stage1(x, is_casual, position_ids)
        stage1_conditioned_out = self.stage1_conditioned(
            stage1_out, latent_condition, latent_condition, mask=None, position_ids=position_ids)
        # Store for residual connection

        residual1 = stage1_conditioned_out
        # residual1_raw = stage1_conditioned_out
        # residual1 = F.pad(residual1_raw, (0, 0, 1, 0))[:, :-1, :]

        # Shortening 1: Coordinate -> Vertex (2x reduction)
        # tokens [x0,y0,x1,y1,x2,y2,x3,y3] => [y0,y1,y2,y3]
        shortened1 = self.shortening1(stage1_conditioned_out)
        position_ids_shortened1 = position_ids[:, self.factor1-1::self.factor1]
        # Stage 2: Coordinate level
        stage2_out = self.stage2(
            shortened1, is_casual, position_ids_shortened1)
        stage2_conditioned_out = self.stage2_conditioned(
            stage2_out, latent_condition, latent_condition, mask=None, position_ids=position_ids_shortened1)

        # Store for residual connection
        residual2 = stage2_conditioned_out
        # residual2_raw = stage2_conditioned_out
        # residual2 = F.pad(residual2_raw, (0, 0, 1, 0))[:, :-1, :]
        #
        # Shortening 2: Vertex -> Face (4x reduction)
        shortened2 = self.shortening2(stage2_conditioned_out)
        position_ids_shortened2 = position_ids_shortened1[:, self.factor2-1::self.factor2]
        # Stage 3: Face level
        stage3_out = self.stage3(
            shortened2, is_casual, position_ids_shortened2)
        stage3_conditioned_out = self.stage3_conditioned(
            stage3_out, latent_condition, latent_condition, mask=None, position_ids=position_ids_shortened2)

        # Upsampling 2: Face -> Vertex
        upsampled2 = self.upsampling2(
            stage3_conditioned_out, target_len=residual2.shape[1])
        #

        # Add residual connection
        # combined2 = upsampled2  # + self.residual_proj2(residual2)
        combined2 = upsampled2 + residual2

        # Stage 4: Reconsturcted Vertex level
        stage4_out = self.stage4(combined2, is_casual, position_ids_shortened1)
        stage4_conditioned_out = self.stage4_conditioned(
            stage4_out, latent_condition, latent_condition, mask=None, position_ids=position_ids_shortened1)

        # Upsampling 1: Vertex -> Coordinate
        upsampled1 = self.upsampling1(
            stage4_conditioned_out, target_len=seq_len)

        # Add residual connection
        # combined1 = upsampled1  # + self.residual_proj1(residual1)
        # combined1 = upsampled1_shifted + residual1
        combined1 = upsampled1 + residual1

        # Stage 5: Reconsturcted Coordinate level
        stage5_out = self.stage5(combined1, is_casual, position_ids)
        output_conditioned = self.stage5_conditioned(
            stage5_out, latent_condition, latent_condition, mask=None, position_ids=position_ids)

        return output_conditioned
        # return stage1_conditioned_out
