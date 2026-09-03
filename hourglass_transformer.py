"""
Flat Transformer Architecture (simplified from Hourglass)
All stages operate at full sequence length — no shortening or upsampling.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from attention import MultiHeadAttention


class HourglassTransformerBlock(nn.Module):
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

    def forward(self, x: torch.Tensor, is_casual: Optional[bool] = True, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        if is_casual:
            _, seq_len, _ = x.shape
            mask = self._causal_mask(seq_len, x.device)
        else:
            mask = None

        attn_out = self.attention(
            x, x, x, mask=mask, position_ids=position_ids)
        x = self.norm1(x + attn_out)
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        return x

    def _causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(
            (seq_len, seq_len), device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        mask = mask.masked_fill(mask == 0, 0.0)
        return mask


class CrossAttentionCondition(nn.Module):
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
        attn_out = self.attention(
            query, key, value, mask=mask, position_ids=position_ids)
        x = self.norm1(query + attn_out)
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        return x


class HourglassStage(nn.Module):
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
    Flat Transformer — same interface as the Hourglass version.
    Each stage runs at full sequence length followed by cross-attention conditioning.
    No shortening or upsampling, so there is no information leak.
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
        super().__init__()
        self.d_model = d_model

        self.stages = nn.ModuleList([
            HourglassStage(n, d_model, n_heads, d_ff, dropout, max_position)
            for n in stage_layers
        ])
        self.conditioners = nn.ModuleList([
            CrossAttentionCondition(
                d_model, n_heads, d_ff, dropout, max_position)
            for _ in stage_layers
        ])

    def forward(
        self,
        x: torch.Tensor,
        latent_condition: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        is_casual: Optional[bool] = True
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        if position_ids is None:
            position_ids = torch.arange(
                seq_len, dtype=torch.long, device=x.device
            ).unsqueeze(0).expand(batch_size, -1)

        for stage, conditioner in zip(self.stages, self.conditioners):
            x = stage(x, is_casual, position_ids)
            x = conditioner(x, latent_condition, latent_condition,
                            mask=None, position_ids=position_ids)

        return x
