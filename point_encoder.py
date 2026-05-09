import torch.nn as nn
import torch
import math


def fourier_features(x, n_freqs=8):
    """
    x: [B, N, 2]
    Encodes each coordinate with sin/cos at 2^0 ... 2^(n_freqs-1) frequencies.
    returns: [B, N, 2 + 4*n_freqs]
    """
    freqs = 2.0 ** torch.arange(n_freqs, device=x.device, dtype=x.dtype)  # [n_freqs]
    x_freq = x.unsqueeze(-1) * freqs * math.pi  # [B, N, 2, n_freqs]
    sin_feat = torch.sin(x_freq).flatten(-2)     # [B, N, 2*n_freqs]
    cos_feat = torch.cos(x_freq).flatten(-2)     # [B, N, 2*n_freqs]
    return torch.cat([x, sin_feat, cos_feat], dim=-1)  # [B, N, 2 + 4*n_freqs]


class PerceiverPointEncoder(nn.Module):

    def __init__(self, d_model=128, input_dim=2, n_latents=32, n_freqs=8):
        super().__init__()

        self.n_freqs = n_freqs
        fourier_dim = input_dim + 4 * n_freqs  # z.B. 2 + 32 = 34

        self.latents = nn.Parameter(torch.randn(n_latents, d_model))

        self.point_proj = nn.Linear(fourier_dim, d_model)

        # Cross-attention: latents attend to points
        self.cross_attention = nn.MultiheadAttention(
            d_model, num_heads=4, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff_cross = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

        # Self-attention between latents
        self.self_attention = nn.MultiheadAttention(
            d_model, num_heads=4, batch_first=True
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.ff_self = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, points):
        """
        points: [batch, n_points, 2]
        output: [batch, n_latents, d_model]
        """
        batch_size = points.shape[0]

        point_features = fourier_features(points, self.n_freqs)  # [B, N, 2+4*n_freqs]
        point_features = self.point_proj(point_features)         # [B, N, d_model]

        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)

        # Cross-attention block (Pre-Norm)
        normed = self.norm1(latents)
        attn_out, _ = self.cross_attention(
            query=normed, key=point_features, value=point_features
        )
        latents = latents + attn_out
        latents = latents + self.ff_cross(self.norm2(latents))

        # Self-attention block (Pre-Norm)
        normed = self.norm3(latents)
        attn2_out, _ = self.self_attention(normed, normed, normed)
        latents = latents + attn2_out
        latents = latents + self.ff_self(self.norm4(latents))

        return latents  # [batch, n_latents, d_model]
