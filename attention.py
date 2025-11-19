import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8, dropout=0.1):
        super().__init__()

        assert d_model % n_heads == 0, "d_model muss durch n_heads teilbar sein"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension pro Head

        # Linear layers für Q, K, V
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        # Output projection
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q,k,v, mask=None):
        """
        Args:
            q,k,v: [batch_size, seq_len, d_model] oder [seq_len, d_model]
            mask: Optional attention mask

        Returns:
            output: Same shape as input
        """
        # Handle single sequence vs batch
        if q.dim() == 2:
            q = q.unsqueeze(0)  # Add batch dimension
            k = k.unsqueeze(0)  # Add batch dimension
            v = v.unsqueeze(0)  # Add batch dimension
            single_sequence = True
        else:
            single_sequence = False

        batch_size, seq_len, d_model = q.shape

        # 1. Linear transformations for Q, K, V
        Q = self.w_q(q)  # [batch_size, seq_len, d_model]
        K = self.w_k(k)  # [batch_size, seq_len, d_model]
        V = self.w_v(v)  # [batch_size, seq_len, d_model]

        # 2. Reshape for multi-head attention
        # [batch_size, seq_len, d_model] -> [batch_size, n_heads, seq_len, d_k]
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 3. Scaled dot-product attention
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # 4. Concatenate heads
        # [batch_size, n_heads, seq_len, d_k] -> [batch_size, seq_len, d_model]
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )

        # 5. Final linear projection
        output = self.w_o(attention_output)

        # Remove batch dimension if input was single sequence
        if single_sequence:
            output = output.squeeze(0)

        return output

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Args:
            Q, K, V: [batch_size, n_heads, seq_len, d_k]
            mask: Optional mask

        Returns:
            output: [batch_size, n_heads, seq_len, d_k]
        """
        d_k = Q.size(-1)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply mask if not provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        output = torch.matmul(attention_weights, V)

        return output

