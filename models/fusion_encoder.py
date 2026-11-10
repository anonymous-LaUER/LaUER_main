"""Signal-image fusion blocks with pre-normalized self-attention, cross-attention, and FFN."""
import torch.nn as nn

from models.attention import mha_fp32

class FusionLayer(nn.Module):
    """One signal-image fusion block; cross-attention reads the specified original token memory."""

    def __init__(self, hidden_size=768, num_heads=12, intermediate_size=3072, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size

        # Self-Attention
        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_size)

        # Cross-Attention
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_size)

        # Feed-Forward
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(hidden_size)

    def forward(self, primary, cross_kv, primary_padding_mask=None, cross_padding_mask=None):

        residual = primary
        primary_norm = self.norm1(primary)
        attn_out, _ = mha_fp32(
            self.self_attn, primary_norm, primary_norm, primary_norm,
            key_padding_mask=primary_padding_mask,
        )
        primary = residual + attn_out

        residual = primary
        primary_norm = self.norm2(primary)
        attn_out, _ = mha_fp32(
            self.cross_attn, primary_norm, cross_kv, cross_kv,
            key_padding_mask=cross_padding_mask,
        )
        primary = residual + attn_out

        # 3. Feed-Forward
        residual = primary
        primary_norm = self.norm3(primary)
        ffn_out = self.ffn(primary_norm)
        primary = residual + ffn_out

        return primary

class FusionEncoder(nn.Module):

    def __init__(self, hidden_size=768, num_heads=12, intermediate_size=3072,
                 num_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            FusionLayer(hidden_size, num_heads, intermediate_size, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, primary, cross_kv, primary_padding_mask=None, cross_padding_mask=None):
        for layer in self.layers:
            primary = layer(primary, cross_kv, primary_padding_mask, cross_padding_mask)
        return primary
