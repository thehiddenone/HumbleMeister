from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..attention import LayerKVCache, MultiHeadAttention


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, seq_len, d_model]
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attention   = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1       = nn.LayerNorm(d_model)
        self.norm2       = nn.LayerNorm(d_model)
        self.dropout     = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        kv_cache: LayerKVCache | None = None,
    ) -> tuple[torch.Tensor, LayerKVCache]:
        attn_out, new_cache = self.attention(self.norm1(x), mask, kv_cache)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x, new_cache
