from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._single_head import SingleHeadAttention


@dataclass
class LayerKVCache:
    k: torch.Tensor  # [batch, n_heads, seq_len, d_k]
    v: torch.Tensor  # [batch, n_heads, seq_len, d_k]


@dataclass
class KVCache:
    """One LayerKVCache per transformer block."""

    layers: list[LayerKVCache] = field(default_factory=list)

    def is_empty(self) -> bool:
        return len(self.layers) == 0


class SlowMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # each head works on a slice of d_model

        self.heads = nn.ModuleList([SingleHeadAttention(d_model, self.d_k) for _ in range(n_heads)])

        # final projection back to d_model after concatenating heads
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # run all heads in parallel, each produces [batch, seq_len, d_k]
        head_outputs = [head(x, mask) for head in self.heads]

        # concatenate along last dimension → [batch, seq_len, d_model]
        concat = torch.cat(head_outputs, dim=-1)

        # final linear projection
        return cast(torch.Tensor, self.W_o(concat))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # all heads' Q, K, V projections in a single matrix each
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        kv_cache: LayerKVCache | None = None,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, LayerKVCache]:
        batch_size, seq_len, _ = x.shape

        # project to Q, K, V — still [batch, seq_len, d_model]
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # reshape to split d_model into n_heads × d_k
        # [batch, seq_len, d_model] → [batch, seq_len, n_heads, d_k]
        # then transpose → [batch, n_heads, seq_len, d_k]
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # append to cache if it exists
        if kv_cache is not None:
            K = torch.cat([kv_cache.k, K], dim=2)  # grow along seq_len dimension
            V = torch.cat([kv_cache.v, V], dim=2)

        # store updated cache
        new_cache = LayerKVCache(k=K, v=V)

        # Use is_causal=True for generation passes — activates the Flash Attention
        # kernel which never materialises the [batch, heads, seq, seq] score matrix.
        # Float additive masks (training, combined causal+padding) force a fallback
        # to the math backend which is O(T²) in memory, so avoid them during inference.
        out = F.scaled_dot_product_attention(
            Q,
            K,
            V,
            attn_mask=None if is_causal else mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=is_causal,
        )

        # merge heads back — transpose → [batch, seq_len, n_heads, d_k]
        # contiguous() needed before view() after transpose
        # then view → [batch, seq_len, d_model]
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.W_o(out), new_cache
