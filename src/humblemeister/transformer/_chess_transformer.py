from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from humblemeister.attention import KVCache, LayerKVCache, make_causal_mask
from humblemeister.embedding import InputEmbedding

from ._transformer import TransformerBlock


class ChessTransformer(nn.Module):
    __input_embedding: InputEmbedding
    __blocks: nn.ModuleList
    __norm: nn.LayerNorm
    __output: nn.Linear
    __value_head: nn.Sequential
    __n_layers: int

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.__input_embedding = InputEmbedding(vocab_size, d_model, max_seq_len, dropout)
        self.__blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.__norm = nn.LayerNorm(d_model)
        self.__output = nn.Linear(d_model, vocab_size, bias=False)
        self.__value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Tanh(),
        )
        self.__n_layers = n_layers

        # weight tying — share weights between input embedding and output projection
        # the output projection maps d_model → vocab_size, which is the transpose
        # of the embedding matrix vocab_size → d_model
        # this reduces parameters and improves performance
        self.__output.weight = self.__input_embedding.token_embedding.embedding.weight

        self.init_weights()

    @property
    def output(self) -> nn.Linear:
        return self.__output

    # def __init_weights(self) -> None:
    #     for p in self.parameters():
    #         if p.dim() > 1:
    #             nn.init.xavier_uniform_(p)

    def init_weights(self) -> None:
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)
            if "W_o" in name:  # scale down output projections
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.__n_layers))

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Full-sequence forward pass (training and no-cache generation).

        Pass is_causal=True for generation (no padding mask) to enable the
        Flash Attention kernel. Pass mask=combined_mask for training where
        a padding mask is needed alongside causal masking.

        Returns:
            logits: [batch, seq, vocab_size]
            value:  [batch, seq]  — position evaluation in [-1, 1] from White's perspective
        """
        if not is_causal and mask is None:
            mask = make_causal_mask(x.size(1), x.device)

        out = self.__input_embedding(x)

        for block in self.__blocks:
            out, _ = block(out, mask, kv_cache=None, is_causal=is_causal)

        hidden = self.__norm(out)
        logits = self.output(hidden)  # [batch, seq, vocab_size]
        value = self.__value_head(hidden).squeeze(-1)  # [batch, seq]
        return logits, value

    def generate_step(
        self,
        token: torch.Tensor,
        kv_cache: KVCache,
    ) -> tuple[torch.Tensor, torch.Tensor, KVCache]:
        """
        Generation path — single token, uses KV cache.

        Args:
            token:    [batch, 1] — the latest token only
            kv_cache: cache of K and V from all previous steps

        Returns:
            logits:    [batch, 1, vocab_size]
            value:     [batch, 1]  — position evaluation in [-1, 1] from White's perspective
            new_cache: updated KVCache with new K and V appended
        """
        # no mask needed — we're only attending to past tokens which are all valid
        out = self.__input_embedding(token)
        new_layers: list[LayerKVCache] = []

        for i, block in enumerate(self.__blocks):
            layer_cache = kv_cache.layers[i] if not kv_cache.is_empty() else None
            out, new_layer_cache = block(out, mask=None, kv_cache=layer_cache)
            new_layers.append(new_layer_cache)

        hidden = self.__norm(out)
        logits = self.output(hidden)
        value = self.__value_head(hidden).squeeze(-1)  # [batch, 1]
        return logits, value, KVCache(layers=new_layers)
