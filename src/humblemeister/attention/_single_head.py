from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SingleHeadAttention(nn.Module):
    __d_k: int
    __W_q: nn.Linear
    __W_k: nn.Linear
    __W_v: nn.Linear

    def __init__(self, d_model: int, d_k: int) -> None:
        super().__init__()
        self.__d_k = d_k

        # linear projections for Q, K, V
        self.__W_q = nn.Linear(d_model, d_k, bias=False)
        self.__W_k = nn.Linear(d_model, d_k, bias=False)
        self.__W_v = nn.Linear(d_model, d_k, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x shape: [batch_size, seq_len, d_model]

        # [batch_size, seq_len, d_k]
        Q = self.__W_q(x)
        K = self.__W_k(x)
        V = self.__W_v(x)

        # attention scores — dot product between every Q and K pair
        # Q @ K.transpose(-2, -1) -> [batch_size, seq_len, seq_len]
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.__d_k)

        # apply causal mask — prevents token i from attending to tokens > i
        if mask is not None:
            scores = scores + mask  # mask contains 0.0 or -inf

        # softmax over last dimension — each token's attention distribution
        weights = F.softmax(scores, dim=-1)  # [batch_size, seq_len, seq_len]

        # weighted sum of values
        # [batch_size, seq_len, d_k]
        out = weights @ V

        return out


def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    # upper triangle filled with -inf, diagonal and below are 0.0
    mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device),
        diagonal=1,
    )
    # [seq_len, seq_len]
    return mask

def make_padding_mask_bugged(attention_mask: torch.Tensor) -> torch.Tensor:
    # attention_mask: [batch, seq_len] — 1 for real tokens, 0 for padding
    # we need to broadcast to [batch, 1, 1, seq_len] so it combines with
    # the causal mask of shape [seq_len, seq_len]
    # result: [batch, 1, 1, seq_len] — 0.0 for real, -inf for padding
    return (1 - attention_mask).unsqueeze(1).unsqueeze(2) * float("-inf")

def make_padding_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    # attention_mask: [batch, seq_len] — 1 for real tokens, 0 for padding
    # output:         [batch, 1, 1, seq_len] — 0.0 for real, -inf for padding
    return torch.where(
        attention_mask.unsqueeze(1).unsqueeze(2).bool(),
        torch.zeros_like(attention_mask.unsqueeze(1).unsqueeze(2), dtype=torch.float),
        torch.full_like(attention_mask.unsqueeze(1).unsqueeze(2), float("-inf"), dtype=torch.float),
    )
