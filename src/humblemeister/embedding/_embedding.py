from __future__ import annotations

import math
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    __embedding: nn.Embedding

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.__embedding = nn.Embedding(vocab_size, d_model)

    @property
    def embedding(self) -> nn.Embedding:
        return self.__embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len] — integer token IDs
        # output shape: [batch_size, seq_len, d_model]
        return self.__embedding(x)


class PositionalEncoding(nn.Module):
    __dropout: nn.Dropout
    __pos_encoding: torch.Tensor

    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.__dropout = nn.Dropout(dropout)

        # build the encoding matrix once, shape: [max_seq_len, d_model]
        pe = torch.zeros(max_seq_len, d_model)

        position = torch.arange(0, max_seq_len).unsqueeze(1).float()  # [max_seq_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # [d_model / 2]

        pe[:, 0::2] = torch.sin(position * div_term)  # even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dimensions

        # add batch dimension → [1, max_seq_len, d_model]
        pe = pe.unsqueeze(0)

        # register as buffer — not a parameter, but part of the model state
        self.__pos_encoding = pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, d_model]
        # self.pe[:, :seq_len] broadcasts across the batch dimension
        x = x + self.__pos_encoding[:, : x.size(1)]
        return self.__dropout(x)


class LearnedPositionalEncoding(nn.Module):
    __dropout: nn.Dropout
    __pos_embedding: nn.Embedding

    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.__dropout = nn.Dropout(dropout)
        self.__pos_embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)     # [seq_len]
        pos_emb = self.__pos_embedding(positions)              # [seq_len, d_model]
        return self.__dropout(x + pos_emb)                     # broadcasts over batch


class InputEmbedding(nn.Module):
    __token_embedding: TokenEmbedding
    __positional_encoding: PositionalEncoding | LearnedPositionalEncoding
    __d_model: int

    @property
    def token_embedding(self) -> TokenEmbedding:
        return self.__token_embedding

    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int = 512, dropout: float = 0.1, learned_pos_encoding = True) -> None:
        super().__init__()
        self.__token_embedding = TokenEmbedding(vocab_size, d_model)
        self.__positional_encoding = LearnedPositionalEncoding(d_model, max_seq_len, dropout) if learned_pos_encoding else PositionalEncoding(d_model, max_seq_len, dropout)
        self.__d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # scale embeddings by sqrt(d_model) — from the original transformer paper
        # this keeps the embedding values in a reasonable range relative to
        # the positional encoding values before adding them together
        token_emb = self.__token_embedding(x) * math.sqrt(self.__d_model)
        return self.__positional_encoding(token_emb)
