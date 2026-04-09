from __future__ import annotations

from dataclasses import dataclass

import chess
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from ._tokenizer import ChessTokenizer


@dataclass
class GameRecord:
    moves: list[chess.Move]
    outcome: float  # 1.0 = white win, 0.0 = black win, 0.5 = draw
    tensor: torch.Tensor
    move_weights: torch.Tensor | None = None  # [n_moves + 1], aligned with targets
    value_evals: torch.Tensor | None = (
        None  # [n_moves + 1], aligned with input_ids; tanh-normalized White's perspective
    )


class ChessDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    __tokenizer: ChessTokenizer
    __games: list[GameRecord]

    def __init__(self, tokenizer: ChessTokenizer) -> None:
        self.__tokenizer = tokenizer
        self.__games = list[GameRecord]()

    @property
    def games(self) -> list[GameRecord]:
        return self.__games

    def add_game(self, record: GameRecord) -> None:
        self.__games.append(record)

    def clear(self) -> None:
        self.__games = []

    def __len__(self) -> int:
        return len(self.__games)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        record = self.__games[idx]
        if record.tensor is None:
            raise TypeError(f"unexpected None at index {idx}")
        n_targets = len(record.tensor) - 1
        # if no weights, fall back to uniform (all ones, same length as targets)
        weights = (
            record.move_weights
            if record.move_weights is not None
            else torch.ones(n_targets, dtype=torch.float32)
        )
        # if no value evals, fall back to zeros (masked out during value loss)
        value_evals = (
            record.value_evals
            if record.value_evals is not None
            else torch.zeros(n_targets, dtype=torch.float32)
        )
        return record.tensor, weights, value_evals

    def collate(
        self, batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        # batch is a list of (game_tensor, move_weights, value_evals) triples from __getitem__.
        tensors: list[torch.Tensor] = [item[0] for item in batch]
        weights: list[torch.Tensor] = [item[1] for item in batch]
        val_evals: list[torch.Tensor] = [item[2] for item in batch]

        # pad all game tensors to the same length — shorter sequences get PAD appended
        padded = pad_sequence(tensors, batch_first=True, padding_value=self.__tokenizer.PAD)

        # pad weight tensors to the same length — padding positions get weight 0.0
        # (these positions are already ignored by ignore_index=PAD in the loss, so the value doesn't matter)
        padded_weights = pad_sequence(weights, batch_first=True, padding_value=0.0)

        # pad value_evals to the same length — padding positions get 0.0 and are masked out
        padded_value_evals = pad_sequence(val_evals, batch_first=True, padding_value=0.0)

        # shift by one: input_ids is the context, targets is what the model should predict next
        # game_tensor = [BOS, m1, m2, ..., mN, EOS]
        # input_ids   = [BOS, m1, m2, ..., mN]       ← fed into the model
        # targets     = [     m1, m2, ..., mN, EOS]  ← what each position should predict
        input_ids = padded[:, :-1]
        targets = padded[:, 1:]
        attention_mask = (input_ids != self.__tokenizer.PAD).long()

        # move_weights[b, t] is the importance of correctly predicting targets[b, t].
        # weight tensor has shape [N+1] (N moves + EOS), already aligned with targets length.
        move_weights = padded_weights[:, : targets.size(1)]

        # value_evals[b, t] is the Stockfish eval of the position at input_ids[b, t].
        # aligned with input_ids (not targets), same length after trimming.
        value_evals = padded_value_evals[:, : input_ids.size(1)]

        # has_value_evals[b] is True if this game had real Stockfish evals (not zero fallback)
        has_value_evals = torch.tensor(
            [item[2].any().item() for item in batch],
            dtype=torch.bool,
        )

        return {
            "input_ids": input_ids,
            "targets": targets,
            "attention_mask": attention_mask,
            "move_weights": move_weights,
            "value_evals": value_evals,
            "has_value_evals": has_value_evals,
        }
