from __future__ import annotations

import json
from pathlib import Path
from typing import overload

import chess
import chess.svg
import torch

from .data import ChessTokenizer
from .inference import sample_move
from .trainer import ChessTrainingConfig
from .transformer import ChessTransformer


class ChessEngine:
    def __init__(
        self,
        model: ChessTransformer,
        tokenizer: ChessTokenizer,
        device: str = "cpu",
        temperature: float = 1.0,
        value_weight: float = 0.0,
    ) -> None:
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.temperature = temperature
        self.value_weight = value_weight
        self.player_color: chess.Color | None = None
        self.board = chess.Board()
        self.move_history: list[int] = [self.tokenizer.BOS]

        self.model.eval()

    # ------------------------------------------------------------------ #
    #  Loading                                                             #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_safetensors(
        cls,
        path: str,
        device: str = "cpu",
        temperature: float = 1.0,
        value_weight: float = 0.0,
    ) -> ChessEngine:
        from safetensors.torch import load_file

        with open(f"{path}/config.json") as f:
            config = ChessTrainingConfig(**json.load(f))

        tokenizer = ChessTokenizer()
        model = cls._build_model(config, tokenizer)
        state_dict = load_file(f"{path}/model.safetensors", device=device)

        # __output.weight is tied to the embedding weight and deduped by safetensors —
        # it won't appear as a separate key, but strict=False is safe because loading
        # the embedding tensor in-place also updates __output.weight (same object)
        incompatible = model.load_state_dict(state_dict, strict=False)
        unexpected_missing = [
            k for k in incompatible.missing_keys if k != "_ChessTransformer__output.weight"
        ]
        if unexpected_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                f"Error loading state_dict — "
                f"missing: {unexpected_missing}, unexpected: {incompatible.unexpected_keys}"
            )

        return cls(model, tokenizer, device, temperature, value_weight)

    @classmethod
    def from_pt(
        cls,
        path: str,
        device: str = "cpu",
        temperature: float = 1.0,
        value_weight: float = 0.0,
    ) -> ChessEngine:
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint["config"]
        tokenizer = ChessTokenizer()
        model = cls._build_model(config, tokenizer)
        model.load_state_dict(checkpoint["model_state"])

        return cls(model, tokenizer, device, temperature, value_weight)

    @classmethod
    def load(
        cls,
        path: str,
        device: str = "cpu",
        temperature: float = 1.0,
        value_weight: float = 0.0,
    ) -> ChessEngine:
        """Load from either a safetensors directory or a .pt file."""
        p = Path(path)
        if p.is_dir() and (p / "model.safetensors").exists():
            return cls.from_safetensors(path, device, temperature, value_weight)
        elif p.suffix == ".pt":
            return cls.from_pt(path, device, temperature, value_weight)
        else:
            raise ValueError(f"could not find a valid model at {path}")

    @staticmethod
    def _build_model(config: ChessTrainingConfig, tokenizer: ChessTokenizer) -> ChessTransformer:
        return ChessTransformer(
            vocab_size=tokenizer.vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
        )

    # ------------------------------------------------------------------ #
    #  Inference                                                           #
    # ------------------------------------------------------------------ #

    def _sample_move(self) -> chess.Move:
        return sample_move(
            model=self.model,
            tokenizer=self.tokenizer,
            board=self.board,
            move_history=self.move_history,
            device=self.device,
            temperature=self.temperature,
            value_weight=self.value_weight,
        )

    def _apply_move(self, move: chess.Move) -> None:
        if move not in self.board.legal_moves:
            raise ValueError(f"{move.uci()} is not a legal move in the current position")
        self.board.push(move)
        self.move_history.append(self.tokenizer.encode_move(move))

    def move(self, player_move: chess.Move | str | None = None) -> chess.Board:
        """
        Play one full turn.

        Args:
            player_move: the move you want to play, as a chess.Move or UCI string
                         e.g. chess.Move.from_uci("e2e4") or just "e2e4"
                         if None, the model plays both sides (self-play)

        Returns:
            the current chess.Board — renderable in Jupyter via chess.svg
        """
        if self.board.is_game_over():
            print(f"game over — {self.board.result()}")
            return self.board

        if player_move is None:
            # self-play: model plays whoever's turn it is
            model_move = self._sample_move()
            print(f"model plays: {model_move.uci()}")
            self._apply_move(model_move)
            if self.board.is_game_over():
                print(f"game over — {self.board.result()}")
            return self.board

        # player's move
        if isinstance(player_move, str):
            player_move = chess.Move.from_uci(player_move)
        self._apply_move(player_move)
        if self.board.is_game_over():
            print(f"game over — {self.board.result()}")
            return self.board

        # model responds
        model_move = self._sample_move()
        print(f"model plays: {model_move.uci()}")
        self._apply_move(model_move)
        if self.board.is_game_over():
            print(f"game over — {self.board.result()}")

        return self.board

    def start_game(self, player_color: chess.Color = chess.WHITE) -> chess.Board:
        """Reset the board and set which color the player controls.
        If the player is black, the model immediately plays white's first move."""
        self.player_color = player_color
        self.board = chess.Board()
        self.move_history = [self.tokenizer.BOS]

        if player_color == chess.BLACK:
            model_move = self._sample_move()
            print(f"model plays: {model_move.uci()}")
            self._apply_move(model_move)

        return self.board

    def reset(self) -> chess.Board:
        """Reset the board, keeping the current player color."""
        self.board = chess.Board()
        self.move_history = [self.tokenizer.BOS]
        return self.board

    def render(self) -> chess.svg.SvgWrapper:
        """Render the current board as SVG — displays inline in Jupyter."""
        lastmove = self.board.peek() if self.board.move_stack else None
        return chess.svg.board(
            self.board,
            lastmove=lastmove,
            size=400,
        )  # type: ignore

    def __repr__(self) -> str:
        return (
            f"FumblemeisterEngine(\n"
            f"  turn={('white' if self.board.turn == chess.WHITE else 'black')}\n"
            f"  moves={self.board.fullmove_number}\n"
            f"  result={self.board.result() if self.board.is_game_over() else 'ongoing'}\n"
            f")"
        )
