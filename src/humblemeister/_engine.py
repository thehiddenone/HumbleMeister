from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import chess
import chess.svg
import torch

from .attention import KVCache
from .config import ChessTrainingConfig
from .data import ChessTokenizer
from .inference import sample_move, sample_move_kv_cache
from .transformer import ChessTransformer


def _config_from_dict(raw: dict) -> ChessTrainingConfig:
    """Build a ChessTrainingConfig from a possibly-stale dict.

    Older checkpoints may carry fields that have since been removed from the
    config (e.g. `self_play_value_weight`). Silently drop any unknown keys so
    older artifacts still load.
    """
    valid = {f.name for f in dataclasses.fields(ChessTrainingConfig)}
    return ChessTrainingConfig(**{k: v for k, v in raw.items() if k in valid})


class ChessModel:
    """
    Holds the transformer weights, tokenizer and device.

    A single ChessModel can be shared across many ChessGame instances — only
    one copy of the weights lives in memory regardless of how many concurrent
    games are running.
    """

    __model: ChessTransformer
    __tokenizer: ChessTokenizer
    __device: torch.device

    def __init__(
        self,
        model: ChessTransformer,
        tokenizer: ChessTokenizer,
        device: str = "cpu",
    ) -> None:
        self.__model = model.to(device)
        self.__tokenizer = tokenizer
        self.__device = torch.device(device)
        self.__model.eval()

    @property
    def tokenizer(self) -> ChessTokenizer:
        return self.__tokenizer

    @property
    def device(self) -> torch.device:
        return self.__device

    def sample(
        self,
        board: chess.Board,
        move_history: list[int],
        temperature: float,
        blunder_threshold: float,
        is_self_play: bool,
        use_kv_cache: bool,
        kv_cache: KVCache,
        kv_cache_tokens: int,
    ) -> tuple[chess.Move, KVCache]:
        """
        Sample one move given the current game state.

        Returns (move, updated_kv_cache).  When use_kv_cache is False the
        returned cache is an empty KVCache (the caller should discard it).
        """
        if use_kv_cache:
            move, new_cache = sample_move_kv_cache(
                model=self.__model,
                tokenizer=self.__tokenizer,
                board=board,
                move_history=move_history,
                device=self.__device,
                temperature=temperature,
                blunder_threshold=blunder_threshold,
                is_self_play=is_self_play,
                cache=kv_cache,
                cache_tokens=kv_cache_tokens,
            )
            return move, new_cache

        move = sample_move(
            model=self.__model,
            tokenizer=self.__tokenizer,
            board=board,
            move_history=move_history,
            device=self.__device,
            temperature=temperature,
            blunder_threshold=blunder_threshold,
            is_self_play=is_self_play,
        )
        return move, KVCache()

    @classmethod
    def from_safetensors(cls, path: str, device: str = "cpu") -> ChessModel:
        from safetensors.torch import load_file

        with open(f"{path}/config.json") as f:
            config = _config_from_dict(json.load(f))

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

        return cls(model, tokenizer, device)

    @classmethod
    def from_pt(cls, path: str, device: str = "cpu") -> ChessModel:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        config_data = checkpoint["config"]
        config = (
            _config_from_dict(config_data) if isinstance(config_data, dict) else config_data
        )
        tokenizer = ChessTokenizer()
        model = cls._build_model(config, tokenizer)
        model.load_state_dict(checkpoint["model_state"])
        return cls(model, tokenizer, device)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> ChessModel:
        """Load from a safetensors directory or a .pt checkpoint file."""
        p = Path(path)
        if p.is_dir() and (p / "model.safetensors").exists():
            return cls.from_safetensors(path, device)
        elif p.suffix == ".pt":
            return cls.from_pt(path, device)
        else:
            raise ValueError(f"could not find a valid model at {path!r}")

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
            pad_id=tokenizer.PAD,
        )


class ChessGame:
    """
    Per-game state for one chess game against (or alongside) a ChessModel.

    Multiple ChessGame instances can share a single ChessModel, so only one
    copy of the model weights lives in memory regardless of concurrent game
    count.
    """

    __chess_model: ChessModel
    __temperature: float
    __blunder_threshold: float
    __is_self_play: bool
    __use_kv_cache: bool
    __player_color: chess.Color | None
    __board: chess.Board
    __move_history: list[int]
    __kv_cache: KVCache
    __kv_cache_tokens: int

    def __init__(
        self,
        chess_model: ChessModel,
        temperature: float = 1.0,
        blunder_threshold: float = 0.15,
        is_self_play: bool = False,
        use_kv_cache: bool = False,
    ) -> None:
        """
        Args:
            chess_model:       shared ChessModel instance (weights + tokenizer + device).
            temperature:       softmax temperature for move sampling; <1 = sharper, >1 = more random.
            blunder_threshold: max allowed value-head gap from the best candidate (in tanh-value units,
                               ~0.15 ≈ 60cp). Candidate moves with a larger gap are excluded before
                               picking. See MOVE_PICKING.md.
            is_self_play:      True → pick by sampling among survivors (exploration); False → argmax
                               among survivors (deterministic play).
            use_kv_cache:      reuse K/V from previous moves each turn; faster per-step but holds the
                               cache in memory for the duration of the game.
        """
        self.__chess_model = chess_model
        self.__temperature = temperature
        self.__blunder_threshold = blunder_threshold
        self.__is_self_play = is_self_play
        self.__use_kv_cache = use_kv_cache
        self.__player_color = None
        self.__board = chess.Board()
        self.__move_history = [chess_model.tokenizer.BOS]
        self.__kv_cache = KVCache()
        self.__kv_cache_tokens = 0

    @property
    def player_color(self) -> chess.Color | None:
        return self.__player_color

    @property
    def board(self) -> chess.Board:
        return self.__board

    def sample_move(self) -> chess.Move:
        move, new_cache = self.__chess_model.sample(
            board=self.__board,
            move_history=self.__move_history,
            temperature=self.__temperature,
            blunder_threshold=self.__blunder_threshold,
            is_self_play=self.__is_self_play,
            use_kv_cache=self.__use_kv_cache,
            kv_cache=self.__kv_cache,
            kv_cache_tokens=self.__kv_cache_tokens,
        )
        if self.__use_kv_cache:
            self.__kv_cache = new_cache
            self.__kv_cache_tokens = len(self.__move_history)
        return move

    def apply_move(self, move: chess.Move) -> None:
        if move not in self.__board.legal_moves:
            raise ValueError(f"{move.uci()} is not a legal move in the current position")
        self.__board.push(move)
        self.__move_history.append(self.__chess_model.tokenizer.encode_move(move))

    def move(
        self, player_move: chess.Move | str | None = None
    ) -> tuple[chess.Board, chess.Move | None]:
        """
        Play one full turn.

        Args:
            player_move: the move you want to play, as a chess.Move or UCI string
                         e.g. chess.Move.from_uci("e2e4") or just "e2e4".
                         If None, the model plays both sides (self-play).

        Returns:
            (board, model_move) — model_move is None when the game is over or
            when the player just moved into a game-over state.
        """
        if self.__board.is_game_over():
            print(f"game over — {self.__board.result()}")
            return self.__board, None

        if player_move is None:
            # self-play: model plays whoever's turn it is
            model_move = self.sample_move()
            print(f"model plays: {model_move.uci()}")
            self.apply_move(model_move)
            if self.__board.is_game_over():
                print(f"game over — {self.__board.result()}")
            return self.__board, model_move

        # player's move
        if isinstance(player_move, str):
            player_move = chess.Move.from_uci(player_move)
        self.apply_move(player_move)
        if self.__board.is_game_over():
            print(f"game over — {self.__board.result()}")
            return self.__board, None

        # model responds
        model_move = self.sample_move()
        print(f"model plays: {model_move.uci()}")
        self.apply_move(model_move)
        if self.__board.is_game_over():
            print(f"game over — {self.__board.result()}")

        return self.__board, model_move

    def start_game(self, player_color: chess.Color = chess.WHITE) -> chess.Board:
        """Reset the board and set which color the player controls.
        If the player is black, the model immediately plays white's first move."""
        self.__player_color = player_color
        self.__board = chess.Board()
        self.__move_history = [self.__chess_model.tokenizer.BOS]
        self.__kv_cache = KVCache()
        self.__kv_cache_tokens = 0

        if player_color == chess.BLACK:
            model_move = self.sample_move()
            print(f"model plays: {model_move.uci()}")
            self.apply_move(model_move)

        return self.__board

    def reset(self) -> chess.Board:
        """Reset the board, keeping the current player color."""
        self.__board = chess.Board()
        self.__move_history = [self.__chess_model.tokenizer.BOS]
        self.__kv_cache = KVCache()
        self.__kv_cache_tokens = 0
        return self.__board

    def render(self) -> chess.svg.SvgWrapper:
        """Render the current board as SVG — displays inline in Jupyter."""
        lastmove = self.__board.peek() if self.__board.move_stack else None
        return chess.svg.board(
            self.__board,
            lastmove=lastmove,
            size=400,
        )  # type: ignore

    def __repr__(self) -> str:
        return (
            f"ChessGame(\n"
            f"  turn={('white' if self.__board.turn == chess.WHITE else 'black')}\n"
            f"  moves={self.__board.fullmove_number}\n"
            f"  result={self.__board.result() if self.__board.is_game_over() else 'ongoing'}\n"
            f")"
        )
