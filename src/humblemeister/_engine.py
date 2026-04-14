from __future__ import annotations

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


class ChessEngine:
    __model: ChessTransformer
    __tokenizer: ChessTokenizer
    __device: torch.device
    __temperature: float
    __value_weight: float
    __use_kv_cache: bool
    __player_color: chess.Color | None
    __board: chess.Board
    __move_history: list[int]
    __kv_cache: KVCache
    __kv_cache_tokens: int

    def __init__(
        self,
        model: ChessTransformer,
        tokenizer: ChessTokenizer,
        device: str = "cpu",
        temperature: float = 1.0,
        value_weight: float = 1.0,
        use_kv_cache: bool = True,
    ) -> None:
        """
        Args:
            model:        trained ChessTransformer; will be moved to device.
            tokenizer:    tokenizer matching the model's vocabulary.
            device:       torch device string — "cpu", "cuda", "cuda:0", etc.
            temperature:  softmax temperature for move sampling; <1 = sharper, >1 = more random.
            value_weight: blend policy logits with value head output.
                          0.0 = pure policy; higher values shift selection toward value-guided moves.
            use_kv_cache: reuse K/V from previous moves each turn; faster per-step but holds the
                          cache in memory for the duration of the game.
        """
        self.__model = model.to(device)
        self.__tokenizer = tokenizer
        self.__device = torch.device(device)
        self.__temperature = temperature
        self.__value_weight = value_weight
        self.__use_kv_cache = use_kv_cache
        self.__player_color = None
        self.__board = chess.Board()
        self.__move_history = [self.__tokenizer.BOS]
        self.__kv_cache = KVCache()
        self.__kv_cache_tokens = 0

        self.__model.eval()

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
        use_kv_cache: bool = False,
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

        return cls(model, tokenizer, device, temperature, value_weight, use_kv_cache)

    @classmethod
    def from_pt(
        cls,
        path: str,
        device: str = "cpu",
        temperature: float = 1.0,
        value_weight: float = 0.0,
        use_kv_cache: bool = False,
    ) -> ChessEngine:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        config_data = checkpoint["config"]
        config = ChessTrainingConfig(**config_data) if isinstance(config_data, dict) else config_data
        tokenizer = ChessTokenizer()
        model = cls._build_model(config, tokenizer)
        model.load_state_dict(checkpoint["model_state"])

        return cls(model, tokenizer, device, temperature, value_weight, use_kv_cache)

    @classmethod
    def load(
        cls,
        path: str,
        device: str = "cpu",
        temperature: float = 1.0,
        value_weight: float = 0.0,
        use_kv_cache: bool = False,
    ) -> ChessEngine:
        """Load from either a safetensors directory or a .pt file."""
        p = Path(path)
        if p.is_dir() and (p / "model.safetensors").exists():
            return cls.from_safetensors(path, device, temperature, value_weight, use_kv_cache)
        elif p.suffix == ".pt":
            return cls.from_pt(path, device, temperature, value_weight, use_kv_cache)
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
            pad_id=tokenizer.PAD,
        )

    # ------------------------------------------------------------------ #
    #  Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def player_color(self) -> chess.Color | None:
        return self.__player_color

    @property
    def board(self) -> chess.Board:
        return self.__board

    # ------------------------------------------------------------------ #
    #  Inference                                                           #
    # ------------------------------------------------------------------ #

    def __reset_kv_cache(self) -> None:
        self.__kv_cache = KVCache()
        self.__kv_cache_tokens = 0

    def sample_move(self) -> chess.Move:
        if self.__use_kv_cache:
            move, self.__kv_cache = sample_move_kv_cache(
                model=self.__model,
                tokenizer=self.__tokenizer,
                board=self.__board,
                move_history=self.__move_history,
                device=self.__device,
                temperature=self.__temperature,
                value_weight=self.__value_weight,
                cache=self.__kv_cache,
                cache_tokens=self.__kv_cache_tokens,
            )
            self.__kv_cache_tokens = len(self.__move_history)
            return move
        return sample_move(
            model=self.__model,
            tokenizer=self.__tokenizer,
            board=self.__board,
            move_history=self.__move_history,
            device=self.__device,
            temperature=self.__temperature,
            value_weight=self.__value_weight,
        )

    def apply_move(self, move: chess.Move) -> None:
        if move not in self.__board.legal_moves:
            raise ValueError(f"{move.uci()} is not a legal move in the current position")
        self.__board.push(move)
        self.__move_history.append(self.__tokenizer.encode_move(move))

    def move(
        self, player_move: chess.Move | str | None = None
    ) -> tuple[chess.Board, chess.Move | None]:
        """
        Play one full turn.

        Args:
            player_move: the move you want to play, as a chess.Move or UCI string
                         e.g. chess.Move.from_uci("e2e4") or just "e2e4"
                         if None, the model plays both sides (self-play)

        Returns:
            the current chess.Board — renderable in Jupyter via chess.svg
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
        self.__move_history = [self.__tokenizer.BOS]
        self.__reset_kv_cache()

        if player_color == chess.BLACK:
            model_move = self.sample_move()
            print(f"model plays: {model_move.uci()}")
            self.apply_move(model_move)

        return self.__board

    def reset(self) -> chess.Board:
        """Reset the board, keeping the current player color."""
        self.__board = chess.Board()
        self.__move_history = [self.__tokenizer.BOS]
        self.__reset_kv_cache()
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
            f"ChessEngine(\n"
            f"  turn={('white' if self.__board.turn == chess.WHITE else 'black')}\n"
            f"  moves={self.__board.fullmove_number}\n"
            f"  result={self.__board.result() if self.__board.is_game_over() else 'ongoing'}\n"
            f")"
        )
