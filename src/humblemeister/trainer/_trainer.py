from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import chess
import safetensors.torch as st
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from humblemeister.attention import KVCache, LayerKVCache, make_causal_mask, make_padding_mask
from humblemeister.data import ChessDataset, ChessGameBank, ChessTokenizer, GameRecord
from humblemeister.evaluation import AsyncBatchEvaluator, StockfishEvaluator, compute_move_weights
from humblemeister.transformer import ChessTransformer

from ._loss_tracker import LossBreakthroughDetector


def _save_temp_batch(batch: list[dict[str, Any]]) -> Path:
    """Write a list of game dicts to a temp file and return its path."""
    fd, path_str = tempfile.mkstemp(suffix=".pt", prefix="chess_sp_")
    os.close(fd)
    path = Path(path_str)
    torch.save(batch, path)
    return path


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.mps.is_available():
        return "mps"
    return "cpu"


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    warmup_epochs: int,
) -> LambdaLR:
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return epoch / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(n_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


@dataclass
class ChessTrainingConfig:
    model_name: str

    # model: tiny
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    train_batch_size: int = 256

    max_seq_len: int = 512
    dropout: float = 0.1
    bf16: bool = True  # bfloat16 mixed precision — halves activation memory, no GradScaler needed

    # training
    n_games: int = 512  # total games loaded per epoch
    n_epochs: int = 2000
    lr: float = 3e-4
    device: str = get_device()

    max_moves: int = 500
    warmup_epochs: int = 50
    outcome_warmup: int = 20000

    # self-play curriculum — ramps in gradually after self_play_start_epoch
    self_play_start_epoch: int = 1000  # all bank games before this
    self_play_ramp_epochs: int = 400  # epochs to ramp from 0% → max ratio
    self_play_max_ratio: float = 0.1  # cap at 30% self-play games
    self_play_batch_size: int = 64  # games per generation sub-batch

    # stockfish evaluation
    use_stockfish: bool = True
    stockfish_path: str = "stockfish"
    stockfish_depth: int = 5  # depth 5 ≈ 1-5ms/position
    stockfish_workers: int = 24  # parallel engine processes for board evaluation
    advantage_temperature: float = 1.0  # sharpness of per-move weight distribution
    value_loss_weight: float = 0.5  # relative weight of value loss vs policy loss
    self_play_kv_cache: bool = True  # use KV cache during self-play generation
    self_play_value_weight: float = 0.5  # weight for value-blended move selection during self-play
    self_play_max_moves: int = 120  # hard draw cap for self-play games
    streaming: bool = False  # stream games in chunks instead of generating all at once
    streaming_chunk_size: int = 64  # games per streaming chunk (generation + grad accumulation)

    # checkpointing
    checkpoint_dir: str = "env/checkpoints"
    checkpoint_every: int = 50  # save every N epochs
    keep_last_n: int = 10  # keep only the last N checkpoints

    # logging
    log_dir: str = "env/logs"
    log_every: int = 1

    @classmethod
    def tiny(cls, name: str) -> ChessTrainingConfig:
        result = cls(name)
        result.d_model = 128
        result.n_heads = 4
        result.n_layers = 4
        result.d_ff = 512
        result.train_batch_size = 512
        return result

    @classmethod
    def small(cls, name: str) -> ChessTrainingConfig:
        result = cls(name)
        result.d_model = 256
        result.n_heads = 4
        result.n_layers = 4
        result.d_ff = 1024
        result.train_batch_size = 512
        return result

    @classmethod
    def medium(cls, name: str) -> ChessTrainingConfig:
        result = cls(name)
        result.d_model = 512
        result.n_heads = 8
        result.n_layers = 8
        result.d_ff = 2048
        result.train_batch_size = 256
        result.n_games = 1024
        return result

    @classmethod
    def large(cls, name: str) -> ChessTrainingConfig:
        result = cls(name)
        result.d_model = 768
        result.n_heads = 12
        result.n_layers = 12
        result.d_ff = 3072
        result.train_batch_size = 64
        result.n_games = 2048
        return result

    @classmethod
    def huge(cls, name: str) -> ChessTrainingConfig:
        result = cls(name)
        result.d_model = 1024
        result.n_heads = 16
        result.n_layers = 16
        result.d_ff = 4096
        result.train_batch_size = 64
        result.n_games = 2048
        return result

    @classmethod
    def giant(cls, name: str) -> ChessTrainingConfig:
        result = cls(name)
        result.d_model = 1536
        result.n_heads = 24
        result.n_layers = 24
        result.d_ff = 6144
        result.train_batch_size = 4
        result.self_play_kv_cache = False
        result.streaming = True
        result.streaming_chunk_size = 12
        result.n_games = 4096
        return result

    @classmethod
    def uber(cls, name: str) -> ChessTrainingConfig:
        result = cls(name)
        result.d_model = 2048
        result.n_heads = 32
        result.n_layers = 32
        result.d_ff = 8192
        result.train_batch_size = 2
        result.streaming = True
        result.streaming_chunk_size = 64
        result.n_games = 8192
        return result

    def save(self, file_path: str | Path) -> None:
        with Path(file_path).open(mode="w") as f:
            json.dump(asdict(self), f)

    @classmethod
    def from_file(cls, file_path: str | Path) -> ChessTrainingConfig:
        with Path(file_path).open() as f:
            d = json.load(f)
            if not isinstance(d, dict):
                raise TypeError(
                    f"Cannot load ChessTrainingConfig: expected dict, got {type(d).__name__}"
                )
            if "model_name" not in d:
                raise ValueError(f"Cannot load ChessTrainingConfig: model_name is not defined")
            model_name = d["model_name"]
            del d["model_name"]
            return cls(model_name, **d)


class ChessTrainer:
    __config: ChessTrainingConfig
    __device: torch.device
    __model: ChessTransformer
    __optimizer: AdamW
    __scheduler: LambdaLR
    __tokenizer: ChessTokenizer
    __dataset: ChessDataset
    __gamebank: ChessGameBank
    __start_epoch: int
    __last_loss: float
    __disable_selfplay: bool
    __selfplay_min_override: float | None
    __selfplay_max_override: float | None
    __run_start_epoch: int
    __run_end_epoch: int
    __sp_gen_time_s: float
    __sp_eval_time_s: float
    __sp_games_generated: int
    __sp_games_included: int

    def __init__(
        self, config: ChessTrainingConfig, game_bank: ChessGameBank, resume: bool = False
    ) -> None:
        self.__config = config
        self.__device = torch.device(config.device)
        self.__start_epoch = 0
        self.__last_loss = 0.0
        self.__disable_selfplay = False
        self.__selfplay_min_override: float | None = None
        self.__selfplay_max_override: float | None = None
        self.__selfplay_stockfish_depth: int = config.stockfish_depth
        self.__selfplay_value_weight: float = config.self_play_value_weight
        self.__run_start_epoch = 0
        self.__run_end_epoch = config.n_epochs
        self.__tokenizer = ChessTokenizer()
        self.__dataset = ChessDataset(self.__tokenizer)
        self.__gamebank = game_bank
        self.__evaluator = (
            StockfishEvaluator(config.stockfish_path, config.stockfish_workers)
            if config.use_stockfish
            else None
        )

        self.__model = ChessTransformer(
            vocab_size=self.__tokenizer.vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
        ).to(self.__device)

        self.__optimizer = AdamW(self.__model.parameters(), lr=config.lr, weight_decay=0.1)
        self.__scheduler = get_scheduler(self.__optimizer, config.n_epochs, config.warmup_epochs)

        if resume:
            self.__load_latest_checkpoint()
        else:
            if self.checkpoints_path.exists():
                shutil.rmtree(self.checkpoints_path)
            os.makedirs(self.checkpoints_path, exist_ok=True)

        log_path = Path(config.log_dir) / self.__config.model_name
        os.makedirs(log_path, exist_ok=True)
        self.writer = SummaryWriter(log_path)

    def save_model(self, path: Path | None = None, safe: bool = True) -> None:
        if path is None:
            path = Path(".") / "env" / "out" / self.__config.model_name
        if path.exists():
            shutil.rmtree(path)
        if safe:
            path.mkdir(parents=True, exist_ok=True)

            # save weights as safetensors
            st.save_model(self.__model, f"{path}/model.safetensors")

            # save config
            with open(f"{path}/config.json", "w") as f:
                json.dump(vars(self.__config), f, indent=2)

            print(f"model saved as safetensors to {path}/")
        else:
            torch.save(
                {
                    "model_state": self.__model.state_dict(),
                    "config": self.__config,
                    "vocab_size": self.__tokenizer.vocab_size,
                },
                path,
            )
            print(f"model saved as pt: {path}")

    @property
    def checkpoints_path(self) -> Path:
        return Path(self.__config.checkpoint_dir) / self.__config.model_name

    #
    # Generation phase
    #

    def _self_play_ratio(self, epoch: int) -> float:
        """
        Returns the fraction of games that should come from self-play this epoch.
        If disable_selfplay is set, always returns 0.0.
        If selfplay_min/max overrides are set, linearly interpolates between them
        over the current run's epoch range (run_start_epoch → run_end_epoch).
        Otherwise falls back to the config schedule.
        """
        if self.__disable_selfplay:
            return 0.0

        if self.__selfplay_min_override is not None and self.__selfplay_max_override is not None:
            run_len = max(self.__run_end_epoch - self.__run_start_epoch, 1)
            progress = (epoch - self.__run_start_epoch) / run_len
            progress = max(0.0, min(1.0, progress))
            return self.__selfplay_min_override + progress * (
                self.__selfplay_max_override - self.__selfplay_min_override
            )

        cfg = self.__config
        progress = (epoch - cfg.self_play_start_epoch) / max(cfg.self_play_ramp_epochs, 1)
        return (
            float(min(cfg.self_play_max_ratio, cfg.self_play_max_ratio * progress))
            if epoch >= cfg.self_play_start_epoch
            else 0.0
        )

    def generate_games(self, epoch: int) -> float:
        """
        Fills the dataset for this epoch.
        Returns the actual self-play ratio used (0.0 = all bank, 0.3 = 30% self-play).
        """
        self.__model.eval()
        self.__dataset.clear()
        self.__eval_time_ms = 0.0  # cumulative Stockfish wall time this epoch
        self.__eval_game_count = 0  # number of games evaluated
        self.__sp_gen_time_s = 0.0
        self.__sp_eval_time_s = 0.0
        self.__sp_games_generated = 0
        self.__sp_games_included = 0

        ratio = self._self_play_ratio(epoch)
        n_self_play = int(self.__config.n_games * ratio)
        n_bank = self.__config.n_games - n_self_play

        # bank games
        while len(self.__dataset) < n_bank:
            self.__generate_from_bank(n_bank)

        # self-play games
        if n_self_play > 0:
            self.__generate_self_play(n_self_play)

        return ratio

    def __generate_from_bank(self, target: int) -> None:
        while len(self.__gamebank) > 0 and len(self.__dataset) < target:
            moves, outcome, weights, value_evals = self.__gamebank.get_random_game()
            if not moves:
                continue

            # validate all moves are in tokenizer vocabulary
            try:
                if len(moves) > self.__config.max_moves:
                    print(
                        f"  skipping game: too many moves {len(moves)} > {self.__config.max_moves}"
                    )
                    continue

                tensor = self.__tokenizer.encode_game_tensor(moves)
                if tensor.max().item() >= self.__tokenizer.vocab_size:
                    print(
                        f"  skipping game: token ID {tensor.max().item()} >= vocab_size {self.__tokenizer.vocab_size}"
                    )
                    continue
                if tensor.min().item() < 0:
                    print(f"  skipping game: negative token ID {tensor.min().item()}")
                    continue
            except KeyError as e:
                print(f"  skipping game: unknown move {e}")
                continue

            # use pre-computed weights from the bank if available;
            # fall back to on-the-fly evaluation only if the bank wasn't pre-evaluated
            if weights is None and self.__evaluator is not None:
                t0 = time.perf_counter()
                weights = compute_move_weights(
                    moves,
                    self.__evaluator,
                    self.__config.stockfish_depth,
                    self.__config.advantage_temperature,
                )
                self.__eval_time_ms += (time.perf_counter() - t0) * 1000
                self.__eval_game_count += 1

            self.__dataset.add_game(
                GameRecord(
                    moves=moves,
                    outcome=outcome,
                    tensor=tensor,
                    move_weights=weights,
                    value_evals=value_evals,
                )
            )

    def __generation_round(self, n: int) -> list[dict[str, Any]]:
        """Dispatch to KV-cache or full-recompute generation based on config."""
        if self.__config.self_play_kv_cache:
            return self.__generation_round_kv_cache(n)
        return self.__generation_round_fw_no_cache(n)

    def __generation_round_kv_cache(self, n: int) -> list[dict[str, Any]]:
        """Generate n self-play games using a single joint KV cache.

        All active games share one [n_active, ...] cache tensor, so each step
        costs one batched generate_step call instead of n_active serial calls.

        When value_weight > 0, torch.repeat_interleave expands the joint cache
        into a flat [total_legal_moves, ...] batch so all legal-move value
        scores across all active games are computed in a single generate_step.

        When games complete, their rows are removed from the joint cache via
        index selection before advancing to the next step.
        """
        boards = [chess.Board() for _ in range(n)]
        move_history = [[self.__tokenizer.BOS] for _ in range(n)]
        active = list(range(n))
        max_moves = self.__config.self_play_max_moves
        value_weight = self.__selfplay_value_weight
        results: list[dict[str, Any]] = []

        # ------------------------------------------------------------ #
        # Prefill: process BOS for all n games in one batched call      #
        # ------------------------------------------------------------ #
        bos = torch.full((n, 1), self.__tokenizer.BOS, dtype=torch.long, device=self.__device)
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.__config.bf16):
                logits, _, joint_cache = self.__model.generate_step(bos, KVCache())
        policy_logits = logits[:, 0, :]  # [n, vocab_size]

        while active:
            n_active = len(active)

            # -------------------------------------------------------- #
            # Enumerate legal moves and slice policy scores per game    #
            # -------------------------------------------------------- #
            legal_moves_per_game: list[list[chess.Move]] = []
            legal_ids_per_game: list[torch.Tensor] = []
            legal_policy_per_game: list[torch.Tensor] = []

            for local_idx, game_idx in enumerate(active):
                legal = list(boards[game_idx].legal_moves)
                legal_moves_per_game.append(legal)
                ids = torch.tensor(
                    [self.__tokenizer.encode_move(m) for m in legal],
                    dtype=torch.long,
                    device=self.__device,
                )
                legal_ids_per_game.append(ids)
                legal_policy_per_game.append(policy_logits[local_idx][ids])

            # -------------------------------------------------------- #
            # Value scoring: one big generate_step (when λ > 0)        #
            # repeat_interleave expands each game's cache row           #
            # n_legal_i times, matching the flat legal-move token list  #
            # -------------------------------------------------------- #
            if value_weight > 0.0:
                counts = [len(ids) for ids in legal_ids_per_game]
                repeats = torch.tensor(counts, dtype=torch.long, device=self.__device)

                flat_cache = KVCache(
                    layers=[
                        LayerKVCache(
                            k=torch.repeat_interleave(layer.k, repeats, dim=0),
                            v=torch.repeat_interleave(layer.v, repeats, dim=0),
                        )
                        for layer in joint_cache.layers
                    ]
                )
                value_tokens = torch.cat(legal_ids_per_game).unsqueeze(1)  # [total_legal, 1]

                with torch.no_grad():
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.__config.bf16):
                        _, value_preds, _ = self.__model.generate_step(value_tokens, flat_cache)
                value_flat = value_preds[:, 0]  # [total_legal]

                offset = 0
                final_scores_per_game: list[torch.Tensor] = []
                for local_idx, game_idx in enumerate(active):
                    n_legal = counts[local_idx]
                    v = value_flat[offset : offset + n_legal]
                    if boards[game_idx].turn == chess.BLACK:
                        v = -v
                    final_scores_per_game.append(
                        legal_policy_per_game[local_idx] + value_weight * v
                    )
                    offset += n_legal
            else:
                final_scores_per_game = legal_policy_per_game

            # -------------------------------------------------------- #
            # Sample, advance boards, collect completed games           #
            # -------------------------------------------------------- #
            sampled_tokens: list[int] = []
            still_active: list[int] = []
            keep_local: list[int] = []  # local indices to retain in joint cache

            for local_idx, game_idx in enumerate(active):
                board = boards[game_idx]
                legal = legal_moves_per_game[local_idx]
                probs = F.softmax(final_scores_per_game[local_idx], dim=0)

                if torch.isnan(probs).any() or torch.isinf(probs).any():
                    move_idx = int(torch.randint(len(legal), (1,)).item())
                else:
                    move_idx = int(torch.multinomial(probs, num_samples=1).item())

                move = legal[move_idx]
                board.push(move)
                token = self.__tokenizer.encode_move(move)
                move_history[game_idx].append(token)

                if board.is_game_over() or len(move_history[game_idx]) >= max_moves:
                    game = self.__extract_game(boards[game_idx], move_history[game_idx])
                    if game is not None:
                        results.append(game)
                else:
                    still_active.append(game_idx)
                    sampled_tokens.append(token)
                    keep_local.append(local_idx)

            active = still_active
            if not active:
                break

            # -------------------------------------------------------- #
            # Drop completed games from the joint cache                 #
            # -------------------------------------------------------- #
            if len(keep_local) < n_active:
                joint_cache = KVCache(
                    layers=[
                        LayerKVCache(k=layer.k[keep_local], v=layer.v[keep_local])
                        for layer in joint_cache.layers
                    ]
                )

            # -------------------------------------------------------- #
            # Advance joint cache with sampled moves → next policy      #
            # -------------------------------------------------------- #
            token_tensor = torch.tensor(
                sampled_tokens, dtype=torch.long, device=self.__device
            ).unsqueeze(
                1
            )  # [n_still_active, 1]

            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.__config.bf16):
                    logits, _, joint_cache = self.__model.generate_step(token_tensor, joint_cache)
            policy_logits = logits[:, 0, :]  # [n_still_active, vocab_size]

        return results

    def __generation_round_fw_no_cache(self, n: int) -> list[dict[str, Any]]:
        """Generate n self-play games using batched full forward recompute.

        At each step all active games have identical sequence lengths, so they
        are stacked into a single [n_active, seq_len] tensor for one batched
        forward pass instead of n_active separate passes.

        When value_weight > 0, the value sequences (history + each legal move)
        are also all the same length, so all legal moves across all games are
        concatenated into one further batched forward pass.
        """
        boards = [chess.Board() for _ in range(n)]
        move_history = [[self.__tokenizer.BOS] for _ in range(n)]
        active = list(range(n))
        max_moves = self.__config.self_play_max_moves
        value_weight = self.__selfplay_value_weight
        results: list[dict[str, Any]] = []

        while active:
            # -------------------------------------------------------- #
            # Policy step: one forward pass for all active games        #
            # -------------------------------------------------------- #
            input_ids = torch.tensor(
                [move_history[i] for i in active],
                dtype=torch.long,
                device=self.__device,
            )  # [n_active, seq_len]

            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.__config.bf16):
                    logits, _ = self.__model(input_ids, is_causal=True)

            policy_logits = logits[:, -1, :].clone()  # [n_active, vocab_size]
            del logits, _

            # -------------------------------------------------------- #
            # Per-game: enumerate legal moves, slice policy scores      #
            # -------------------------------------------------------- #
            legal_moves_per_game: list[list[chess.Move]] = []
            legal_ids_per_game: list[torch.Tensor] = []
            legal_policy_per_game: list[torch.Tensor] = []

            for idx, game_idx in enumerate(active):
                legal = list(boards[game_idx].legal_moves)
                legal_moves_per_game.append(legal)
                ids = torch.tensor(
                    [self.__tokenizer.encode_move(m) for m in legal],
                    dtype=torch.long,
                    device=self.__device,
                )
                legal_ids_per_game.append(ids)
                legal_policy_per_game.append(policy_logits[idx][ids])

            # -------------------------------------------------------- #
            # Value step: one big batched forward pass (when λ > 0)    #
            # All value sequences are length seq_len+1 — same length   #
            # for every (game, legal_move) pair.                        #
            # -------------------------------------------------------- #
            if value_weight > 0.0:
                value_seqs: list[list[int]] = []
                counts: list[int] = []
                for idx, game_idx in enumerate(active):
                    hist = move_history[game_idx]
                    for tid in legal_ids_per_game[idx]:
                        value_seqs.append(hist + [int(tid.item())])
                    counts.append(len(legal_ids_per_game[idx]))

                value_input = torch.tensor(
                    value_seqs, dtype=torch.long, device=self.__device
                )  # [total_legal, seq_len+1]

                with torch.no_grad():
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.__config.bf16):
                        value_logits, value_preds = self.__model(value_input, is_causal=True)
                del value_logits, value_input

                value_flat = value_preds[:, -1].clone()  # [total_legal]
                del value_preds

                offset = 0
                final_scores_per_game: list[torch.Tensor] = []
                for idx, game_idx in enumerate(active):
                    n_legal = counts[idx]
                    v = value_flat[offset : offset + n_legal]
                    if boards[game_idx].turn == chess.BLACK:
                        v = -v
                    final_scores_per_game.append(legal_policy_per_game[idx] + value_weight * v)
                    offset += n_legal
            else:
                final_scores_per_game = legal_policy_per_game

            # -------------------------------------------------------- #
            # Sample and advance each game                              #
            # -------------------------------------------------------- #
            still_active = []
            for idx, game_idx in enumerate(active):
                board = boards[game_idx]
                legal = legal_moves_per_game[idx]
                probs = F.softmax(final_scores_per_game[idx], dim=0)

                if torch.isnan(probs).any() or torch.isinf(probs).any():
                    move_idx = int(torch.randint(len(legal), (1,)).item())
                else:
                    move_idx = int(torch.multinomial(probs, num_samples=1).item())

                move = legal[move_idx]
                board.push(move)
                move_history[game_idx].append(self.__tokenizer.encode_move(move))

                if board.is_game_over() or len(move_history[game_idx]) >= max_moves:
                    game = self.__extract_game(boards[game_idx], move_history[game_idx])
                    if game is not None:
                        results.append(game)
                else:
                    still_active.append(game_idx)

            active = still_active

        return results

    def __extract_game(self, board: chess.Board, token_history: list[int]) -> dict[str, Any] | None:
        """Extract a completed self-play game into a serialisable dict, or None if invalid."""
        result = board.result()  # "1-0", "0-1", "1/2-1/2", or "*" if max_moves reached
        outcome = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}.get(result, 0.5)

        moves: list[chess.Move] = [
            m
            for t in token_history
            if t not in (self.__tokenizer.BOS, self.__tokenizer.EOS, self.__tokenizer.PAD)
            for m in (self.__tokenizer.decode_move(t),)
            if m is not None
        ]

        if len(moves) > self.__config.max_moves:
            return None

        return {"moves": [m.uci() for m in moves], "outcome": outcome, "weights": None}

    def __add_batch_to_dataset(self, batch_path: Path) -> None:
        """Load an evaluated batch from a temp file, add games to the dataset, delete the file."""
        data: list[dict[str, Any]] = torch.load(batch_path, weights_only=False)
        batch_path.unlink()

        for item in data:
            moves_uci: list[str] = item.get("moves", [])
            try:
                moves = [chess.Move.from_uci(uci) for uci in moves_uci]
                tensor = self.__tokenizer.encode_game_tensor(moves)
            except Exception:
                continue

            self.__dataset.add_game(
                GameRecord(
                    moves=moves,
                    outcome=float(item["outcome"]),
                    tensor=tensor,
                    move_weights=item.get("weights"),
                )
            )
            self.__sp_games_included += 1

    def __generate_self_play(self, n_self_play: int) -> None:
        """
        Generate n_self_play games using async Stockfish evaluation.

        Generation and evaluation overlap: while one batch is being evaluated
        in a forked child process, the next batch is being generated on the GPU.
        The number of concurrent forks is capped at stockfish_workers.
        """
        batch_size = self.__config.self_play_batch_size
        evaluator = (
            AsyncBatchEvaluator(
                stockfish_path=self.__config.stockfish_path,
                depth=self.__selfplay_stockfish_depth,
                temperature=self.__config.advantage_temperature,
                n_workers=self.__config.stockfish_workers,
            )
            if self.__config.use_stockfish
            else None
        )

        generated = 0
        try:
            while generated < n_self_play:
                this_batch = min(batch_size, n_self_play - generated)
                t_gen = time.perf_counter()
                batch = self.__generation_round(this_batch)
                self.__sp_gen_time_s += time.perf_counter() - t_gen

                if not batch:
                    continue
                generated += len(batch)
                self.__sp_games_generated += len(batch)

                if evaluator is not None:
                    # save to temp file and submit for async evaluation;
                    # submit() may block here if the worker pool is full,
                    # returning any batch that just finished as it yields a slot
                    t_eval = time.perf_counter()
                    completed = evaluator.submit(_save_temp_batch(batch))
                    self.__sp_eval_time_s += time.perf_counter() - t_eval
                    for path in completed:
                        self.__add_batch_to_dataset(path)
                else:
                    # no Stockfish — add directly with no weights
                    for item in batch:
                        try:
                            moves = [chess.Move.from_uci(uci) for uci in item["moves"]]
                            tensor = self.__tokenizer.encode_game_tensor(moves)
                            self.__dataset.add_game(
                                GameRecord(
                                    moves=moves,
                                    outcome=float(item["outcome"]),
                                    tensor=tensor,
                                    move_weights=None,
                                )
                            )
                            self.__sp_games_included += 1
                        except Exception:
                            continue

            # drain any batches still being evaluated
            if evaluator is not None:
                t_eval = time.perf_counter()
                for path in evaluator.drain():
                    self.__add_batch_to_dataset(path)
                self.__sp_eval_time_s += time.perf_counter() - t_eval

            # keep memory under control
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        finally:
            if evaluator is not None:
                evaluator.close()

    def __run_epoch_streaming(
        self, epoch: int, n_bank: int, n_self_play: int, ratio: float
    ) -> tuple[float, float, bool]:
        """
        Streaming training: generate games in self_play_batch_size chunks, accumulate
        gradients across all chunks, then take a single optimizer.step() per epoch.
        Never holds more than self_play_batch_size games in memory at one time.

        Self-play games are evaluated synchronously per chunk (no async pipeline).
        """
        chunk_size = self.__config.streaming_chunk_size
        train_bs = self.__config.train_batch_size

        # Init per-epoch metrics
        self.__eval_time_ms = 0.0
        self.__eval_game_count = 0
        self.__sp_gen_time_s = 0.0
        self.__sp_eval_time_s = 0.0
        self.__sp_games_generated = 0
        self.__sp_games_included = 0

        # Build list of (n_bank_this, n_sp_this) chunks
        bank_left = n_bank
        sp_left = n_self_play
        chunks: list[tuple[int, int]] = []
        while bank_left > 0 or sp_left > 0:
            b = min(bank_left, chunk_size)
            sp = min(sp_left, chunk_size - b)
            chunks.append((b, sp))
            bank_left -= b
            sp_left -= sp

        if not chunks:
            return 0.0, 0.0, False

        outcome_scale = (
            min(1.0, epoch / max(self.__config.outcome_warmup, 1)) if ratio > 0.0 else 0.0
        )

        total_loss = 0.0
        total_value_loss = 0.0
        total_mini_batches = 0

        self.__optimizer.zero_grad()

        for b_this, sp_this in chunks: #tqdm(chunks, desc="chunks", leave=False):
            self.__dataset.clear()

            # ---------------------------------------------------------- #
            # Generate games for this chunk (eval mode, no grad)         #
            # ---------------------------------------------------------- #
            self.__model.eval()

            if b_this > 0:
                self.__generate_from_bank(b_this)

            if sp_this > 0:
                t_gen = time.perf_counter()
                print(f'self-playing {sp_this} games')
                sp_batch = self.__generation_round(sp_this)
                self.__sp_gen_time_s += time.perf_counter() - t_gen
                self.__sp_games_generated += len(sp_batch)

                t_eval = time.perf_counter()
                for item in sp_batch:
                    try:
                        moves = [chess.Move.from_uci(uci) for uci in item["moves"]]
                        tensor = self.__tokenizer.encode_game_tensor(moves)
                        weights: torch.Tensor | None = None
                        if self.__evaluator is not None:
                            weights = compute_move_weights(
                                moves,
                                self.__evaluator,
                                self.__selfplay_stockfish_depth,
                                self.__config.advantage_temperature,
                            )
                        self.__dataset.add_game(
                            GameRecord(
                                moves=moves,
                                outcome=float(item["outcome"]),
                                tensor=tensor,
                                move_weights=weights,
                            )
                        )
                        self.__sp_games_included += 1
                    except Exception:
                        continue
                self.__sp_eval_time_s += time.perf_counter() - t_eval

                # keep memory under control
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            if len(self.__dataset) == 0:
                continue

            # ---------------------------------------------------------- #
            # Train on this chunk (train mode, accumulate gradients)     #
            # ---------------------------------------------------------- #
            self.__model.train()

            dataloader = DataLoader(
                self.__dataset,
                batch_size=min(train_bs, len(self.__dataset)),
                shuffle=True,
                collate_fn=self.__dataset.collate,
            )

            if outcome_scale > 0.0:
                chunk_outcomes = torch.tensor(
                    [g.outcome for g in self.__dataset.games], device=self.__device
                )
                outcome_weights = (chunk_outcomes * 2 - 1).mean()
            else:
                outcome_weights = torch.tensor(0.0, device=self.__device)

            for batch in dataloader:
                input_ids = cast(torch.Tensor, batch["input_ids"]).to(self.__device)
                targets = cast(torch.Tensor, batch["targets"]).to(self.__device)
                attention_mask = cast(torch.Tensor, batch["attention_mask"]).to(self.__device)
                move_weights = cast(torch.Tensor, batch["move_weights"]).to(self.__device)
                value_targets = cast(torch.Tensor, batch["value_evals"]).to(self.__device)
                has_evals = cast(torch.Tensor, batch["has_value_evals"]).to(self.__device)

                causal_mask = make_causal_mask(input_ids.size(1), self.__device)
                padding_mask = make_padding_mask(attention_mask).to(self.__device)
                combined_mask = causal_mask + padding_mask

                assert (
                    input_ids.max().item() < self.__tokenizer.vocab_size
                ), f"token ID {input_ids.max().item()} >= vocab_size {self.__tokenizer.vocab_size}"
                assert input_ids.min().item() >= 0, f"negative token ID {input_ids.min().item()}"
                assert (
                    targets.max().item() < self.__tokenizer.vocab_size
                ), f"target token ID {targets.max().item()} >= vocab_size {self.__tokenizer.vocab_size}"

                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.__config.bf16):
                    logits, value_pred = self.__model(input_ids, combined_mask)

                    per_token_loss = F.cross_entropy(
                        logits.view(-1, self.__tokenizer.vocab_size),
                        targets.view(-1),
                        ignore_index=self.__tokenizer.PAD,
                        label_smoothing=0.1,
                        reduction="none",
                    )

                    pad_mask = (targets.view(-1) != self.__tokenizer.PAD).float()
                    weights_flat = move_weights.view(-1)
                    denom = pad_mask.sum().clamp(min=1)
                    policy_loss = (per_token_loss * weights_flat * pad_mask).sum() / denom

                    token_mask = attention_mask.bool() & has_evals.unsqueeze(1)
                    if token_mask.any():
                        value_loss_t = F.mse_loss(value_pred[token_mask], value_targets[token_mask])
                        loss = policy_loss + self.__config.value_loss_weight * value_loss_t
                    else:
                        value_loss_t = torch.tensor(0.0, device=self.__device)
                        loss = policy_loss

                if torch.isnan(loss) or torch.isinf(loss):
                    print("warning: NaN/inf loss detected, skipping mini-batch")
                    continue

                scaled_loss = loss * (1.0 + outcome_scale * outcome_weights)
                # Accumulate gradients without per-batch scaling;
                # we normalize by total_mini_batches after all chunks complete
                scaled_loss.backward()  # type: ignore[no-untyped-call]

                total_loss += scaled_loss.item()
                total_value_loss += value_loss_t.item()
                total_mini_batches += 1

        if total_mini_batches == 0:
            print("warning: all mini-batches had NaN/inf loss, skipping update")
            self.__optimizer.zero_grad()
            return 0.0, 0.0, False

        # Normalize: divide accumulated gradients so the effective update equals
        # the mean loss over all mini-batches (same semantics as train_on_games)
        for p in self.__model.parameters():
            if p.grad is not None:
                p.grad /= total_mini_batches

        for name, p in self.__model.named_parameters():
            if torch.isnan(p).any() or torch.isinf(p).any():
                print(f"  bad weight: {name} nan={torch.isnan(p).any()} inf={torch.isinf(p).any()}")

        if any(torch.isnan(p.grad).any() for p in self.__model.parameters() if p.grad is not None):
            print("warning: NaN gradients detected, skipping update")
            self.__optimizer.zero_grad()
            return total_loss / total_mini_batches, total_value_loss / total_mini_batches, False

        torch.nn.utils.clip_grad_norm_(self.__model.parameters(), max_norm=1.0)
        self.__optimizer.step()
        self.__optimizer.zero_grad()

        self.__last_loss = total_loss / total_mini_batches
        return self.__last_loss, total_value_loss / total_mini_batches, True

    #
    # Training phase
    #

    def train_on_games(self, epoch: int, self_play_ratio: float) -> tuple[float, float, bool]:
        self.__model.train()

        n_games = len(self.__dataset)
        batch_size = min(self.__config.train_batch_size, n_games)
        n_batches = math.ceil(n_games / batch_size)

        dataloader = DataLoader(
            self.__dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.__dataset.collate,
        )

        # outcome weighting only applies when self-play games are present — those games
        # were generated by the model itself, so we can reinforce winning patterns.
        # bank games are worth learning from equally regardless of outcome.
        if self_play_ratio > 0.0:
            outcome_scale = min(1.0, epoch / max(self.__config.outcome_warmup, 1))
            outcomes = torch.tensor(
                [g.outcome for g in self.__dataset.games],
                device=self.__device,
            )
            outcome_weights = (outcomes * 2 - 1).mean()  # [-1, +1]
        else:
            outcome_scale = 0.0
            outcome_weights = torch.tensor(0.0, device=self.__device)

        total_loss = 0.0
        total_value_loss = 0.0
        nan_batches = 0
        self.__optimizer.zero_grad()

        for batch in dataloader:
            input_ids = cast(torch.Tensor, batch["input_ids"]).to(self.__device)
            targets = cast(torch.Tensor, batch["targets"]).to(self.__device)
            attention_mask = cast(torch.Tensor, batch["attention_mask"]).to(self.__device)

            causal_mask = make_causal_mask(input_ids.size(1), self.__device)
            padding_mask = make_padding_mask(attention_mask).to(self.__device)
            combined_mask = causal_mask + padding_mask

            assert (
                input_ids.max().item() < self.__tokenizer.vocab_size
            ), f"token ID {input_ids.max().item()} >= vocab_size {self.__tokenizer.vocab_size}"
            assert input_ids.min().item() >= 0, f"negative token ID {input_ids.min().item()}"
            assert (
                targets.max().item() < self.__tokenizer.vocab_size
            ), f"target token ID {targets.max().item()} >= vocab_size {self.__tokenizer.vocab_size}"

            move_weights = cast(torch.Tensor, batch["move_weights"]).to(self.__device)
            value_targets = cast(torch.Tensor, batch["value_evals"]).to(self.__device)
            has_evals = cast(torch.Tensor, batch["has_value_evals"]).to(self.__device)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.__config.bf16):
                logits, value_pred = self.__model(input_ids, combined_mask)

                per_token_loss = F.cross_entropy(
                    logits.view(-1, self.__tokenizer.vocab_size),
                    targets.view(-1),
                    ignore_index=self.__tokenizer.PAD,
                    label_smoothing=0.1,
                    reduction="none",
                )  # [batch * seq_len]

                pad_mask = (targets.view(-1) != self.__tokenizer.PAD).float()
                weights_flat = move_weights.view(-1)
                denom = pad_mask.sum().clamp(min=1)
                policy_loss = (per_token_loss * weights_flat * pad_mask).sum() / denom

                # value loss — only on positions from games that have Stockfish evals
                token_mask = attention_mask.bool() & has_evals.unsqueeze(1)
                if token_mask.any():
                    value_loss = F.mse_loss(value_pred[token_mask], value_targets[token_mask])
                    loss = policy_loss + self.__config.value_loss_weight * value_loss
                else:
                    value_loss = torch.tensor(0.0, device=self.__device)
                    loss = policy_loss

            if torch.isnan(loss) or torch.isinf(loss):
                print("warning: NaN/inf loss detected, skipping mini-batch")
                nan_batches += 1
                continue

            scaled_loss = loss * (1.0 + outcome_scale * outcome_weights)

            # scale gradient by 1/n_batches so the effective update equals
            # the average loss across all mini-batches
            (scaled_loss / n_batches).backward()  # type: ignore[no-untyped-call]

            total_loss += scaled_loss.item()
            total_value_loss += value_loss.item()

        if nan_batches == n_batches:
            print("warning: all mini-batches had NaN/inf loss, skipping update")
            self.__optimizer.zero_grad()
            return total_loss, total_value_loss, False

        # check weights
        for name, p in self.__model.named_parameters():
            if torch.isnan(p).any() or torch.isinf(p).any():
                print(f"  bad weight: {name} nan={torch.isnan(p).any()} inf={torch.isinf(p).any()}")

        # check gradients for NaN before stepping
        if any(torch.isnan(p.grad).any() for p in self.__model.parameters() if p.grad is not None):
            print("warning: NaN gradients detected, skipping update")
            self.__optimizer.zero_grad()
            return total_loss, total_value_loss, False

        torch.nn.utils.clip_grad_norm_(self.__model.parameters(), max_norm=1.0)
        self.__optimizer.step()
        self.__optimizer.zero_grad()

        self.__last_loss = total_loss / n_batches
        return self.__last_loss, total_value_loss / n_batches, True

    #
    # Value head pretraining
    #

    def pretrain_value_head(self, n_epochs: int, lr: float = 1e-3) -> None:
        """
        Train the value head to predict Stockfish evaluations on bank games,
        before self-play begins.  The transformer body and policy head are frozen;
        only the value head parameters are updated.

        Requires the game bank to have been Stockfish-evaluated (evaluate_moves).
        Games without value_evals are silently skipped.
        """
        # freeze everything except the value head
        value_params = [p for name, p in self.__model.named_parameters() if "value_head" in name]
        for p in self.__model.parameters():
            p.requires_grad = False
        for p in value_params:
            p.requires_grad = True

        optimizer = AdamW(value_params, lr=lr)

        print(
            f"pretraining value head — {sum(p.numel() for p in value_params):,} params, {n_epochs} epochs"
        )

        self.__model.train()

        for epoch in tqdm(range(n_epochs), desc="value pretrain"):
            self.__dataset.clear()
            self.__generate_from_bank(self.__config.n_games)

            # skip epoch if no games have value evals
            games_with_evals = [g for g in self.__dataset.games if g.value_evals is not None]
            if not games_with_evals:
                print(
                    f"  epoch {epoch}: no value_evals in bank — skipping (run evaluate_moves first)"
                )
                break

            dataloader = DataLoader(
                self.__dataset,
                batch_size=min(self.__config.train_batch_size, len(self.__dataset)),
                shuffle=True,
                collate_fn=self.__dataset.collate,
            )

            total_loss = 0.0
            n_batches = 0

            for batch in dataloader:
                input_ids = cast(torch.Tensor, batch["input_ids"]).to(self.__device)
                attention_mask = cast(torch.Tensor, batch["attention_mask"]).to(self.__device)
                value_targets = cast(torch.Tensor, batch["value_evals"]).to(self.__device)
                has_evals = cast(torch.Tensor, batch["has_value_evals"]).to(self.__device)

                # skip batch if no game has real evals
                if not has_evals.any():
                    continue

                causal_mask = make_causal_mask(input_ids.size(1), self.__device)
                padding_mask = make_padding_mask(attention_mask).to(self.__device)

                # mask: real tokens in games that actually have Stockfish evals
                token_mask = attention_mask.bool() & has_evals.unsqueeze(1)

                if not token_mask.any():
                    continue

                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.__config.bf16):
                    _, value_pred = self.__model(input_ids, causal_mask + padding_mask)
                    loss = F.mse_loss(value_pred[token_mask], value_targets[token_mask])

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                loss.backward()  # type: ignore[no-untyped-call]
                torch.nn.utils.clip_grad_norm_(value_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            self.writer.add_scalar("value_pretrain/loss", avg_loss, epoch)

        # restore all parameters to trainable for subsequent policy training
        for p in self.__model.parameters():
            p.requires_grad = True

        self.__save_checkpoint(self.__start_epoch - 1, self.__last_loss)
        print("value head pretraining complete")

    #
    # Main loop
    #

    def run(
        self,
        max_epochs: int | None = None,
        disable_selfplay: bool = False,
        self_play_min: float | None = None,
        self_play_max: float | None = None,
        self_play_stockfish_depth: int | None = 5,
        self_play_value_weight: float | None = None,
    ) -> None:
        print(f"training on {self.__config.device}")
        print(f"model parameters: {sum(p.numel() for p in self.__model.parameters()):,}")

        end_epoch = self.__start_epoch + max_epochs if max_epochs else self.__config.n_epochs
        if end_epoch > self.__config.n_epochs:
            self.__config.n_epochs = end_epoch
            self.__scheduler = get_scheduler(
                self.__optimizer, end_epoch, self.__config.warmup_epochs
            )

        self.__disable_selfplay = disable_selfplay
        self.__selfplay_min_override = self_play_min
        self.__selfplay_max_override = self_play_max
        self.__selfplay_stockfish_depth = (
            self_play_stockfish_depth
            if self_play_stockfish_depth is not None
            else self.__config.stockfish_depth
        )
        self.__selfplay_value_weight = (
            self_play_value_weight
            if self_play_value_weight is not None
            else self.__config.self_play_value_weight
        )
        self.__run_start_epoch = self.__start_epoch
        self.__run_end_epoch = end_epoch

        loss = 0.0
        value_loss = 0.0
        self_play_ratio = 0.0
        breakthrough_detector = LossBreakthroughDetector()
        for epoch in tqdm(range(self.__start_epoch, end_epoch)):
            epoch_start = time.perf_counter()

            cycle = 10
            attempt = 1
            while True:
                if self.__config.streaming:
                    ratio = self._self_play_ratio(epoch)
                    n_sp = int(self.__config.n_games * ratio)
                    n_bank = self.__config.n_games - n_sp
                    loss, value_loss, stepped = self.__run_epoch_streaming(
                        epoch, n_bank, n_sp, ratio
                    )
                else:
                    ratio = self.generate_games(epoch)
                    loss, value_loss, stepped = self.train_on_games(epoch, ratio)
                if stepped:
                    self_play_ratio = ratio
                    attempt = 1
                    break
                print(f"failed to step {attempt} / {cycle} , retrying")
                if attempt % cycle == 0:
                    print("bullshit warning, reinitializing weights")
                    self.__model.init_weights()
                    attempt = 1
                else:
                    attempt += 1

            self.__scheduler.step()

            epoch_s = time.perf_counter() - epoch_start
            eval_ms = self.__eval_time_ms / max(self.__eval_game_count, 1)

            if epoch % self.__config.log_every == 0:
                self.__log(epoch, loss, value_loss, self_play_ratio, epoch_s, eval_ms)

            is_scheduled = epoch > 0 and epoch % self.__config.checkpoint_every == 0
            is_breakthrough = breakthrough_detector.update(loss)
            if is_scheduled or is_breakthrough:
                self.__save_checkpoint(epoch, loss)

            # keep memory under control
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        self.__start_epoch = end_epoch
        self.__save_checkpoint(end_epoch - 1, loss)
        self.writer.close()
        if self.__evaluator is not None:
            self.__evaluator.close()

    #
    # Logging
    #

    def __log(
        self,
        epoch: int,
        loss: float,
        value_loss: float,
        self_play_ratio: float,
        epoch_s: float,
        eval_ms: float,
    ) -> None:
        games = self.__dataset.games
        n_games = len(games)
        outcomes = [g.outcome for g in games]
        lengths = [len(g.moves) for g in games]

        avg_len = sum(lengths) / max(n_games, 1)
        win_rate = sum(1 for o in outcomes if o == 1.0) / max(n_games, 1)
        draw_rate = sum(1 for o in outcomes if o == 0.5) / max(n_games, 1)
        loss_rate = sum(1 for o in outcomes if o == 0.0) / max(n_games, 1)
        current_lr = self.__optimizer.param_groups[0]["lr"]

        # tensorboard
        self.writer.add_scalar("loss/train", loss, epoch)
        self.writer.add_scalar("loss/value", value_loss, epoch)
        self.writer.add_scalar("games/avg_length", avg_len, epoch)
        self.writer.add_scalar("games/win_rate", win_rate, epoch)
        self.writer.add_scalar("games/draw_rate", draw_rate, epoch)
        self.writer.add_scalar("games/loss_rate", loss_rate, epoch)
        self.writer.add_scalar("curriculum/self_play", self_play_ratio, epoch)
        self.writer.add_scalar("lr", current_lr, epoch)
        self.writer.add_scalar("perf/epoch_s", epoch_s, epoch)
        self.writer.add_scalar("perf/eval_ms_per_game", eval_ms, epoch)
        self.writer.add_scalar("self_play/gen_time_s", self.__sp_gen_time_s, epoch)
        self.writer.add_scalar("self_play/eval_time_s", self.__sp_eval_time_s, epoch)
        self.writer.add_scalar("self_play/games_generated", self.__sp_games_generated, epoch)
        self.writer.add_scalar("self_play/games_included", self.__sp_games_included, epoch)

    #
    # Checkpointing
    #

    def __get_checkpoint_path(self, epoch: int) -> Path:
        return self.checkpoints_path / f"checkpoint_epoch_{epoch:04d}.pt"

    def __save_checkpoint(self, epoch: int, loss: float) -> None:
        path = self.__get_checkpoint_path(epoch)
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.__model.state_dict(),
                "optimizer_state": self.__optimizer.state_dict(),
                "scheduler_state": self.__scheduler.state_dict(),
                "loss": loss,
            },
            path,
        )
        self.__prune_checkpoints()

    def __prune_checkpoints(self) -> None:
        checkpoints = sorted(self.checkpoints_path.glob("checkpoint_epoch_*.pt"))
        for old in checkpoints[: -self.__config.keep_last_n]:
            old.unlink()

    def __load_latest_checkpoint(self) -> None:
        checkpoints = sorted(self.checkpoints_path.glob("checkpoint_epoch_*.pt"))
        if not checkpoints:
            print("no checkpoints found, starting from scratch")
            return

        path = checkpoints[-1]
        checkpoint = torch.load(path, map_location=self.__device)

        result = self.__model.load_state_dict(checkpoint["model_state"], strict=False)
        if result.missing_keys:
            print(f"  initialized missing keys from scratch: {result.missing_keys}")
        if result.unexpected_keys:
            print(f"  ignored unexpected keys: {result.unexpected_keys}")
        try:
            self.__optimizer.load_state_dict(checkpoint["optimizer_state"])
        except ValueError:
            print("  optimizer state incompatible (model changed), starting optimizer fresh")
        self.__scheduler.load_state_dict(checkpoint["scheduler_state"])
        self.__start_epoch = checkpoint["epoch"] + 1
        self.__last_loss = float(checkpoint["loss"])

        print(
            f"resumed from {path.name} (epoch {checkpoint['epoch']}, loss {checkpoint['loss']:.4f})"
        )

    def _check_tensors(self, label: str, **tensors: torch.Tensor) -> bool:
        """Returns True if any tensor contains NaN or inf."""
        found = False
        for name, t in tensors.items():
            has_nan = torch.isnan(t).any().item()
            has_inf = torch.isinf(t).any().item()
            if has_nan or has_inf:
                print(
                    f"  {label} — {name}: nan={has_nan} inf={has_inf} min={t.min():.4f} max={t.max():.4f}"
                )
                found = True
        return found
