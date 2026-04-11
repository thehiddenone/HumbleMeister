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

from humblemeister.attention import KVCache, make_causal_mask, make_padding_mask
from humblemeister.data import ChessDataset, ChessGameBank, ChessTokenizer, GameRecord
from humblemeister.evaluation import AsyncBatchEvaluator, StockfishEvaluator, compute_move_weights
from humblemeister.inference import sample_move
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
    self_play_max_ratio: float = 0.3  # cap at 30% self-play games
    self_play_batch_size: int = 16  # games per generation sub-batch (bounds KV-cache VRAM)

    # stockfish evaluation
    use_stockfish: bool = True
    stockfish_path: str = "stockfish"
    stockfish_depth: int = 5  # depth 5 ≈ 1-5ms/position
    stockfish_workers: int = 4  # parallel engine processes for board evaluation
    advantage_temperature: float = 1.0  # sharpness of per-move weight distribution
    value_loss_weight: float = 0.5  # relative weight of value loss vs policy loss
    self_play_kv_cache: bool = False  # use KV cache during self-play generation (faster but more VRAM)
    self_play_value_weight: float = 0.0  # λ for value-blended move selection during self-play

    # checkpointing
    checkpoint_dir: str = "env/checkpoints"
    checkpoint_every: int = 50  # save every N epochs
    keep_last_n: int = 5  # keep only the last N checkpoints

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

        self.writer = SummaryWriter(config.log_dir)

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
        """Generate n self-play games using KV cache. Faster per step, higher VRAM."""
        boards = [chess.Board() for _ in range(n)]
        move_history = [[self.__tokenizer.BOS] for _ in range(n)]
        kv_caches = [KVCache() for _ in range(n)]
        active = list(range(n))
        max_moves = self.__config.max_moves
        results: list[dict[str, Any]] = []

        with torch.no_grad():
            while active:
                latest_tokens = torch.tensor(
                    [[move_history[i][-1]] for i in active],
                    dtype=torch.long,
                    device=self.__device,
                )  # [n_active, 1]

                next_logits_list = []
                for idx, game_idx in enumerate(active):
                    token = latest_tokens[idx].unsqueeze(0)  # [1, 1]
                    logits, _, new_cache = self.__model.generate_step(
                        token,
                        kv_caches[game_idx],
                    )
                    next_logits_list.append(logits[0, 0])  # [vocab_size]
                    kv_caches[game_idx] = new_cache

                still_active = []
                for idx, game_idx in enumerate(active):
                    board = boards[game_idx]
                    legal_mask = self.__tokenizer.get_legal_mask(board).to(self.__device)
                    masked_logits = next_logits_list[idx] + legal_mask
                    probs = F.softmax(masked_logits, dim=-1)

                    if torch.isnan(probs).any() or torch.isinf(probs).any():
                        legal_indices = (legal_mask == 0.0).nonzero(as_tuple=True)[0]
                        token_id = legal_indices[torch.randint(len(legal_indices), (1,))].item()
                    else:
                        token_id = torch.multinomial(probs, num_samples=1).item()

                    move = self.__tokenizer.decode_move(int(token_id))
                    if move is None:
                        raise RuntimeError(f"unexpected move: {move}")

                    board.push(move)
                    move_history[game_idx].append(int(token_id))

                    if board.is_game_over() or len(move_history[game_idx]) >= max_moves:
                        game = self.__extract_game(boards[game_idx], move_history[game_idx])
                        if game is not None:
                            results.append(game)
                        kv_caches[game_idx] = None  # type: ignore
                    else:
                        still_active.append(game_idx)

                active = still_active

        return results

    def __generation_round_fw_no_cache(self, n: int) -> list[dict[str, Any]]:
        """Generate n self-play games using full forward recompute. Higher compute, lower VRAM."""
        boards = [chess.Board() for _ in range(n)]
        move_history = [[self.__tokenizer.BOS] for _ in range(n)]
        active = list(range(n))
        max_moves = self.__config.max_moves
        results: list[dict[str, Any]] = []

        while active:
            still_active = []
            for game_idx in active:
                board = boards[game_idx]
                move = sample_move(
                    model=self.__model,
                    tokenizer=self.__tokenizer,
                    board=board,
                    move_history=move_history[game_idx],
                    device=self.__device,
                    value_weight=self.__selfplay_value_weight,
                    bf16=self.__config.bf16,
                )
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
                batch = self.__generation_round(batch_size)
                if not batch:
                    continue
                generated += len(batch)

                if evaluator is not None:
                    # save to temp file and submit for async evaluation;
                    # submit() may block here if the worker pool is full,
                    # returning any batch that just finished as it yields a slot
                    completed = evaluator.submit(_save_temp_batch(batch))
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
                        except Exception:
                            continue

            # drain any batches still being evaluated
            if evaluator is not None:
                for path in evaluator.drain():
                    self.__add_batch_to_dataset(path)

        finally:
            if evaluator is not None:
                evaluator.close()

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

        end_epoch = min(self.__start_epoch + max_epochs, self.__config.n_epochs) if max_epochs else self.__config.n_epochs

        self.__disable_selfplay = disable_selfplay
        self.__selfplay_min_override = self_play_min
        self.__selfplay_max_override = self_play_max
        self.__selfplay_stockfish_depth = self_play_stockfish_depth if self_play_stockfish_depth is not None else self.__config.stockfish_depth
        self.__selfplay_value_weight = self_play_value_weight if self_play_value_weight is not None else self.__config.self_play_value_weight
        self.__run_start_epoch = self.__start_epoch
        self.__run_end_epoch = end_epoch

        loss = 0.0
        value_loss = 0.0
        breakthrough_detector = LossBreakthroughDetector()
        for epoch in tqdm(range(self.__start_epoch, end_epoch)):
            epoch_start = time.perf_counter()

            cycle = 10
            attempt = 1
            while True:
                self_play_ratio = self.generate_games(epoch)
                loss, value_loss, stepped = self.train_on_games(epoch, self_play_ratio)
                if stepped:
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
                if is_breakthrough:
                    print(f"  loss breakthrough at epoch {epoch} ({loss:.4f}) — saving checkpoint")
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
