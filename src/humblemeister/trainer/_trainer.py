from __future__ import annotations

import ctypes
import gc
import json
import math
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import safetensors.torch as st
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from humblemeister.attention import make_causal_mask, make_padding_mask
from humblemeister.config import (
    BatchLengthSampling,
    ChessTrainingConfig,
    SelfPlayLossMode,
    TrainingSelfAttention,
)
from humblemeister.data import (
    ChessDataset,
    ChessGameBank,
    ChessTokenizer,
    GameRecord,
    LengthBucketBatchSampler,
)
from humblemeister.evaluation import StockfishEvaluator, compute_move_weights
from humblemeister.transformer import ChessTransformer

from ._loss_tracker import LossBreakthroughDetector
from ._self_play_gpu import SelfPlayGPU


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    warmup_epochs: int,
    start_epoch: int = 0,
) -> LambdaLR:
    """Cosine-annealing LR with linear warmup.

    The cosine curve spans [warmup_epochs, n_epochs].  ``start_epoch``
    offsets the internal step counter so that ``step=0`` corresponds to
    ``epoch=start_epoch`` — this lets a resumed run pick up at the correct
    position on the curve without replaying scheduler steps.
    """

    def lr_lambda(step: int) -> float:
        epoch = start_epoch + step
        if epoch < warmup_epochs:
            return epoch / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(n_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


class ChessTrainer:
    __config: ChessTrainingConfig
    __device: torch.device
    __model: ChessTransformer
    __optimizer: AdamW
    __scheduler: LambdaLR
    __tokenizer: ChessTokenizer
    __dataset: ChessDataset
    __gamebank: ChessGameBank
    __last_loss: float
    __disable_selfplay: bool
    __selfplay_min_override: float | None
    __selfplay_max_override: float | None
    __run_start_epoch: int
    __run_end_epoch: int
    __self_play: SelfPlayGPU
    __sp_gen_time_s: float
    __sp_eval_time_s: float
    __sp_games_generated: int
    __sp_games_included: int

    def __init__(self, config: ChessTrainingConfig, game_bank: ChessGameBank) -> None:
        self.__config = config
        self.__device = torch.device(config.device)
        self.__last_loss = 0.0
        self.__disable_selfplay = False
        self.__selfplay_min_override: float | None = None
        self.__selfplay_max_override: float | None = None
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
        self.__self_play = SelfPlayGPU(
            batch_size=config.self_play_batch_size,
            max_moves=config.self_play_max_moves,
            value_weight=config.self_play_value_weight,
            stockfish_path=config.stockfish_path,
            stockfish_depth=config.stockfish_depth,
            stockfish_workers=config.stockfish_workers,
            advantage_temperature=config.advantage_temperature,
        )

        self.__model = ChessTransformer(
            vocab_size=self.__tokenizer.vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            pad_id=self.__tokenizer.PAD,
        ).to(self.__device)

        self.__optimizer = AdamW(self.__model.parameters(), lr=config.lr, weight_decay=0.1)
        self.__scheduler = get_scheduler(self.__optimizer, config.n_epochs, config.warmup_epochs)

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
                    "config": asdict(self.__config),
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
            tokens, outcome, weights, value_evals = self.__gamebank.get_random_game()
            n_moves = tokens.numel() - 2  # subtract BOS + EOS
            if n_moves <= 0:
                continue
            if n_moves > self.__config.max_moves:
                continue

            # Keep the bank's int16 representation directly in the GameRecord.
            # The cast to int64 happens in ChessDataset.collate, right before
            # pad_sequence, so the larger dtype is only alive for one batch.
            tensor = tokens

            # use pre-computed weights from the bank if available;
            # fall back to on-the-fly evaluation only if the bank wasn't pre-evaluated
            if weights is None and self.__evaluator is not None:
                moves = self.__tokenizer.decode_game(tokens.tolist())
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
                    outcome=outcome,
                    tensor=tensor,
                    move_weights=weights,
                    value_evals=value_evals,
                )
            )

    def __build_dataloader(
        self,
        dataset: ChessDataset,
        batch_size: int,
        epoch: int,
    ) -> DataLoader[Any]:
        """Assemble a DataLoader for one training pass.

        When training_self_attention is FLASH and batch_length_sampling is
        BUCKETED, a LengthBucketBatchSampler yields fixed-length batches (zero
        intra-batch padding). Any other combination falls back to the standard
        shuffle=True loader.
        """
        fa_on = (
            TrainingSelfAttention(self.__config.training_self_attention)
            == TrainingSelfAttention.FLASH
        )
        bucket_on = (
            BatchLengthSampling(self.__config.batch_length_sampling) == BatchLengthSampling.BUCKETED
        )
        if fa_on and bucket_on:
            sampler = LengthBucketBatchSampler(
                dataset,
                batch_size=batch_size,
                n_games=min(self.__config.n_games, len(dataset)),
                seed=self.__config.bucket_sampler_seed,
                epoch=epoch,
            )
            return DataLoader(dataset, batch_sampler=sampler, collate_fn=dataset.collate)
        return DataLoader(
            dataset,
            batch_size=min(batch_size, len(dataset)),
            shuffle=True,
            collate_fn=dataset.collate,
        )

    def __forward_train(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the model forward in training mode, dispatching between the
        Flash-Attention-friendly causal-only path and the legacy combined
        causal+padding mask path based on config.training_self_attention.
        """
        mode = TrainingSelfAttention(self.__config.training_self_attention)
        if mode == TrainingSelfAttention.FLASH:
            logits, value = self.__model(input_ids, mask=None, is_causal=True)
            return logits, value
        causal_mask = make_causal_mask(input_ids.size(1), self.__device)
        padding_mask = make_padding_mask(attention_mask).to(self.__device)
        logits, value = self.__model(input_ids, causal_mask + padding_mask)
        return logits, value

    def __policy_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        move_weights: torch.Tensor,
        is_self_play: torch.Tensor,
        loss_mode: SelfPlayLossMode,
    ) -> torch.Tensor:
        """
        Compute the policy loss, branching per sample based on loss_mode.

        VALUE_ONLY:          self-play samples contribute zero policy loss;
                             only the value head trains on them.
        ADVANTAGE_WEIGHTED:  bank games use label_smoothing=0.1 (imitation);
                             self-play games use label_smoothing=0.0 (clean
                             policy gradient weighted by Stockfish advantages).
        """
        vocab = self.__tokenizer.vocab_size
        pad = self.__tokenizer.PAD
        seq = targets.size(1)

        is_sp_flat = is_self_play.unsqueeze(1).expand(-1, seq).reshape(-1)  # [B*T]
        logits_flat = logits.view(-1, vocab)
        targets_flat = targets.view(-1)

        if loss_mode == SelfPlayLossMode.VALUE_ONLY:
            per_token = F.cross_entropy(
                logits_flat,
                targets_flat,
                ignore_index=pad,
                label_smoothing=0.1,
                reduction="none",
            )
            policy_mask = (~is_sp_flat).float()
        else:  # ADVANTAGE_WEIGHTED
            smooth = F.cross_entropy(
                logits_flat,
                targets_flat,
                ignore_index=pad,
                label_smoothing=0.1,
                reduction="none",
            )
            raw = F.cross_entropy(
                logits_flat,
                targets_flat,
                ignore_index=pad,
                label_smoothing=0.0,
                reduction="none",
            )
            per_token = torch.where(is_sp_flat, raw, smooth)
            policy_mask = torch.ones(targets_flat.size(0), device=targets_flat.device)

        pad_mask = (targets_flat != pad).float()
        denom = pad_mask.sum().clamp(min=1)
        return (per_token * move_weights.view(-1) * pad_mask * policy_mask).sum() / denom

    def __generate_self_play(self, n_self_play: int) -> None:
        """Generate n_self_play games via SelfPlay and add them to the dataset."""
        t_gen = time.perf_counter()
        self.__model.eval()
        with tqdm(total=n_self_play, desc="self-play", unit="game", leave=False) as pbar:
            records = self.__self_play.generate(
                self.__model, self.__config, self.__tokenizer, n_self_play
            )
            pbar.update(len(records))
        self.__model.train()
        self.__sp_gen_time_s += time.perf_counter() - t_gen
        self.__sp_games_generated += len(records)

        for record in records:
            self.__dataset.add_game(record)
            self.__sp_games_included += 1

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
        loss_mode = SelfPlayLossMode(self.__config.self_play_loss_mode)

        sp_pbar = (
            tqdm(total=n_self_play, desc="self-play", unit="game", leave=False)
            if n_self_play > 0
            else None
        )
        for b_this, sp_this in chunks:
            self.__dataset.clear()

            # ---------------------------------------------------------- #
            # Generate games for this chunk (eval mode, no grad)         #
            # ---------------------------------------------------------- #
            self.__model.eval()

            if b_this > 0:
                self.__generate_from_bank(b_this)

            if sp_this > 0:
                t_gen = time.perf_counter()
                records = self.__self_play.generate(
                    self.__model, self.__config, self.__tokenizer, sp_this
                )
                self.__sp_gen_time_s += time.perf_counter() - t_gen
                self.__sp_games_generated += len(records)
                for record in records:
                    self.__dataset.add_game(record)
                    self.__sp_games_included += 1
                if sp_pbar is not None:
                    sp_pbar.update(len(records))

            if len(self.__dataset) == 0:
                continue

            # ---------------------------------------------------------- #
            # Train on this chunk (train mode, accumulate gradients)     #
            # ---------------------------------------------------------- #
            self.__model.train()

            dataloader = self.__build_dataloader(self.__dataset, train_bs, epoch)

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
                is_self_play = cast(torch.Tensor, batch["is_self_play"]).to(self.__device)

                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.__config.bf16):
                    logits, value_pred = self.__forward_train(input_ids, attention_mask)

                    policy_loss = self.__policy_loss(
                        logits, targets, move_weights, is_self_play, loss_mode
                    )

                    token_mask = attention_mask.bool() & has_evals.unsqueeze(1)
                    if token_mask.any():
                        value_loss_t = F.mse_loss(value_pred[token_mask], value_targets[token_mask])
                        loss = policy_loss + self.__config.value_loss_weight * value_loss_t
                    else:
                        value_loss_t = policy_loss.new_zeros(())
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

        if sp_pbar is not None:
            sp_pbar.close()

        if total_mini_batches == 0:
            print("warning: all mini-batches had NaN/inf loss, skipping update")
            self.__optimizer.zero_grad()
            return 0.0, 0.0, False

        # Normalize: divide accumulated gradients so the effective update equals
        # the mean loss over all mini-batches (same semantics as train_on_games)
        for p in self.__model.parameters():
            if p.grad is not None:
                p.grad /= total_mini_batches

        grad_norm = torch.nn.utils.clip_grad_norm_(self.__model.parameters(), max_norm=1.0)
        if not torch.isfinite(grad_norm):
            print(f"warning: non-finite gradient norm ({grad_norm:.4g}), skipping update")
            self.__optimizer.zero_grad()
            return total_loss / total_mini_batches, total_value_loss / total_mini_batches, False

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

        dataloader = self.__build_dataloader(self.__dataset, batch_size, epoch)

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
        loss_mode = SelfPlayLossMode(self.__config.self_play_loss_mode)

        for batch in dataloader:
            input_ids = cast(torch.Tensor, batch["input_ids"]).to(self.__device)
            targets = cast(torch.Tensor, batch["targets"]).to(self.__device)
            attention_mask = cast(torch.Tensor, batch["attention_mask"]).to(self.__device)
            move_weights = cast(torch.Tensor, batch["move_weights"]).to(self.__device)
            value_targets = cast(torch.Tensor, batch["value_evals"]).to(self.__device)
            has_evals = cast(torch.Tensor, batch["has_value_evals"]).to(self.__device)
            is_self_play = cast(torch.Tensor, batch["is_self_play"]).to(self.__device)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.__config.bf16):
                logits, value_pred = self.__forward_train(input_ids, attention_mask)

                policy_loss = self.__policy_loss(
                    logits, targets, move_weights, is_self_play, loss_mode
                )

                # value loss — only on positions from games that have Stockfish evals
                token_mask = attention_mask.bool() & has_evals.unsqueeze(1)
                if token_mask.any():
                    value_loss = F.mse_loss(value_pred[token_mask], value_targets[token_mask])
                    loss = policy_loss + self.__config.value_loss_weight * value_loss
                else:
                    value_loss = policy_loss.new_zeros(())
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

        # clip_grad_norm_ computes the global gradient norm as a by-product.
        # Checking that single scalar is far cheaper than iterating every parameter
        # with torch.isnan(p.grad).any() (which allocates a bool tensor per param).
        grad_norm = torch.nn.utils.clip_grad_norm_(self.__model.parameters(), max_norm=1.0)
        if not torch.isfinite(grad_norm):
            print(f"warning: non-finite gradient norm ({grad_norm:.4g}), skipping update")
            self.__optimizer.zero_grad()
            return total_loss, total_value_loss, False

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

            dataloader = self.__build_dataloader(
                self.__dataset, self.__config.train_batch_size, epoch
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

                # mask: real tokens in games that actually have Stockfish evals
                token_mask = attention_mask.bool() & has_evals.unsqueeze(1)

                if not token_mask.any():
                    continue

                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.__config.bf16):
                    _, value_pred = self.__forward_train(input_ids, attention_mask)
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

        self.__save_checkpoint(0, self.__last_loss)
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
        """Start training from scratch (epoch 0).  Clears existing checkpoints."""
        if self.checkpoints_path.exists():
            shutil.rmtree(self.checkpoints_path)
        os.makedirs(self.checkpoints_path, exist_ok=True)

        end_epoch = max_epochs or self.__config.n_epochs
        self.__scheduler = get_scheduler(
            self.__optimizer,
            end_epoch,
            self.__config.warmup_epochs,
        )
        self.__train_loop(
            start_epoch=0,
            end_epoch=end_epoch,
            disable_selfplay=disable_selfplay,
            self_play_min=self_play_min,
            self_play_max=self_play_max,
            self_play_stockfish_depth=self_play_stockfish_depth,
            self_play_value_weight=self_play_value_weight,
        )

    def resume(
        self,
        start_epoch: int | None = None,
        end_epoch: int | None = None,
        max_epochs: int | None = None,
        disable_selfplay: bool = False,
        self_play_min: float | None = None,
        self_play_max: float | None = None,
        self_play_stockfish_depth: int | None = 5,
        self_play_value_weight: float | None = None,
    ) -> None:
        """Resume training from the latest checkpoint.

        The scheduler is rebuilt from scratch so that the LR curve is
        consistent with the given ``start_epoch`` / ``end_epoch``.

        Args:
            start_epoch:  Override the starting epoch (default: checkpoint epoch + 1).
            end_epoch:    Override the target end epoch (default: from checkpoint).
            max_epochs:   Run for this many epochs from start_epoch (alternative to
                          ``end_epoch``; ignored when ``end_epoch`` is set).
            disable_selfplay / self_play_*:  same semantics as ``run()``.
        """
        checkpoint = self.__load_latest_checkpoint()
        if checkpoint is None:
            raise RuntimeError("no checkpoint found — use run() to start fresh")

        cp_epoch: int = checkpoint["epoch"]
        cp_end: int = checkpoint.get("end_epoch", self.__config.n_epochs)
        # Release the checkpoint dict (which holds a full model_state copy) before
        # entering the long-running training loop — otherwise it lives on the stack
        # for the entire training run, wasting RAM proportional to model size.
        del checkpoint

        current = start_epoch if start_epoch is not None else cp_epoch + 1
        if end_epoch is not None:
            target_end = end_epoch
        elif max_epochs is not None:
            target_end = current + max_epochs
        else:
            target_end = cp_end

        if current >= target_end:
            print(f"nothing to do: start_epoch={current} >= end_epoch={target_end}")
            return

        os.makedirs(self.checkpoints_path, exist_ok=True)
        self.__scheduler = get_scheduler(
            self.__optimizer,
            target_end,
            self.__config.warmup_epochs,
            start_epoch=current,
        )
        print(
            f"LR curve: epoch {current} → {target_end} " f"(warmup {self.__config.warmup_epochs})"
        )
        self.__train_loop(
            start_epoch=current,
            end_epoch=target_end,
            disable_selfplay=disable_selfplay,
            self_play_min=self_play_min,
            self_play_max=self_play_max,
            self_play_stockfish_depth=self_play_stockfish_depth,
            self_play_value_weight=self_play_value_weight,
        )

    def __train_loop(
        self,
        start_epoch: int,
        end_epoch: int,
        disable_selfplay: bool,
        self_play_min: float | None,
        self_play_max: float | None,
        self_play_stockfish_depth: int | None,
        self_play_value_weight: float | None,
    ) -> None:
        print(f"training on {self.__config.device}")
        print(f"model parameters: {sum(p.numel() for p in self.__model.parameters()):,}")

        self.__disable_selfplay = disable_selfplay
        self.__selfplay_min_override = self_play_min
        self.__selfplay_max_override = self_play_max
        if self_play_stockfish_depth is not None:
            self.__self_play._stockfish_depth = self_play_stockfish_depth
        if self_play_value_weight is not None:
            self.__self_play._value_weight = self_play_value_weight
        self.__run_start_epoch = start_epoch
        self.__run_end_epoch = end_epoch

        loss = 0.0
        value_loss = 0.0
        self_play_ratio = 0.0
        breakthrough_detector = LossBreakthroughDetector()
        for epoch in tqdm(range(start_epoch, end_epoch)):
            epoch_start = time.perf_counter()

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
            if not stepped:
                raise RuntimeError('The trainer has failed to make a step.')

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

            # Every 50 epochs: run the cyclic GC and ask glibc to return any
            # freed-but-unreturned pages to the OS.  Python's allocator holds
            # arenas even after freeing objects; malloc_trim(0) releases them,
            # preventing the process RSS from growing indefinitely due to
            # heap fragmentation from the repeated per-epoch alloc/free churn.
            if epoch > 0 and epoch % 50 == 0:
                gc.collect()
                try:
                    ctypes.CDLL("libc.so.6").malloc_trim(0)
                except Exception:
                    pass  # non-Linux or libc unavailable — silently skip

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
        lengths = [max(g.tensor.numel() - 2, 0) for g in games]

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
                "end_epoch": self.__run_end_epoch,
                "model_state": self.__model.state_dict(),
                "optimizer_state": self.__optimizer.state_dict(),
                "loss": loss,
            },
            path,
        )
        self.__prune_checkpoints()

    def __prune_checkpoints(self) -> None:
        checkpoints = sorted(self.checkpoints_path.glob("checkpoint_epoch_*.pt"))
        for old in checkpoints[: -self.__config.keep_last_n]:
            old.unlink()

    def __load_latest_checkpoint(self) -> dict[str, Any] | None:
        """Load the latest checkpoint, restoring model and optimizer state.

        Returns the raw checkpoint dict (with keys ``epoch``, ``end_epoch``,
        ``loss``, etc.) or ``None`` if no checkpoints exist.
        """
        checkpoints = sorted(self.checkpoints_path.glob("checkpoint_epoch_*.pt"))
        if not checkpoints:
            return None

        path = checkpoints[-1]
        checkpoint = torch.load(path, map_location=self.__device, weights_only=True)

        result = self.__model.load_state_dict(checkpoint["model_state"], strict=False)
        if result.missing_keys:
            print(f"  initialized missing keys from scratch: {result.missing_keys}")
        if result.unexpected_keys:
            print(f"  ignored unexpected keys: {result.unexpected_keys}")
        try:
            self.__optimizer.load_state_dict(checkpoint["optimizer_state"])
        except ValueError:
            print("  optimizer state incompatible (model changed), starting optimizer fresh")
        self.__last_loss = float(checkpoint["loss"])

        print(
            f"resumed from {path.name} (epoch {checkpoint['epoch']}, loss {checkpoint['loss']:.4f})"
        )
        return dict(checkpoint)

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
