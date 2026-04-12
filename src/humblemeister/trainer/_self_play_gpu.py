from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import chess
import torch
import torch.nn.functional as F

from humblemeister.attention import KVCache, LayerKVCache
from humblemeister.config import ChessTrainingConfig
from humblemeister.data import ChessTokenizer, GameRecord
from humblemeister.evaluation import AsyncBatchEvaluator
from humblemeister.transformer import ChessTransformer

from ._self_play_cpu import _stockfish_outcome


def _select_cache_rows(cache: KVCache, indices: torch.Tensor) -> KVCache:
    """Return a new KVCache containing only the rows at `indices`."""
    return KVCache(layers=[
        LayerKVCache(k=layer.k[indices], v=layer.v[indices])
        for layer in cache.layers
    ])


class SelfPlayGPU:
    """
    GPU self-play generation — runs in the trainer process using the live model.

    Games are generated in batches: `batch_size` boards advance in lock-step,
    sharing a single stacked KV cache `[n_active, n_heads, seq_len, d_head]`.
    When a game ends its row is dropped from the cache so only active games pay
    the per-step cost.

    Policy path: one batched `generate_step` call per move step across all
    active games — O(1) activations per token, O(seq) cache growth.

    Value path (value_weight > 0): one batched full-recompute forward pass per
    active game per step over its legal moves.  Uses `is_causal=True` (Flash
    Attention) so peak memory is one layer's activations at a time.
    """

    def __init__(
        self,
        batch_size: int = 1,
        max_moves: int = 120,
        temperature: float = 1.0,
        value_weight: float = 0.0,
        stockfish_path: str = "stockfish",
        stockfish_depth: int = 5,
        stockfish_workers: int = 24,
        advantage_temperature: float = 1.0,
        draw_score_lo: float = 0.4,
        draw_score_hi: float = 0.6,
        bf16: bool = True,
    ) -> None:
        """
        Args:
            batch_size:            number of games to run in parallel per `_play_game` call.
            max_moves:             hard cap on game length (half-moves); Stockfish evaluates
                                   the final position when the cap is reached.
            temperature:           softmax temperature for policy sampling.
            value_weight:          blend weight for the value head (0 = pure policy).
                                   Values > 0 add a batched full-recompute forward pass per
                                   active game per step — higher VRAM and compute cost.
            stockfish_path:        path to the Stockfish binary.
            stockfish_depth:       Stockfish search depth for outcome and move-weight eval.
            stockfish_workers:     worker pool size for AsyncBatchEvaluator.
            advantage_temperature: temperature for per-move advantage softmax in move weights.
            draw_score_lo:         White win-probability below this → Black wins (default 0.4).
            draw_score_hi:         White win-probability above this → White wins (default 0.6).
            bf16:                  use bfloat16 autocast on CUDA devices.
        """
        self._batch_size = batch_size
        self._max_moves = max_moves
        self._temperature = temperature
        self._value_weight = value_weight
        self._stockfish_path = stockfish_path
        self._stockfish_depth = stockfish_depth
        self._stockfish_workers = stockfish_workers
        self._advantage_temperature = advantage_temperature
        self._draw_lo = draw_score_lo
        self._draw_hi = draw_score_hi
        self._bf16 = bf16

    def _play_game(
        self,
        model: ChessTransformer,
        tokenizer: ChessTokenizer,
        device: torch.device,
        batch_size: int = 1,
    ) -> list[dict[str, Any]]:
        """
        Play `batch_size` games simultaneously.

        All active games share a single stacked KV cache.  When a game ends its
        row is removed so the cache never grows beyond the active count.
        Returns a list of dicts (one per game) with keys 'moves', 'outcome', 'weights'.
        """
        use_autocast = self._bf16 and device.type == "cuda"

        boards: list[chess.Board] = [chess.Board() for _ in range(batch_size)]
        histories: list[list[int]] = [[tokenizer.BOS] for _ in range(batch_size)]

        # orig_idx maps the current active slot → original game index so results
        # are returned in the original order regardless of finishing order.
        orig_idx: list[int] = list(range(batch_size))
        results: list[dict[str, Any]] = [{}] * batch_size

        # Prime the stacked cache with BOS for all games.
        bos = torch.full((batch_size, 1), tokenizer.BOS, dtype=torch.long, device=device)
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_autocast):
                logits, _, cache = model.generate_step(bos, KVCache())
        # policy_logits: [n_active, vocab_size]
        policy_logits = logits[:, 0, :].clone()
        del logits, bos

        while boards:
            # ------------------------------------------------------------------ #
            #  Check termination for each active game                            #
            # ------------------------------------------------------------------ #
            still_active: list[int] = []
            for i, board in enumerate(boards):
                if board.is_game_over():
                    result = board.result()
                    outcome = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}.get(result, 0.5)
                elif len(board.move_stack) >= self._max_moves:
                    outcome = _stockfish_outcome(
                        board, self._stockfish_path, self._stockfish_depth,
                        self._draw_lo, self._draw_hi,
                    )
                else:
                    still_active.append(i)
                    continue  # not done yet — skip recording

                results[orig_idx[i]] = {
                    "moves": [m.uci() for m in board.move_stack],
                    "outcome": outcome,
                    "weights": None,
                }

            # Drop finished games from the active set.
            if len(still_active) < len(boards):
                if not still_active:
                    break
                boards = [boards[i] for i in still_active]
                histories = [histories[i] for i in still_active]
                orig_idx = [orig_idx[i] for i in still_active]
                policy_logits = policy_logits[still_active]
                idx_t = torch.tensor(still_active, device=device)
                cache = _select_cache_rows(cache, idx_t)

            n_active = len(boards)

            # ------------------------------------------------------------------ #
            #  Sample one move per active game                                    #
            # ------------------------------------------------------------------ #
            chosen_ids = torch.empty(n_active, dtype=torch.long, device=device)

            for i, (board, history) in enumerate(zip(boards, histories)):
                legal_moves = list(board.legal_moves)
                legal_tids = torch.tensor(
                    [tokenizer.encode_move(m) for m in legal_moves],
                    dtype=torch.long,
                    device=device,
                )  # [n_legal]

                legal_policy = policy_logits[i][legal_tids] / self._temperature

                if self._value_weight > 0.0:
                    # Batched full recompute over all resulting positions for this game.
                    # Peak memory: one layer of [n_legal, seq_len+1, d_model] at a time.
                    value_input = torch.tensor(
                        [history + [int(tid.item())] for tid in legal_tids],
                        dtype=torch.long,
                        device=device,
                    )  # [n_legal, seq_len+1]
                    with torch.no_grad():
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_autocast):
                            _, value_preds = model(value_input, is_causal=True)
                    value_scores = value_preds[:, -1].clone()  # [n_legal]
                    if board.turn == chess.BLACK:
                        value_scores = -value_scores
                    del value_input, value_preds
                    final_scores = legal_policy + self._value_weight * value_scores
                else:
                    final_scores = legal_policy

                probs = F.softmax(final_scores, dim=0)
                if torch.isnan(probs).any() or torch.isinf(probs).any():
                    pick = int(torch.randint(len(legal_moves), (1,)).item())
                else:
                    pick = int(torch.multinomial(probs, num_samples=1).item())

                chosen_ids[i] = legal_tids[pick]
                board.push(legal_moves[pick])
                history.append(int(legal_tids[pick].item()))
                del legal_tids, legal_policy, final_scores, probs

            # ------------------------------------------------------------------ #
            #  Advance the stacked KV cache with all chosen tokens               #
            # ------------------------------------------------------------------ #
            tok = chosen_ids.unsqueeze(1)  # [n_active, 1]
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_autocast):
                    logits, _, cache = model.generate_step(tok, cache)
            policy_logits = logits[:, 0, :].clone()  # [n_active, vocab_size]
            del logits, tok

        return results

    def generate(
        self,
        model: ChessTransformer,
        model_config: ChessTrainingConfig,
        tokenizer: ChessTokenizer,
        n_games: int,
    ) -> list[GameRecord]:
        """
        Generate n_games self-play games using the live GPU model.

        Games are generated in batches of `self._batch_size`.  The model is
        temporarily set to eval mode and restored afterwards.  model_config is
        accepted for interface parity with SelfPlayCPU but is not used here.

        Args:
            model:        the live ChessTransformer (on any device).
            model_config: unused — kept for interface parity with SelfPlayCPU.
            tokenizer:    ChessTokenizer matching the model vocabulary.
            n_games:      total number of games to generate.

        Returns:
            List of GameRecord objects with move_weights populated by Stockfish.
        """
        if n_games <= 0:
            return []

        device = next(model.parameters()).device
        was_training = model.training
        model.eval()

        raw_games: list[dict[str, Any]] = []
        batches = range(0, n_games, self._batch_size)
        try:
            for batch_start in range(0, n_games, self._batch_size):
                batch_n = min(self._batch_size, n_games - batch_start)
                raw_games.extend(self._play_game(model, tokenizer, device, batch_n))
        finally:
            if was_training:
                model.train()

        if not raw_games:
            return []

        # ------------------------------------------------------------------ #
        #  Stockfish per-move importance weights                              #
        # ------------------------------------------------------------------ #
        with tempfile.TemporaryDirectory(prefix="humblemeister_sp_") as tmpdir:
            batch_path = Path(tmpdir) / "batch.pt"
            torch.save(raw_games, batch_path)

            n_sf_workers = min(self._stockfish_workers, len(raw_games))
            with AsyncBatchEvaluator(
                stockfish_path=self._stockfish_path,
                depth=self._stockfish_depth,
                temperature=self._advantage_temperature,
                n_workers=n_sf_workers,
            ) as evaluator:
                evaluator.submit(batch_path)
                evaluator.drain()

            all_games: list[dict[str, Any]] = torch.load(batch_path, weights_only=False)

        # ------------------------------------------------------------------ #
        #  Convert to GameRecord                                              #
        # ------------------------------------------------------------------ #
        records: list[GameRecord] = []
        for item in all_games:
            moves_uci: list[str] = item.get("moves", [])
            try:
                moves = [chess.Move.from_uci(uci) for uci in moves_uci]
                tensor = tokenizer.encode_game_tensor(moves)
            except Exception:
                continue
            records.append(
                GameRecord(
                    moves=moves,
                    outcome=float(item["outcome"]),
                    tensor=tensor,
                    move_weights=item.get("weights"),
                )
            )

        return records
