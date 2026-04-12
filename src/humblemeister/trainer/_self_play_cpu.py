from __future__ import annotations

import math
import tempfile
from dataclasses import asdict
from multiprocessing import Process
from pathlib import Path
from typing import Any

import chess
import chess.engine
import torch

from humblemeister.config import ChessTrainingConfig
from humblemeister.data import ChessTokenizer, GameRecord
from humblemeister.evaluation import AsyncBatchEvaluator
from humblemeister.transformer import ChessTransformer

from .._engine import ChessEngine

# ------------------------------------------------------------------ #
#  Module-level worker helpers (must be at module level for "spawn")  #
# ------------------------------------------------------------------ #


def _stockfish_outcome(
    board: chess.Board,
    stockfish_path: str,
    depth: int,
    draw_lo: float,
    draw_hi: float,
) -> float:
    """
    Evaluate board with Stockfish and return a game outcome from White's perspective.

    The Stockfish centipawn score is converted to a win probability via a logistic
    curve (200 cp ≈ +10 pp).  Anything in [draw_lo, draw_hi] is a draw (0.5),
    outside that range the leading side wins.
    """
    try:
        sf = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        try:
            info = sf.analyse(board, chess.engine.Limit(depth=depth))
        finally:
            sf.quit()

        score_obj = info.get("score")
        if score_obj is None:
            return 0.5

        cp = score_obj.white().score(mate_score=2000)
        if cp is None:
            return 0.5

        # logistic conversion: 0 cp → 0.5, +200 cp → ~0.73, -200 cp → ~0.27
        wp = 1.0 / (1.0 + math.exp(-cp / 200.0))
        if draw_lo <= wp <= draw_hi:
            return 0.5
        return 1.0 if wp > draw_hi else 0.0

    except Exception:
        return 0.5  # safest default if Stockfish fails


def _worker_fn(
    model_path: str,
    n_games: int,
    output_path: str,
    temperature: float,
    value_weight: float,
    use_kv_cache: bool,
    max_moves: int,
    stockfish_path: str,
    stockfish_depth: int,
    draw_lo: float,
    draw_hi: float,
) -> None:
    """
    Runs in a forked child process.

    Loads the model on CPU, generates n_games self-play games, and saves them as a
    list[dict] to output_path.  Each dict has keys "moves" (list of UCI strings),
    "outcome" (float), and "weights" (None — filled in later by AsyncBatchEvaluator).

    Per-game latency metrics are written to a .log file alongside output_path.
    """
    import time

    log_path = output_path.replace(".pt", ".log")
    engine = ChessEngine.load(
        model_path,
        device="cpu",
        temperature=temperature,
        value_weight=value_weight,
        use_kv_cache=use_kv_cache,
    )

    games: list[dict[str, Any]] = []

    with open(log_path, "w", buffering=1) as log:
        for game_idx in range(n_games):
            engine.start_game(chess.WHITE)
            move_latencies: list[float] = []

            while True:
                if engine.board.is_game_over():
                    result = engine.board.result()
                    outcome = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}.get(result, 0.5)
                    end_reason = f"result={result}"
                    break

                if len(engine.board.move_stack) >= max_moves:
                    outcome = _stockfish_outcome(
                        engine.board, stockfish_path, stockfish_depth, draw_lo, draw_hi
                    )
                    end_reason = f"max_moves outcome={outcome:.1f}"
                    break

                t0 = time.perf_counter()
                move = engine.sample_move()
                move_latencies.append(time.perf_counter() - t0)
                engine.apply_move(move)

            n_moves = len(engine.board.move_stack)
            if move_latencies:
                avg_ms = sum(move_latencies) / len(move_latencies) * 1000
                min_ms = min(move_latencies) * 1000
                max_ms = max(move_latencies) * 1000
                log.write(
                    f"game {game_idx + 1}/{n_games}  moves={n_moves}  {end_reason}  "
                    f"move_ms avg={avg_ms:.0f} min={min_ms:.0f} max={max_ms:.0f}\n"
                )

            games.append(
                {
                    "moves": [m.uci() for m in engine.board.move_stack],
                    "outcome": outcome,
                    "weights": None,
                }
            )

    torch.save(games, output_path)


class SelfPlayCPU:
    """
    Orchestrates multi-process self-play generation using ChessEngine on CPU.

    Workflow
    --------
    1. Save the current model weights to a temp file (cleaned up automatically).
    2. Spawn up to n_workers child processes; each loads the model on CPU via
       ChessEngine.load() and plays its share of games.
    3. At max_moves, Stockfish evaluates the final position to decide the outcome
       instead of calling it a draw unconditionally.
    4. Collect all game dicts, send them as a single batch to AsyncBatchEvaluator
       for Stockfish move-weight computation (per-move importance for the policy loss).
    5. Return a list of GameRecord objects ready for the training dataset.
    """

    def __init__(
        self,
        n_workers: int = 1,
        max_moves: int = 160,
        temperature: float = 1.0,
        value_weight: float = 1.0,
        use_kv_cache: bool = True,
        stockfish_path: str = "stockfish",
        stockfish_depth: int = 5,
        stockfish_workers: int = 24,
        advantage_temperature: float = 1.0,
        draw_score_lo: float = 0.4,
        draw_score_hi: float = 0.6,
    ) -> None:
        """
        Args:
            n_workers:             maximum number of parallel child processes for generation.
            max_moves:             hard cap on game length (half-moves / plies); when reached
                                   Stockfish evaluates the position to determine the winner.
            temperature:           ChessEngine move-sampling temperature.
            value_weight:          ChessEngine value-head blend weight (0 = pure policy).
            use_kv_cache:          whether ChessEngine uses KV caching during generation.
            stockfish_path:        path to the Stockfish binary.
            stockfish_depth:       Stockfish search depth used for both outcome determination
                                   and move-weight computation.
            stockfish_workers:     worker pool size for AsyncBatchEvaluator.
            advantage_temperature: temperature for softmax over per-move advantages when
                                   computing move weights (lower = sharper weighting).
            draw_score_lo:         lower bound of White's win-probability range treated as a draw
                                   (default 0.4 — position below this means Black won).
            draw_score_hi:         upper bound of White's win-probability range treated as a draw
                                   (default 0.6 — position above this means White won).
        """
        self._n_workers = n_workers
        self._max_moves = max_moves
        self._temperature = temperature
        self._value_weight = value_weight
        self._use_kv_cache = use_kv_cache
        self._stockfish_path = stockfish_path
        self._stockfish_depth = stockfish_depth
        self._stockfish_workers = stockfish_workers
        self._advantage_temperature = advantage_temperature
        self._draw_lo = draw_score_lo
        self._draw_hi = draw_score_hi

    def generate(
        self,
        model: ChessTransformer,
        model_config: ChessTrainingConfig,
        tokenizer: ChessTokenizer,
        n_games: int,
    ) -> list[GameRecord]:
        """
        Generate n_games self-play games.

        The model is saved to a temporary file before spawning workers.  The temp
        directory (and all intermediate files) is deleted automatically when this
        method returns, whether or not an exception occurred.

        Args:
            model:        trained ChessTransformer (on any device — copied to CPU for workers).
            model_config: ChessTrainingConfig that describes the model architecture.
            tokenizer:    ChessTokenizer matching the model vocabulary.
            n_games:      total number of self-play games to generate.

        Returns:
            List of GameRecord objects, each with move_weights populated by Stockfish
            (if use_stockfish=True) and value_evals=None.
        """
        if n_games <= 0:
            return []

        with tempfile.TemporaryDirectory(prefix="humblemeister_sp_") as tmpdir:
            tmpdir_path = Path(tmpdir)

            # ---------------------------------------------------------- #
            # Step 1: save model weights to a temp file                   #
            # ---------------------------------------------------------- #
            model_path = tmpdir_path / "model.pt"
            state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save({"config": asdict(model_config), "model_state": state_dict}, model_path)

            # ---------------------------------------------------------- #
            # Step 2: distribute games across workers and spawn them      #
            # ---------------------------------------------------------- #
            n_workers = min(n_games, self._n_workers)
            base, remainder = divmod(n_games, n_workers)
            game_counts = [base + (1 if i < remainder else 0) for i in range(n_workers)]

            procs = []
            output_paths: list[Path] = []

            for i, count in enumerate(game_counts):
                if count == 0:
                    continue
                out_path = tmpdir_path / f"worker_{i}.pt"
                output_paths.append(out_path)
                p = Process(
                    target=_worker_fn,
                    args=(
                        str(model_path),
                        count,
                        str(out_path),
                        self._temperature,
                        self._value_weight,
                        self._use_kv_cache,
                        self._max_moves,
                        self._stockfish_path,  # always needed for max_moves outcome evaluation
                        self._stockfish_depth,
                        self._draw_lo,
                        self._draw_hi,
                    ),
                    daemon=True,
                )
                p.start()
                procs.append(p)

            for p in procs:
                p.join()

            # ---------------------------------------------------------- #
            # Step 3: collect all raw game dicts from worker output files #
            # ---------------------------------------------------------- #
            all_games: list[dict[str, Any]] = []
            for out_path in output_paths:
                if out_path.exists():
                    data: list[dict[str, Any]] = torch.load(out_path, weights_only=False)
                    all_games.extend(data)

            if not all_games:
                return []

            # ---------------------------------------------------------- #
            # Step 4: Stockfish evaluation for per-move importance weights #
            # ---------------------------------------------------------- #
            batch_path = tmpdir_path / "batch.pt"
            torch.save(all_games, batch_path)

            n_sf_workers = min(self._stockfish_workers, len(all_games))
            with AsyncBatchEvaluator(
                stockfish_path=self._stockfish_path,
                depth=self._stockfish_depth,
                temperature=self._advantage_temperature,
                n_workers=n_sf_workers,
            ) as evaluator:
                evaluator.submit(batch_path)
                evaluator.drain()

            # batch_path is updated in-place by the evaluator
            all_games = torch.load(batch_path, weights_only=False)

            # ---------------------------------------------------------- #
            # Step 5: convert to GameRecords                              #
            # ---------------------------------------------------------- #
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
