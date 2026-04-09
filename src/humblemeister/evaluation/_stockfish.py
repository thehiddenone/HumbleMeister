from __future__ import annotations

import queue
import chess
import chess.engine
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import TracebackType


_MATE_CAP_CP = 2000  # clamp mate scores to ±2000cp so they don't dominate advantage normalization


class StockfishEvaluator:
    """
    Wraps a pool of N Stockfish processes and a thread pool so that multiple
    board positions can be evaluated in parallel.

    SimpleEngine is not thread-safe, so each worker thread exclusively borrows
    one engine from a Queue, evaluates, then returns it.  The Queue blocks if
    all engines are busy, naturally throttling concurrency to n_workers.
    """

    def __init__(self, path: str = "stockfish", n_workers: int = 4) -> None:
        # one engine process per worker — never shared between threads simultaneously
        self._engines = [
            chess.engine.SimpleEngine.popen_uci(path)
            for _ in range(n_workers)
        ]
        self._pool: queue.Queue[chess.engine.SimpleEngine] = queue.Queue()
        for engine in self._engines:
            self._pool.put(engine)

        self._executor = ThreadPoolExecutor(max_workers=n_workers)

    # ------------------------------------------------------------------ #
    #  Single evaluation                                                   #
    # ------------------------------------------------------------------ #

    def evaluate(self, board: chess.Board, depth: int = 5) -> float:
        """
        Evaluates one position. Borrows an engine from the pool, blocks if all
        are busy. Returns centipawns from the side-to-move's perspective.
        Mate scores are capped at ±2000cp.
        """
        engine = self._pool.get()
        try:
            return _score(engine, board, depth)
        finally:
            self._pool.put(engine)

    # ------------------------------------------------------------------ #
    #  Parallel batch evaluation                                           #
    # ------------------------------------------------------------------ #

    def evaluate_many(self, boards: list[chess.Board], depth: int = 5) -> list[float]:
        """
        Evaluates all boards in parallel across the engine pool.
        Returns results in the same order as the input list.
        """
        # submit all evaluations — each future captures its index for ordering
        futures = {
            self._executor.submit(self.evaluate, board, depth): i
            for i, board in enumerate(boards)
        }
        results = [0.0] * len(boards)
        for future in as_completed(futures):
            results[futures[future]] = future.result()
        return results

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        self._executor.shutdown(wait=True)
        for engine in self._engines:
            engine.quit()

    def __enter__(self) -> StockfishEvaluator:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val:  BaseException | None,
        _exc_tb:   TracebackType | None,
    ) -> None:
        self.close()


def _score(engine: chess.engine.SimpleEngine, board: chess.Board, depth: int) -> float:
    """Evaluate one position with a given engine. Not thread-safe on its own —
    callers must ensure exclusive access to the engine."""
    info = engine.analyse(board, chess.engine.Limit(depth=depth))

    # "score" is optional in InfoDict — always present after analyse() in practice,
    # but the type system doesn't guarantee it
    pov_score = info.get("score")
    if pov_score is None:
        return 0.0

    score = pov_score.relative  # centipawns from the side-to-move's perspective

    if score.is_mate():
        mate = score.mate()
        assert mate is not None  # guaranteed by is_mate()
        return float(_MATE_CAP_CP if mate > 0 else -_MATE_CAP_CP)

    cp = score.score()
    assert cp is not None  # guaranteed when not a mate score
    return float(cp)


def compute_move_weights(
    moves:       list[chess.Move],
    evaluator:   StockfishEvaluator,
    depth:       int   = 5,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Returns a weight tensor of shape [N + 1] where N = len(moves).

    Index t corresponds to the prediction target at collate time:
      - t in [0, N-1]: importance weight for predicting move t
      - t = N:         weight for the EOS token (always 1.0)

    Advantage derivation
    ────────────────────
    Stockfish reports centipawns from the side-to-move's perspective, so the
    sign flips after each move (white's turn → black's turn → ...).

    Let e[i] = eval at position i from the perspective of the player to move at i.

      e[0] = starting position (white to move)
      e[1] = after move 1     (black to move) — positive means black is better
      e[2] = after move 2     (white to move) — positive means white is better
      ...

    Advantage of move t, from the mover's perspective:
        adv[t] = (value of new position for the mover)
               - (value of old position for the mover)
               = -e[t] - e[t-1]

    Why -e[t]? Because e[t] is from the *opponent's* POV after the move was
    played; negating it converts it back to the mover's POV.

    Weight transformation
    ─────────────────────
    1. Z-score advantages within the game so signal is relative, not absolute.
    2. Softmax with temperature → probability distribution over moves.
    3. Scale by N so the average weight = 1.0 (no net change to overall loss magnitude).
    """
    if not moves:
        return torch.ones(1)  # just EOS

    # replay the game to collect all N+1 board states (before + after each move)
    board  = chess.Board()
    boards = [board.copy()]
    for move in moves:
        board.push(move)
        boards.append(board.copy())

    # evaluate all positions in parallel — the single sequential bottleneck is gone
    evals = evaluator.evaluate_many(boards, depth)

    N    = len(moves)
    advs = torch.tensor(
        [-evals[t] - evals[t - 1] for t in range(1, N + 1)],
        dtype=torch.float32,
    )

    # z-score so advantage is game-relative rather than position-absolute
    std = advs.std()
    if std > 0:
        advs = (advs - advs.mean()) / (std + 1e-8)

    # softmax → probability distribution; scale so mean weight = 1.0
    weights = F.softmax(advs / temperature, dim=0) * N

    # EOS token gets a neutral weight of 1.0
    return torch.cat([weights, torch.ones(1)], dim=0)  # [N + 1]
