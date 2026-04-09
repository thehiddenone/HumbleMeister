from __future__ import annotations

import os
import select
from pathlib import Path
from typing import Any

import chess
import chess.engine
import torch
import torch.nn.functional as F

_MATE_CAP_CP = 2000


def _score_board(engine: chess.engine.SimpleEngine, board: chess.Board, depth: int) -> float:
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    pov = info.get("score")
    if pov is None:
        return 0.0
    rel = pov.relative
    if rel.is_mate():
        mate = rel.mate()
        assert mate is not None
        return float(_MATE_CAP_CP if mate > 0 else -_MATE_CAP_CP)
    cp = rel.score()
    return float(cp) if cp is not None else 0.0


def _compute_weights(evals: list[float], n_moves: int, temperature: float) -> torch.Tensor:
    if n_moves < 2:
        return torch.ones(n_moves + 1)
    advs = torch.tensor(
        [-evals[t] - evals[t - 1] for t in range(1, n_moves + 1)],
        dtype=torch.float32,
    )
    std = advs.std(correction=0)
    if std > 0:
        advs = (advs - advs.mean()) / (std + 1e-8)
    weights = F.softmax(advs / temperature, dim=0) * n_moves
    return torch.cat([weights, torch.ones(1)], dim=0)


def _evaluate_batch_worker(
    batch_path: str,
    stockfish_path: str,
    depth: int,
    temperature: float,
    write_fd: int,
) -> None:
    """
    Runs inside a forked child process.
    Loads the batch from batch_path, evaluates every game with Stockfish,
    writes per-move weights back in-place, then closes write_fd to signal
    completion to the parent.  Always calls os._exit() — never returns.
    """
    engine = None
    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        data: list[dict[str, Any]] = torch.load(batch_path, weights_only=False)
        for item in data:
            moves_uci: list[str] = item.get("moves", [])
            n_moves = len(moves_uci)
            try:
                moves = [chess.Move.from_uci(u) for u in moves_uci]
                board = chess.Board()
                boards = [board.copy()]
                for move in moves:
                    board.push(move)
                    boards.append(board.copy())
                evals = [_score_board(engine, b, depth) for b in boards]
                item["weights"] = _compute_weights(evals, n_moves, temperature)
            except Exception:
                item["weights"] = torch.ones(n_moves + 1)
        torch.save(data, batch_path)
    except Exception:
        pass  # parent will load whatever is in the file (partial or no weights)
    finally:
        if engine is not None:
            try:
                engine.quit()
            except Exception:
                pass
        os.close(write_fd)

    os._exit(0)


class AsyncBatchEvaluator:
    """
    Pool of forked Stockfish workers that evaluate game batches in parallel.

    Each submitted batch is a temp file in the format list[dict] with keys
    "moves" (list[str] UCI), "outcome" (float), and optionally "weights".
    The child process evaluates all games, writes weights back in-place, and
    signals completion by closing its pipe write-end.  The parent detects
    completion via select() on the read-end.

    The pool is bounded to n_workers concurrent forks.  submit() blocks if
    all slots are taken, waiting for one to finish before forking a new one.
    """

    def __init__(
        self,
        stockfish_path: str,
        depth: int,
        temperature: float,
        n_workers: int,
    ) -> None:
        self._stockfish_path = stockfish_path
        self._depth = depth
        self._temperature = temperature
        self._n_workers = n_workers
        self._active: list[tuple[int, int, Path]] = []  # (pid, read_fd, batch_path)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def submit(self, batch_path: Path) -> list[Path]:
        """
        Submit a batch file for async Stockfish evaluation.
        If all n_workers slots are busy, blocks until one finishes.
        Returns paths of any batches that completed while waiting.
        """
        completed: list[Path] = []
        if len(self._active) >= self._n_workers:
            completed.append(self._wait_one())

        r_fd, w_fd = os.pipe()
        pid = os.fork()
        if pid == 0:
            # child — evaluate and exit
            os.close(r_fd)
            _evaluate_batch_worker(
                str(batch_path),
                self._stockfish_path,
                self._depth,
                self._temperature,
                w_fd,
            )
            os._exit(0)  # _evaluate_batch_worker calls os._exit, but be safe
        else:
            # parent — track the new worker
            os.close(w_fd)
            self._active.append((pid, r_fd, batch_path))

        return completed

    def drain(self) -> list[Path]:
        """Block until all in-flight workers finish. Returns their batch paths."""
        completed: list[Path] = []
        while self._active:
            completed.append(self._wait_one())
        return completed

    def close(self) -> None:
        self.drain()

    def __enter__(self) -> AsyncBatchEvaluator:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    #  Internals                                                           #
    # ------------------------------------------------------------------ #

    def _wait_one(self) -> Path:
        """Block until any one worker closes its pipe. Returns its batch path."""
        read_fds = [fd for _, fd, _ in self._active]
        readable, _, _ = select.select(read_fds, [], [])
        return self._reap(readable[0])

    def _reap(self, fd: int) -> Path:
        """Drain pipe to EOF, reap the child process, remove from active list."""
        while os.read(fd, 4096):
            pass
        os.close(fd)

        for i, (pid, rfd, path) in enumerate(self._active):
            if rfd == fd:
                os.waitpid(pid, 0)
                self._active.pop(i)
                return path

        raise RuntimeError(f"no active entry for fd {fd}")
