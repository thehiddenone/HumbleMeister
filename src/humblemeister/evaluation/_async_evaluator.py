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


def _persistent_worker(
    stockfish_path: str,
    depth: int,
    temperature: float,
    job_r_fd: int,
    done_w_fd: int,
) -> None:
    """
    Runs in a forked child process for the lifetime of AsyncBatchEvaluator.
    Reads batch file paths from job_r_fd (one newline-terminated path per job),
    evaluates each batch in-place, then writes "OK\\n" to done_w_fd.
    Exits cleanly when job_r_fd is closed by the parent.
    """
    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    except Exception:
        os.close(job_r_fd)
        os.close(done_w_fd)
        os._exit(1)

    buf = b""
    try:
        while True:
            chunk = os.read(job_r_fd, 4096)
            if not chunk:
                break  # parent closed job pipe → shutdown
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                batch_path = line.decode().strip()
                if not batch_path:
                    continue
                # evaluate batch file in-place
                try:
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
                    pass
                os.write(done_w_fd, b"OK\n")
    finally:
        try:
            engine.quit()
        except Exception:
            pass
        os.close(job_r_fd)
        os.close(done_w_fd)

    os._exit(0)


class AsyncBatchEvaluator:
    """
    Pool of persistent Stockfish worker processes that evaluate game batches in parallel.

    Workers are forked once on construction and kept alive for the lifetime of the
    evaluator — Stockfish startup cost is paid only once per worker, not per batch.
    Job dispatch and completion signaling use pipes carrying short strings;
    batch data stays in temp files on disk.

    submit() sends a batch file path to an idle worker (blocking if all are busy).
    drain() waits for all in-flight workers to finish.
    close() shuts down all workers cleanly.
    """

    def __init__(
        self,
        stockfish_path: str,
        depth: int,
        temperature: float,
        n_workers: int = 24,
    ) -> None:
        # (pid, job_w_fd, done_r_fd) for each worker
        self._workers: list[tuple[int, int, int]] = []
        self._idle: list[int] = []  # indices into self._workers
        self._busy: dict[int, tuple[int, Path]] = {}  # done_r_fd → (worker_idx, batch_path)
        self._done_bufs: dict[int, bytes] = {}  # partial-read buffers keyed by done_r_fd

        for i in range(n_workers):
            job_r, job_w = os.pipe()
            done_r, done_w = os.pipe()
            pid = os.fork()
            if pid == 0:
                # child — close parent-side fds and run worker loop
                os.close(job_w)
                os.close(done_r)
                _persistent_worker(stockfish_path, depth, temperature, job_r, done_w)
                os._exit(0)
            else:
                # parent — close child-side fds, track worker
                os.close(job_r)
                os.close(done_w)
                self._workers.append((pid, job_w, done_r))
                self._idle.append(i)
                self._done_bufs[done_r] = b""

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def submit(self, batch_path: Path) -> list[Path]:
        """
        Submit a batch file for async Stockfish evaluation.
        If all workers are busy, blocks until one finishes.
        Returns paths of any batches that completed while waiting.
        """
        completed: list[Path] = []
        if not self._idle:
            completed.append(self._wait_one())

        worker_idx = self._idle.pop()
        _, job_w, done_r = self._workers[worker_idx]
        os.write(job_w, (str(batch_path) + "\n").encode())
        self._busy[done_r] = (worker_idx, batch_path)

        return completed

    def drain(self) -> list[Path]:
        """Block until all in-flight workers finish. Returns their batch paths."""
        completed: list[Path] = []
        while self._busy:
            completed.append(self._wait_one())
        return completed

    def close(self) -> None:
        """Drain pending work then shut down all worker processes."""
        self.drain()
        for pid, job_w, done_r in self._workers:
            try:
                os.close(job_w)  # EOF on worker's job_r → worker exits
            except OSError:
                pass
        for pid, job_w, done_r in self._workers:
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass
            try:
                os.close(done_r)
            except OSError:
                pass

    def __enter__(self) -> AsyncBatchEvaluator:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    #  Internals                                                           #
    # ------------------------------------------------------------------ #

    def _wait_one(self) -> Path:
        """Block until any one busy worker signals completion. Returns its batch path."""
        done_fds = list(self._busy.keys())
        readable, _, _ = select.select(done_fds, [], [])
        fd = readable[0]

        # read "OK\n" from the done pipe — buffer handles partial reads
        buf = self._done_bufs[fd]
        while b"\n" not in buf:
            buf += os.read(fd, 64)
        # consume exactly one response; keep any remainder for the next job
        self._done_bufs[fd] = buf[buf.index(b"\n") + 1 :]

        worker_idx, batch_path = self._busy.pop(fd)
        self._idle.append(worker_idx)
        return batch_path
