from __future__ import annotations

import chess
import chess.engine
import chess.pgn
import io
import json
import math
import os
import py7zr
import random
import select
import tempfile
import zipfile

import torch
import torch.nn.functional as F

from dataclasses import dataclass
from pathlib import Path

# ipywidgets is optional — only available inside a Jupyter environment.
# Import once at module load; assign None fallbacks so every reference is
# always bound. Methods check _WIDGETS_AVAILABLE before calling anything,
# and narrow bar/status via `is not None` so Pylance stays happy.
try:
    import ipywidgets as widgets
    from IPython.display import display as _display
    _WIDGETS_AVAILABLE = True
except ImportError:
    widgets  = None  # type: ignore[assignment]
    _display = None  # type: ignore[assignment]
    _WIDGETS_AVAILABLE = False


_SHARD_SIZE  = 1000  # games per shard file
_MATE_CAP_CP = 2000  # centipawn cap for mate scores


# ── Pipe message protocol ──────────────────────────────────────────────────
# Child sends one newline-terminated line per shard:
#   "OK <n_ok> <n_fail>\n"   — shard written successfully
#   "FAIL <n_fail>\n"        — shard-level failure (could not load / save shard)


def _evaluate_group(
    shard_paths: list[str],
    stockfish_path: str,
    depth: int,
    temperature: float,
    write_fd: int,
) -> None:
    """
    Called inside a forked child process.  Opens one Stockfish instance, evaluates
    every game in each assigned shard, overwrites the shard file with weights, and
    reports a status line per shard back to the parent via write_fd.
    Must call os._exit() when done — never raises.
    """
    def _score_board(engine: chess.engine.SimpleEngine, board: chess.Board) -> float:
        info  = engine.analyse(board, chess.engine.Limit(depth=depth))
        pov   = info.get("score")
        if pov is None:
            return 0.0
        rel = pov.relative
        if rel.is_mate():
            mate = rel.mate()
            assert mate is not None
            return float(_MATE_CAP_CP if mate > 0 else -_MATE_CAP_CP)
        cp = rel.score()
        return float(cp) if cp is not None else 0.0

    def _value_evals(evals: list[float]) -> torch.Tensor:
        # evals[t] is centipawns from the side-to-move's perspective at position t.
        # Convert to White's perspective: even positions are White to move (keep sign),
        # odd positions are Black to move (flip sign).  Then squash to [-1, 1] with tanh.
        white_pov = [e if i % 2 == 0 else -e for i, e in enumerate(evals)]
        return torch.tensor(
            [math.tanh(v / 400.0) for v in white_pov],
            dtype=torch.float32,
        )

    def _weights(evals: list[float], n_moves: int) -> torch.Tensor:
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

    def _report(msg: str) -> None:
        os.write(write_fd, msg.encode())

    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    except Exception as exc:
        # can't even open Stockfish — fail all shards
        for sp in shard_paths:
            _report(f"FAIL 0\n")
        os.close(write_fd)
        os._exit(1)

    try:
        for shard_path in shard_paths:
            n_ok = n_fail = 0
            data = list[dict]()
            try:
                data: list[dict] = torch.load(shard_path, weights_only=False)
                for item in data:
                    moves_uci: list[str] = item.get("moves", [])
                    n_moves = len(moves_uci)
                    try:
                        moves  = [chess.Move.from_uci(u) for u in moves_uci]
                        board  = chess.Board()
                        boards: list[chess.Board] = [board.copy()]
                        for move in moves:
                            board.push(move)
                            boards.append(board.copy())
                        evals = [_score_board(engine, b) for b in boards]
                        item["weights"]     = _weights(evals, n_moves)
                        item["value_evals"] = _value_evals(evals)
                        n_ok += 1
                    except Exception:
                        item["weights"] = torch.ones(n_moves + 1)
                        n_fail += 1
                torch.save(data, shard_path)
                _report(f"OK {n_ok} {n_fail}\n")
            except Exception:
                _report(f"FAIL {len(data) if 'data' in dir() else 0}\n")
    finally:
        engine.quit()
        os.close(write_fd)

    os._exit(0)


def _fill_value_evals_group(
    shard_paths: list[str],
    stockfish_path: str,
    depth: int,
    write_fd: int,
) -> None:
    """
    Called inside a forked child process.  Like _evaluate_group but only fills in
    missing value_evals — games that already have value_evals are left untouched.
    Reports one status line per shard: "OK <n_filled> <n_skipped> <n_fail>\n"
    Must call os._exit() when done — never raises.
    """
    def _score_board(engine: chess.engine.SimpleEngine, board: chess.Board) -> float:
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        pov  = info.get("score")
        if pov is None:
            return 0.0
        rel = pov.relative
        if rel.is_mate():
            mate = rel.mate()
            assert mate is not None
            return float(_MATE_CAP_CP if mate > 0 else -_MATE_CAP_CP)
        cp = rel.score()
        return float(cp) if cp is not None else 0.0

    def _report(msg: str) -> None:
        os.write(write_fd, msg.encode())

    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    except Exception:
        for _ in shard_paths:
            _report("FAIL 0\n")
        os.close(write_fd)
        os._exit(1)

    try:
        for shard_path in shard_paths:
            n_filled = n_skipped = n_fail = 0
            data = list[dict]()
            try:
                data: list[dict] = torch.load(shard_path, weights_only=False)
                for item in data:
                    if item.get("value_evals") is not None:
                        n_skipped += 1
                        continue
                    moves_uci: list[str] = item.get("moves", [])
                    try:
                        moves  = [chess.Move.from_uci(u) for u in moves_uci]
                        board  = chess.Board()
                        boards: list[chess.Board] = [board.copy()]
                        for move in moves:
                            board.push(move)
                            boards.append(board.copy())
                        evals = [_score_board(engine, b) for b in boards]
                        white_pov = [e if i % 2 == 0 else -e for i, e in enumerate(evals)]
                        item["value_evals"] = torch.tensor(
                            [math.tanh(v / 400.0) for v in white_pov],
                            dtype=torch.float32,
                        )
                        n_filled += 1
                    except Exception:
                        n_fail += 1
                torch.save(data, shard_path)
                _report(f"OK {n_filled} {n_skipped} {n_fail}\n")
            except Exception:
                _report(f"FAIL 0\n")
    finally:
        engine.quit()
        os.close(write_fd)

    os._exit(0)


# ── Helpers ────────────────────────────────────────────────────────────────

class _SilentGameBuilder(chess.pgn.GameBuilder):
    """GameBuilder that silently ignores parse errors such as result tokens
    ('1-0', '0-1', '1/2-1/2') appearing inside the move list."""
    def handle_error(self, error: Exception) -> None:
        pass


@dataclass
class _BankRecord:
    moves:        list[chess.Move]
    outcome:      float
    move_weights: torch.Tensor | None  # [n_moves + 1]; None until externally evaluated
    value_evals:  torch.Tensor | None  # [n_moves + 1]; tanh-normalized, White's perspective


# ── Main class ─────────────────────────────────────────────────────────────

class ChessGameBank:

    def __init__(self, elo_filter: int | None = None) -> None:
        self.__records: list[_BankRecord] = []
        self.__cursor = 0
        self.__elo_filter = elo_filter

    # ------------------------------------------------------------------ #
    #  PGN loading                                                         #
    # ------------------------------------------------------------------ #

    def __read_pgn_archive(self, file_path: str) -> list[chess.pgn.Game]:
        if file_path.endswith('.7z'):
            with py7zr.SevenZipFile(file_path, 'r') as zf:
                names = zf.namelist()
                if len(names) != 1:
                    raise ValueError(f"Expected exactly 1 file in 7z, found {len(names)}: {names}")
                with tempfile.TemporaryDirectory() as tmp:
                    zf.extractall(path=tmp)
                    extracted = os.path.join(tmp, names[0])
                    with open(extracted, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read().splitlines()
        else:
            with zipfile.ZipFile(file_path, 'r') as zf:
                names = zf.namelist()
                if len(names) != 1:
                    raise ValueError(f"Expected exactly 1 file in ZIP, found {len(names)}: {names}")
                with zf.open(names[0]) as f:
                    content = f.read().decode('utf-8', errors='ignore').splitlines()

        total_lines = len(content)
        bar    = widgets.IntProgress(min=0, max=max(total_lines, 1), description='Parsing:') if _WIDGETS_AVAILABLE else None  # type: ignore[union-attr]
        status = widgets.HTML(value=f'<small>0 / {total_lines:,} lines</small>')             if _WIDGETS_AVAILABLE else None  # type: ignore[union-attr]
        if bar is not None and status is not None:
            _display(widgets.HBox([bar, status]))  # type: ignore[union-attr]

        result: list[chess.pgn.Game] = []
        acc:    list[str]            = []
        for n, line in enumerate(content, start=1):
            if line.startswith('[Event "'):
                if acc:
                    game = chess.pgn.read_game(io.StringIO('\n'.join(acc)), Visitor=_SilentGameBuilder)
                    if game:
                        result.append(game)
                    acc = []
            acc.append(line)
            if bar is not None and status is not None and n % 10 == 0:
                bar.value    = n
                status.value = f'<small>{n:,} / {total_lines:,} lines &mdash; {len(result):,} games</small>'

        if bar is not None and status is not None:
            bar.bar_style = 'success'
            status.value  = f'<small>{total_lines:,} lines &mdash; {len(result):,} games</small>'

        return result

    def load_games(self, dir_path: str) -> None:
        """Load games from all PGN zip and 7z archives found under dir_path."""
        archive_files = [
            os.path.join(root, file)
            for root, _, files in os.walk(dir_path)
            for file in files
            if file.endswith('.zip') or file.endswith('.7z')
        ]

        total_files = len(archive_files)
        bar    = widgets.IntProgress(min=0, max=max(total_files, 1), description='Loading:') if _WIDGETS_AVAILABLE else None  # type: ignore[union-attr]
        status = widgets.HTML(value='')                                                        if _WIDGETS_AVAILABLE else None  # type: ignore[union-attr]
        if bar is not None and status is not None:
            _display(widgets.HBox([bar, status]))  # type: ignore[union-attr]

        for n, path in enumerate(archive_files, start=1):
            if bar is not None and status is not None:
                bar.value    = n
                status.value = f'<small>[{n}/{total_files}] {os.path.basename(path)}</small>'
            for game in self.__read_pgn_archive(path):
                if self.__elo_filter is not None:
                    try:
                        white_elo = int(game.headers.get("WhiteElo", ""))
                        black_elo = int(game.headers.get("BlackElo", ""))
                    except (ValueError, TypeError):
                        continue
                    if white_elo < self.__elo_filter or black_elo < self.__elo_filter:
                        continue

                result  = game.headers.get("Result", "*")
                outcome = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}.get(result, 0.5)
                self.__records.append(_BankRecord(
                    moves        = list(game.mainline_moves()),
                    outcome      = outcome,
                    move_weights = None,
                    value_evals  = None,
                ))

        if bar is not None and status is not None:
            bar.bar_style = 'success'
            status.value  = f'<small>loaded &mdash; {len(self.__records):,} games</small>'

        random.shuffle(self.__records)

    # ------------------------------------------------------------------ #
    #  Stockfish evaluation                                                #
    # ------------------------------------------------------------------ #

    def evaluate_moves(
        self,
        path:           str,
        n_workers:      int,
        stockfish_path: str   = "stockfish",
        depth:          int   = 5,
        temperature:    float = 1.0,
    ) -> None:
        """
        Evaluate every shard in the saved bank at `path` with Stockfish, writing
        per-move advantage weights back into the shard files.  When done, discards
        in-memory records and reloads from the updated shards.

        Forks n_workers child processes, each receiving a slice of the shard list.
        Each child opens its own Stockfish, processes its shards, and reports one
        status line per shard back to the parent via a pipe.  The parent multiplexes
        all pipe read-ends with select() and updates the ipywidgets progress bar.
        """
        p = Path(path)
        shard_paths = sorted(str(sp) for sp in p.glob("shard_*.pt"))
        n_shards    = len(shard_paths)
        if n_shards == 0:
            raise ValueError(f"No shard files found in {path!r} — run save() first")

        # divide shards as evenly as possible across workers
        # worker i gets indices [starts[i], starts[i+1])
        actual_workers = min(n_workers, n_shards)
        base, extra    = divmod(n_shards, actual_workers)
        groups: list[list[str]] = []
        start = 0
        for w in range(actual_workers):
            size = base + (1 if w < extra else 0)
            groups.append(shard_paths[start : start + size])
            start += size

        print(f"evaluating {n_shards} shards with {actual_workers} workers "
              f"(~{base}–{base + (1 if extra else 0)} shards/worker)")

        bar    = widgets.IntProgress(min=0, max=n_shards, description='Evaluating:') if _WIDGETS_AVAILABLE else None  # type: ignore[union-attr]
        status = widgets.HTML(value=f'<small>0 / {n_shards} shards</small>')          if _WIDGETS_AVAILABLE else None  # type: ignore[union-attr]
        if bar is not None and status is not None:
            _display(widgets.HBox([bar, status]))  # type: ignore[union-attr]

        # fork one child per group; keep the read end of each pipe
        read_fds: list[int] = []
        pids:     list[int] = []
        for group in groups:
            r_fd, w_fd = os.pipe()
            pid = os.fork()
            if pid == 0:
                # ── child ───────────────────────────────────────────────
                os.close(r_fd)
                _evaluate_group(group, stockfish_path, depth, temperature, w_fd)
                os._exit(0)  # _evaluate_group calls _exit itself, but be safe
            else:
                # ── parent ──────────────────────────────────────────────
                os.close(w_fd)
                read_fds.append(r_fd)
                pids.append(pid)

        # gather reports from all children via select()
        buffers:    dict[int, bytes] = {fd: b"" for fd in read_fds}
        open_fds:   set[int]         = set(read_fds)
        done_shards = ok_games = fail_games = 0

        while open_fds:
            readable, _, _ = select.select(list(open_fds), [], [])
            for fd in readable:
                chunk = os.read(fd, 4096)
                if not chunk:
                    # EOF — child closed its write end
                    os.close(fd)
                    open_fds.discard(fd)
                    continue
                buffers[fd] += chunk
                # process all complete lines in the buffer
                while b"\n" in buffers[fd]:
                    line, buffers[fd] = buffers[fd].split(b"\n", 1)
                    parts = line.decode().split()
                    if not parts:
                        continue
                    if parts[0] == "OK" and len(parts) == 3:
                        ok_games   += int(parts[1])
                        fail_games += int(parts[2])
                    elif parts[0] == "FAIL" and len(parts) == 2:
                        fail_games += int(parts[1])
                    done_shards += 1
                    if bar is not None and status is not None:
                        bar.value    = done_shards
                        status.value = (f'<small>{done_shards} / {n_shards} shards &mdash; '
                                        f'ok={ok_games:,} fail={fail_games:,}</small>')

        # reap all child processes
        for pid in pids:
            os.waitpid(pid, 0)

        print(f"evaluation complete — ok={ok_games:,}  fail={fail_games:,}  "
              f"total={ok_games + fail_games:,}")

        if bar is not None and status is not None:
            bar.bar_style = 'success'
            status.value  = f'<small>done &mdash; ok={ok_games:,}  fail={fail_games:,}</small>'

        # discard in-memory records and reload from the updated shard files
        self.__records = []
        self.__cursor  = 0
        self.load(path)

    def fill_value_evals(
        self,
        path:           str,
        n_workers:      int,
        stockfish_path: str = "stockfish",
        depth:          int = 5,
    ) -> None:
        """
        Add value_evals to shards that were saved before value evals existed.
        Games that already have value_evals are skipped; only missing ones are evaluated.
        When done, discards in-memory records and reloads from the updated shards.
        """
        p = Path(path)
        shard_paths = sorted(str(sp) for sp in p.glob("shard_*.pt"))
        n_shards    = len(shard_paths)
        if n_shards == 0:
            raise ValueError(f"No shard files found in {path!r} — run save() first")

        actual_workers = min(n_workers, n_shards)
        base, extra    = divmod(n_shards, actual_workers)
        groups: list[list[str]] = []
        start = 0
        for w in range(actual_workers):
            size = base + (1 if w < extra else 0)
            groups.append(shard_paths[start : start + size])
            start += size

        print(f"filling value_evals in {n_shards} shards with {actual_workers} workers")

        bar    = widgets.IntProgress(min=0, max=n_shards, description='Filling:') if _WIDGETS_AVAILABLE else None  # type: ignore[union-attr]
        status = widgets.HTML(value=f'<small>0 / {n_shards} shards</small>')       if _WIDGETS_AVAILABLE else None  # type: ignore[union-attr]
        if bar is not None and status is not None:
            _display(widgets.HBox([bar, status]))  # type: ignore[union-attr]

        read_fds: list[int] = []
        pids:     list[int] = []
        for group in groups:
            r_fd, w_fd = os.pipe()
            pid = os.fork()
            if pid == 0:
                os.close(r_fd)
                _fill_value_evals_group(group, stockfish_path, depth, w_fd)
                os._exit(0)
            else:
                os.close(w_fd)
                read_fds.append(r_fd)
                pids.append(pid)

        buffers:    dict[int, bytes] = {fd: b"" for fd in read_fds}
        open_fds:   set[int]         = set(read_fds)
        done_shards = filled = skipped = failed = 0

        while open_fds:
            readable, _, _ = select.select(list(open_fds), [], [])
            for fd in readable:
                chunk = os.read(fd, 4096)
                if not chunk:
                    os.close(fd)
                    open_fds.discard(fd)
                    continue
                buffers[fd] += chunk
                while b"\n" in buffers[fd]:
                    line, buffers[fd] = buffers[fd].split(b"\n", 1)
                    parts = line.decode().split()
                    if not parts:
                        continue
                    if parts[0] == "OK" and len(parts) == 4:
                        filled  += int(parts[1])
                        skipped += int(parts[2])
                        failed  += int(parts[3])
                    done_shards += 1
                    if bar is not None and status is not None:
                        bar.value    = done_shards
                        status.value = (f'<small>{done_shards} / {n_shards} shards &mdash; '
                                        f'filled={filled:,} skipped={skipped:,} fail={failed:,}</small>')

        for pid in pids:
            os.waitpid(pid, 0)

        print(f"fill complete — filled={filled:,}  skipped={skipped:,}  fail={failed:,}")

        if bar is not None and status is not None:
            bar.bar_style = 'success'
            status.value  = f'<small>done &mdash; filled={filled:,}  skipped={skipped:,}  fail={failed:,}</small>'

        self.__records = []
        self.__cursor  = 0
        self.load(path)

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        """Save the bank to a directory of shard files."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)

        shards = [
            self.__records[i : i + _SHARD_SIZE]
            for i in range(0, len(self.__records), _SHARD_SIZE)
        ]

        with open(p / "meta.json", "w") as f:
            json.dump({"n_games": len(self.__records), "n_shards": len(shards), "shard_size": _SHARD_SIZE}, f)

        bar    = widgets.IntProgress(min=0, max=max(len(shards), 1), description='Saving:') if _WIDGETS_AVAILABLE else None  # type: ignore[union-attr]
        status = widgets.HTML(value=f'<small>0 / {len(shards)} shards</small>')             if _WIDGETS_AVAILABLE else None  # type: ignore[union-attr]
        if bar is not None and status is not None:
            _display(widgets.HBox([bar, status]))  # type: ignore[union-attr]

        for i, shard in enumerate(shards):
            data = [
                {
                    "moves":       [m.uci() for m in r.moves],
                    "outcome":     r.outcome,
                    "weights":     r.move_weights,
                    "value_evals": r.value_evals,
                }
                for r in shard
            ]
            torch.save(data, p / f"shard_{i:04d}.pt")
            if bar is not None and status is not None:
                bar.value    = i + 1
                status.value = f'<small>{i + 1} / {len(shards)} shards</small>'

        if bar is not None and status is not None:
            bar.bar_style = 'success'
            status.value  = f'<small>done &mdash; {len(self.__records):,} games saved to {p}/</small>'

    def load(self, path: str) -> None:
        """Load a previously saved bank from a shard directory."""
        p = Path(path)
        with open(p / "meta.json") as f:
            meta = json.load(f)
        n_shards: int = meta["n_shards"]

        bar    = widgets.IntProgress(min=0, max=max(n_shards, 1), description='Loading:') if _WIDGETS_AVAILABLE else None  # type: ignore[union-attr]
        status = widgets.HTML(value=f'<small>0 / {n_shards} shards</small>')              if _WIDGETS_AVAILABLE else None  # type: ignore[union-attr]
        if bar is not None and status is not None:
            _display(widgets.HBox([bar, status]))  # type: ignore[union-attr]

        for i in range(n_shards):
            shard: list[dict] = torch.load(p / f"shard_{i:04d}.pt", weights_only=False)
            for item in shard:
                moves = [chess.Move.from_uci(uci) for uci in item["moves"]]
                self.__records.append(_BankRecord(
                    moves        = moves,
                    outcome      = float(item["outcome"]),
                    move_weights = item.get("weights"),
                    value_evals  = item.get("value_evals"),
                ))
            if bar is not None and status is not None:
                bar.value    = i + 1
                status.value = f'<small>{i + 1} / {n_shards} shards &mdash; {len(self.__records):,} games</small>'

        if bar is not None and status is not None:
            bar.bar_style = 'success'
            status.value  = f'<small>done &mdash; {len(self.__records):,} games loaded</small>'

        random.shuffle(self.__records)

    # ------------------------------------------------------------------ #
    #  Access                                                              #
    # ------------------------------------------------------------------ #

    def get_random_game(self) -> tuple[list[chess.Move], float, torch.Tensor | None, torch.Tensor | None]:
        """Returns (moves, outcome, move_weights, value_evals). Both tensors are None until externally evaluated."""
        if self.__cursor >= len(self.__records):
            random.shuffle(self.__records)
            self.__cursor = 0

        record = self.__records[self.__cursor]
        self.__cursor += 1
        return record.moves, record.outcome, record.move_weights, record.value_evals

    def __len__(self) -> int:
        return len(self.__records)
