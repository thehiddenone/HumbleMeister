from __future__ import annotations

import io
import json
import math
import multiprocessing
import os
import random
import select
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
import chess.engine
import chess.pgn
import py7zr
import time
import torch
import torch.nn.functional as F
import traceback

# ipywidgets is optional — only available inside a Jupyter environment.
# Import once at module load; assign None fallbacks so every reference is
# always bound. Methods check _WIDGETS_AVAILABLE before calling anything,
# and narrow bar/status via `is not None` so Pylance stays happy.
try:
    import ipywidgets as widgets
    from IPython.display import display as _display
except ImportError:
    widgets = None
    _display = None  # type: ignore[assignment]


_SHARD_SIZE = 1000  # games per shard file
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
            data: list[dict[str, Any]] = []
            try:
                data = torch.load(shard_path, weights_only=False)
                for item in data:
                    moves_uci: list[str] = item.get("moves", [])
                    n_moves = len(moves_uci)
                    try:
                        moves = [chess.Move.from_uci(u) for u in moves_uci]
                        board = chess.Board()
                        boards: list[chess.Board] = [board.copy()]
                        for move in moves:
                            board.push(move)
                            boards.append(board.copy())
                        evals = [_score_board(engine, b) for b in boards]
                        item["weights"] = _weights(evals, n_moves)
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
            data: list[dict[str, Any]] = []
            try:
                data = torch.load(shard_path, weights_only=False)
                for item in data:
                    if item.get("value_evals") is not None:
                        n_skipped += 1
                        continue
                    moves_uci: list[str] = item.get("moves", [])
                    try:
                        moves = [chess.Move.from_uci(u) for u in moves_uci]
                        board = chess.Board()
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


def _elo_passes(acc: list[str], elo_filter: int) -> bool:
    """
    Fast ELO check on raw PGN lines — no chess.pgn parsing needed.
    Scans only until both tags are found, then stops.
    Returns False if either tag is missing or below the threshold.
    """
    white_elo = black_elo = None
    for line in acc:
        if line.startswith('[WhiteElo "'):
            try:
                white_elo = int(line[11 : line.index('"', 11)])
            except (ValueError, IndexError):
                return False
        elif line.startswith('[BlackElo "'):
            try:
                black_elo = int(line[11 : line.index('"', 11)])
            except (ValueError, IndexError):
                return False
        if white_elo is not None and black_elo is not None:
            break
    if white_elo is None or black_elo is None:
        return False
    return white_elo >= elo_filter and black_elo >= elo_filter


def _read_chess_file(args: tuple[str, int | None, io.TextIOWrapper | None]) -> list[dict[str, Any]]:
    """
    Module-level worker function — one call per archive file.

    Extracts the archive, iterates lines, applies the ELO pre-filter on raw
    text (no PGN parsing for rejected games), then calls chess.pgn.read_game
    only for games that pass.  Returns a list of dicts with keys 'moves'
    (list of UCI strings) and 'outcome' (float).
    """
    file_path, elo_filter, log = args

    if log is not None:
        log.write(f'_read_chess_file {file_path}\n')

    content = []
    if file_path.endswith(".7z"):
        with py7zr.SevenZipFile(file_path, "r") as zf:
            names = zf.namelist()
            if len(names) != 1:
                return []
            with tempfile.TemporaryDirectory() as tmp:
                zf.extractall(path=tmp)
                with open(os.path.join(tmp, names[0]), "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read().splitlines()
    elif file_path.endswith(".zip"):
        with zipfile.ZipFile(file_path, "r") as zf:
            names = zf.namelist()
            if len(names) != 1:
                return []
            with zf.open(names[0]) as f:
                content = f.read().decode("utf-8", errors="ignore").splitlines()
    elif file_path.endswith(".pgn"):
        with open(file_path, "rt") as f:
            content = f.read().splitlines()

    results: list[dict[str, Any]] = []
    acc: list[str] = []

    def _flush(acc: list[str]) -> None:
        if not acc:
            return
        if elo_filter is not None and not _elo_passes(acc, elo_filter):
            return
        try:
            game = chess.pgn.read_game(io.StringIO("\n".join(acc)), Visitor=_SilentGameBuilder)
            if game is None:
                return
            result_str = game.headers.get("Result", "*")
            outcome = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}.get(result_str, 0.5)
            moves = list(game.mainline_moves())
            if moves:
                results.append({"moves": [m.uci() for m in moves], "outcome": outcome})
        except Exception:
            pass

    if log is not None:
        log.write(f'len(content) = {len(content)}\n')

    for line in content:
        if line.startswith('[Event "'):
            _flush(acc)
            acc = []
        acc.append(line)
    _flush(acc)

    if log is not None:
        log.write(f'len(results) = {len(results)}\n')

    return results


def _convert_file(
    args: tuple[str, int | None, str, str, int, float],
) -> tuple[str, int, int]:
    """Worker: read one PGN/7z/zip file, evaluate every game with Stockfish,
    and write shards directly to *out_dir*.

    Shard filenames are derived from the input file's stem so that
    concurrent workers never collide: ``{stem}_{shard:04d}.pt``.

    Returns ``(stem, n_games, n_shards_written)``.
    """
    file_path, elo_filter, out_dir, stockfish_path, depth, temperature = args

    with open(file_path + '.log', 'wt', buffering=1) as log:

        log.write(f'starting conversion of {file_path}\n')

        # derive a unique stem from the input filename (strip all archive extensions)
        stem = Path(file_path).name
        for ext in (".7z", ".zip", ".pgn"):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break

        # check if the file was converted already
        if os.path.isfile(os.path.join(out_dir, f"{stem}_0000.pt")):
            idx = 0
            n_shards = 0
            len_games = 0
            pt_path = os.path.join(out_dir, f"{stem}_{idx:04d}.pt")
            while os.path.isfile(pt_path):
                shard: list[dict[str, Any]] = torch.load(pt_path, weights_only=False)
                n_shards += 1
                len_games += len(shard)
                idx += 1
                pt_path = os.path.join(out_dir, f"{stem}_{idx:04d}.pt")
            log.write(f'alrady converted {file_path}: {stem}, {len_games}, {n_shards}\n')
            return (stem, len_games, n_shards)

        games = _read_chess_file((file_path, elo_filter, log))
        if not games:
            log.write(f'failure: no games\n')
            return (stem, 0, 0)

        log.write(f'read {len(games)} games\n')

        try:
            engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        except Exception:
            log.write(f'failure: failed to init stockfish\n')
            return (stem, 0, 0)

        t0 = time.time()
        conversion_time = 0.0
        eval_time = 0.0
        weighting_time = 0.0
        if engine is not None:
            try:
                for game_idx, item in enumerate(games):
                    moves_uci: list[str] = item["moves"]
                    n_moves = len(moves_uci)
                    try:
                        t_c = time.time()
                        moves = [chess.Move.from_uci(u) for u in moves_uci]
                        board = chess.Board()
                        boards: list[chess.Board] = [board.copy()]
                        for move in moves:
                            board.push(move)
                            boards.append(board.copy())
                        conversion_time += time.time() - t_c

                        t_e = time.time()
                        evals: list[float] = []
                        for b in boards:
                            info = engine.analyse(b, chess.engine.Limit(depth=depth))
                            pov = info.get("score")
                            if pov is None:
                                evals.append(0.0)
                                continue
                            rel = pov.relative
                            if rel.is_mate():
                                mate = rel.mate()
                                assert mate is not None
                                evals.append(float(_MATE_CAP_CP if mate > 0 else -_MATE_CAP_CP))
                            else:
                                cp = rel.score()
                                evals.append(float(cp) if cp is not None else 0.0)
                        eval_time += time.time() - t_e

                        t_w = time.time()
                        # advantage weights
                        if n_moves < 2:
                            item["weights"] = torch.ones(n_moves + 1)
                        else:
                            advs = torch.tensor(
                                [-evals[t] - evals[t - 1] for t in range(1, n_moves + 1)],
                                dtype=torch.float32,
                            )
                            std = advs.std(correction=0)
                            if std > 0:
                                advs = (advs - advs.mean()) / (std + 1e-8)
                            weights = F.softmax(advs / temperature, dim=0) * n_moves
                            item["weights"] = torch.cat([weights, torch.ones(1)], dim=0)

                        # value evals (tanh-normalised, White's perspective)
                        white_pov = [e if i % 2 == 0 else -e for i, e in enumerate(evals)]
                        item["value_evals"] = torch.tensor(
                            [math.tanh(v / 400.0) for v in white_pov],
                            dtype=torch.float32,
                        )
                        weighting_time += time.time() - t_w
                    except Exception:
                        log.write(f'exception was raised: {traceback.format_exc()}\n')
                        item["weights"] = torch.ones(n_moves + 1)
                        item["value_evals"] = None

                    game_count = game_idx + 1
                    if game_idx > 0 and game_count % 10 == 0:
                        dt = time.time() - t0
                        log.write(f'converted {game_idx + 1} games so far, elapsed time {dt}, {dt / game_count} per game\n')
                        log.write(f'conversion_time = {conversion_time} {conversion_time/game_count}\n')
                        log.write(f'eval_time = {eval_time} {eval_time/game_count}\n')
                        log.write(f'weighting_time = {weighting_time} {weighting_time/game_count}\n')
                        log.write('---\n')
            finally:
                engine.quit()
                log.write('closing the engine')

        # write shards to disk immediately — no data sent back to parent
        n_shards = 0
        for start in range(0, len(games), _SHARD_SIZE):
            shard = games[start : start + _SHARD_SIZE]
            torch.save(shard, os.path.join(out_dir, f"{stem}_{n_shards:04d}.pt"))
            n_shards += 1

    return (stem, len(games), n_shards)


class _SilentGameBuilder(chess.pgn.GameBuilder[chess.pgn.Game]):
    """GameBuilder that silently ignores parse errors such as result tokens
    ('1-0', '0-1', '1/2-1/2') appearing inside the move list."""

    def handle_error(self, error: Exception) -> None:
        pass


@dataclass
class _BankRecord:
    moves: list[chess.Move]
    outcome: float
    move_weights: torch.Tensor | None  # [n_moves + 1]; None until externally evaluated
    value_evals: torch.Tensor | None  # [n_moves + 1]; tanh-normalized, White's perspective


# ── Main class ─────────────────────────────────────────────────────────────


class ChessGameBank:

    def __init__(self, elo_filter: int | None = None) -> None:
        self.__records: list[_BankRecord] = []
        self.__cursor = 0
        self.__elo_filter = elo_filter

    # ------------------------------------------------------------------ #
    #  PGN loading                                                         #
    # ------------------------------------------------------------------ #

    def load_games(self, dir_path: str, n_workers: int = 8) -> None:
        """
        Load games from all PGN zip and 7z archives found under dir_path.

        Spawns up to n_workers processes (one per file) so multiple archives are
        parsed in parallel.  Within each worker, the ELO filter is applied on raw
        PGN text before chess.pgn.read_game is called, skipping rejected games
        entirely.  Progress updates as each file completes.
        """
        archive_files = [
            os.path.join(root, file)
            for root, _, files in os.walk(dir_path)
            for file in files
            if file.endswith(".zip") or file.endswith(".7z") or file.endswith(".pgn")
        ]

        total_files = len(archive_files)
        if total_files == 0:
            return

        bar = (
            widgets.IntProgress(min=0, max=total_files, description="Loading:")
            if widgets is not None
            else None
        )
        status = widgets.HTML(value="") if widgets is not None else None
        if bar is not None and status is not None:
            _display(widgets.HBox([bar, status]))  # type: ignore[no-untyped-call]

        args = [(path, self.__elo_filter, None) for path in archive_files]
        n_procs = min(n_workers, total_files)
        files_done = 0

        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(processes=n_procs) as pool:
            for game_dicts in pool.imap_unordered(_read_chess_file, args):
                for gd in game_dicts:
                    self.__records.append(
                        _BankRecord(
                            moves=[chess.Move.from_uci(uci) for uci in gd["moves"]],
                            outcome=gd["outcome"],
                            move_weights=None,
                            value_evals=None,
                        )
                    )
                files_done += 1
                if bar is not None and status is not None:
                    bar.value = files_done
                    status.value = (
                        f"<small>[{files_done}/{total_files}] "
                        f"{len(self.__records):,} games</small>"
                    )

        if bar is not None and status is not None:
            bar.bar_style = "success"
            status.value = f"<small>done &mdash; {len(self.__records):,} games</small>"

        random.shuffle(self.__records)

    # ------------------------------------------------------------------ #
    #  PGN → evaluated .pt conversion                                      #
    # ------------------------------------------------------------------ #

    def convert_games(
        self,
        source_path: str,
        output_path: str,
        n_workers: int = 8,
        stockfish_path: str = "stockfish",
        depth: int = 5,
        temperature: float = 1.0,
    ) -> None:
        """Read PGN files from *source_path*, evaluate every game with
        Stockfish, and write the results as .pt shards to *output_path*
        (compatible with :meth:`load`).

        Each worker processes one file end-to-end: reads the PGN, replays
        every game, evaluates positions with its own Stockfish instance,
        computes advantage weights + value evals, and writes shards
        directly to *output_path*.  The parent never holds game data —
        only lightweight ``(stem, n_games, n_shards)`` results come back
        through the pool.

        After all workers finish the parent renames the per-file shard
        files to sequential ``shard_XXXX.pt`` and writes ``meta.json``.

        Args:
            source_path:    directory containing .pgn / .zip / .7z files.
            output_path:    directory where shard .pt files will be written.
            n_workers:      max number of parallel worker processes.
            stockfish_path: path to the Stockfish binary.
            depth:          Stockfish search depth.
            temperature:    advantage-weight softmax temperature.
        """
        archive_files = [
            os.path.join(root, file)
            for root, _, files in os.walk(source_path)
            for file in files
            if file.endswith(".zip") or file.endswith(".7z") or file.endswith(".pgn")
        ]
        total_files = len(archive_files)
        if total_files == 0:
            print("no PGN files found")
            return

        p = Path(output_path)
        p.mkdir(parents=True, exist_ok=True)

        args = [
            (path, self.__elo_filter, str(p), stockfish_path, depth, temperature)
            for path in archive_files
        ]
        n_procs = min(n_workers, total_files)

        bar = (
            widgets.IntProgress(min=0, max=total_files, description="Converting:")
            if widgets is not None
            else None
        )
        status = widgets.HTML(value="") if widgets is not None else None
        if bar is not None and status is not None:
            _display(widgets.HBox([bar, status]))  # type: ignore[no-untyped-call]

        total_games = 0
        total_shards = 0
        files_done = 0

        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(processes=n_procs) as pool:
            for _stem, n_games, n_shards in pool.imap_unordered(_convert_file, args):
                total_games += n_games
                total_shards += n_shards
                files_done += 1
                if bar is not None and status is not None:
                    bar.value = files_done
                    status.value = (
                        f"<small>[{files_done}/{total_files}] "
                        f"{total_games:,} games, {total_shards} shards</small>"
                    )

        if total_shards == 0:
            print("no games converted")
            return

        # rename per-file shards to sequential shard_XXXX.pt
        tmp_shards = sorted(
            sp for sp in p.iterdir()
            if sp.suffix == ".pt" and sp.name != "meta.json"
        )
        for i, old in enumerate(tmp_shards):
            old.rename(p / f"shard_{i:04d}.pt")

        with open(p / "meta.json", "w") as f:
            json.dump(
                {
                    "n_games": total_games,
                    "n_shards": len(tmp_shards),
                    "shard_size": _SHARD_SIZE,
                },
                f,
            )

        if bar is not None and status is not None:
            bar.bar_style = "success"
            status.value = (
                f"<small>done &mdash; {total_games:,} games → "
                f"{len(tmp_shards)} shards in {p}/</small>"
            )

        print(
            f"converted {total_games:,} games to {len(tmp_shards)} shards in {p}/"
        )

    # ------------------------------------------------------------------ #
    #  Stockfish evaluation                                                #
    # ------------------------------------------------------------------ #

    def evaluate_moves(
        self,
        path: str,
        n_workers: int,
        stockfish_path: str = "stockfish",
        depth: int = 5,
        temperature: float = 1.0,
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
        n_shards = len(shard_paths)
        if n_shards == 0:
            raise ValueError(f"No shard files found in {path!r} — run save() first")

        # divide shards as evenly as possible across workers
        # worker i gets indices [starts[i], starts[i+1])
        actual_workers = min(n_workers, n_shards)
        base, extra = divmod(n_shards, actual_workers)
        groups: list[list[str]] = []
        start = 0
        for w in range(actual_workers):
            size = base + (1 if w < extra else 0)
            groups.append(shard_paths[start : start + size])
            start += size

        print(
            f"evaluating {n_shards} shards with {actual_workers} workers "
            f"(~{base}–{base + (1 if extra else 0)} shards/worker)"
        )

        bar = (
            widgets.IntProgress(min=0, max=n_shards, description="Evaluating:")
            if widgets is not None
            else None
        )
        status = (
            widgets.HTML(value=f"<small>0 / {n_shards} shards</small>")
            if widgets is not None
            else None
        )
        if bar is not None and status is not None:
            _display(widgets.HBox([bar, status]))  # type: ignore[no-untyped-call]

        # fork one child per group; keep the read end of each pipe
        read_fds: list[int] = []
        pids: list[int] = []
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
        buffers: dict[int, bytes] = {fd: b"" for fd in read_fds}
        open_fds: set[int] = set(read_fds)
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
                        ok_games += int(parts[1])
                        fail_games += int(parts[2])
                    elif parts[0] == "FAIL" and len(parts) == 2:
                        fail_games += int(parts[1])
                    done_shards += 1
                    if bar is not None and status is not None:
                        bar.value = done_shards
                        status.value = (
                            f"<small>{done_shards} / {n_shards} shards &mdash; "
                            f"ok={ok_games:,} fail={fail_games:,}</small>"
                        )

        # reap all child processes
        for pid in pids:
            os.waitpid(pid, 0)

        print(
            f"evaluation complete — ok={ok_games:,}  fail={fail_games:,}  "
            f"total={ok_games + fail_games:,}"
        )

        if bar is not None and status is not None:
            bar.bar_style = "success"
            status.value = f"<small>done &mdash; ok={ok_games:,}  fail={fail_games:,}</small>"

        # discard in-memory records and reload from the updated shard files
        self.__records = []
        self.__cursor = 0
        self.load(path)

    def fill_value_evals(
        self,
        path: str,
        n_workers: int,
        stockfish_path: str = "stockfish",
        depth: int = 5,
    ) -> None:
        """
        Add value_evals to shards that were saved before value evals existed.
        Games that already have value_evals are skipped; only missing ones are evaluated.
        When done, discards in-memory records and reloads from the updated shards.
        """
        p = Path(path)
        shard_paths = sorted(str(sp) for sp in p.glob("shard_*.pt"))
        n_shards = len(shard_paths)
        if n_shards == 0:
            raise ValueError(f"No shard files found in {path!r} — run save() first")

        actual_workers = min(n_workers, n_shards)
        base, extra = divmod(n_shards, actual_workers)
        groups: list[list[str]] = []
        start = 0
        for w in range(actual_workers):
            size = base + (1 if w < extra else 0)
            groups.append(shard_paths[start : start + size])
            start += size

        print(f"filling value_evals in {n_shards} shards with {actual_workers} workers")

        bar = (
            widgets.IntProgress(min=0, max=n_shards, description="Filling:")
            if widgets is not None
            else None
        )
        status = (
            widgets.HTML(value=f"<small>0 / {n_shards} shards</small>")
            if widgets is not None
            else None
        )
        if bar is not None and status is not None:
            _display(widgets.HBox([bar, status]))  # type: ignore[no-untyped-call]

        read_fds: list[int] = []
        pids: list[int] = []
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

        buffers: dict[int, bytes] = {fd: b"" for fd in read_fds}
        open_fds: set[int] = set(read_fds)
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
                        filled += int(parts[1])
                        skipped += int(parts[2])
                        failed += int(parts[3])
                    done_shards += 1
                    if bar is not None and status is not None:
                        bar.value = done_shards
                        status.value = (
                            f"<small>{done_shards} / {n_shards} shards &mdash; "
                            f"filled={filled:,} skipped={skipped:,} fail={failed:,}</small>"
                        )

        for pid in pids:
            os.waitpid(pid, 0)

        print(f"fill complete — filled={filled:,}  skipped={skipped:,}  fail={failed:,}")

        if bar is not None and status is not None:
            bar.bar_style = "success"
            status.value = f"<small>done &mdash; filled={filled:,}  skipped={skipped:,}  fail={failed:,}</small>"

        self.__records = []
        self.__cursor = 0
        self.load(path)

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save(self, path: str, shard_size: int | None = None) -> None:
        """Save the bank to a directory of shard files."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)

        if shard_size is None:
            shard_size = _SHARD_SIZE

        shards = [
            self.__records[i : i + shard_size] for i in range(0, len(self.__records), shard_size)
        ]

        with open(p / "meta.json", "w") as f:
            json.dump(
                {
                    "n_games": len(self.__records),
                    "n_shards": len(shards),
                    "shard_size": shard_size,
                },
                f,
            )

        bar = (
            widgets.IntProgress(min=0, max=max(len(shards), 1), description="Saving:")
            if widgets is not None
            else None
        )
        status = (
            widgets.HTML(value=f"<small>0 / {len(shards)} shards</small>")
            if widgets is not None
            else None
        )
        if bar is not None and status is not None:
            _display(widgets.HBox([bar, status]))  # type: ignore[no-untyped-call]

        for i, shard in enumerate(shards):
            data = [
                {
                    "moves": [m.uci() for m in r.moves],
                    "outcome": r.outcome,
                    "weights": r.move_weights,
                    "value_evals": r.value_evals,
                }
                for r in shard
            ]
            torch.save(data, p / f"shard_{i:04d}.pt")
            if bar is not None and status is not None:
                bar.value = i + 1
                status.value = f"<small>{i + 1} / {len(shards)} shards</small>"

        if bar is not None and status is not None:
            bar.bar_style = "success"
            status.value = (
                f"<small>done &mdash; {len(self.__records):,} games saved to {p}/</small>"
            )

    def load(self, path: str) -> None:
        """Load a previously saved bank from a shard directory."""
        p = Path(path)
        with open(p / "meta.json") as f:
            meta = json.load(f)
        n_shards: int = meta["n_shards"]

        bar = (
            widgets.IntProgress(min=0, max=max(n_shards, 1), description="Loading:")
            if widgets is not None
            else None
        )
        status = (
            widgets.HTML(value=f"<small>0 / {n_shards} shards</small>")
            if widgets is not None
            else None
        )
        if bar is not None and status is not None:
            _display(widgets.HBox([bar, status]))  # type: ignore[no-untyped-call]

        for i in range(n_shards):
            shard: list[dict[str, Any]] = torch.load(p / f"shard_{i:04d}.pt", weights_only=False)
            for item in shard:
                moves = [chess.Move.from_uci(uci) for uci in item["moves"]]
                self.__records.append(
                    _BankRecord(
                        moves=moves,
                        outcome=float(item["outcome"]),
                        move_weights=item.get("weights"),
                        value_evals=item.get("value_evals"),
                    )
                )
            if bar is not None and status is not None:
                bar.value = i + 1
                status.value = f"<small>{i + 1} / {n_shards} shards &mdash; {len(self.__records):,} games</small>"

        if bar is not None and status is not None:
            bar.bar_style = "success"
            status.value = f"<small>done &mdash; {len(self.__records):,} games loaded</small>"

        random.shuffle(self.__records)

    # ------------------------------------------------------------------ #
    #  Access                                                              #
    # ------------------------------------------------------------------ #

    def get_random_game(
        self,
    ) -> tuple[list[chess.Move], float, torch.Tensor | None, torch.Tensor | None]:
        """Returns (moves, outcome, move_weights, value_evals). Both tensors are None until externally evaluated."""
        if self.__cursor >= len(self.__records):
            random.shuffle(self.__records)
            self.__cursor = 0

        record = self.__records[self.__cursor]
        self.__cursor += 1
        return record.moves, record.outcome, record.move_weights, record.value_evals

    def __len__(self) -> int:
        return len(self.__records)
