"""Dataset: the Lichess evaluation database (https://database.lichess.org/#evals).

Reads only that format: one JSON object per line, {"fen": ..., "evals": [{"pvs": [{"cp" or "mate"}], "depth"}]}.
Targets are Stockfish's score (white's view) as tanh(cp / CP_SCALE), or +-1 for a forced mate.
"""
import json
import time

import numpy as np
import chess

from chessbot.config import CP_SCALE, DATA_DIR, EVALS_FILE, NUM_POSITIONS
from chessbot.encoding.planes import PACKED_SIZE, pack_board


def eval_target(position: dict) -> float:
    """Stockfish's score for a position (white's view), squashed into -1..1 to match the model's tanh."""
    # A line can have several evals at different search depths: the deepest is the most accurate.
    deepest = max(position["evals"], key=lambda e: e["depth"])
    best = deepest["pvs"][0]  # pvs[0] is Stockfish's best line

    if "mate" in best:
        return 1.0 if best["mate"] > 0 else -1.0  # forced mate: as good (or bad) as it gets
    return float(np.tanh(best["cp"] / CP_SCALE))  # cp = centipawns, 100 = one pawn up


def build_positions(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Read the first n lines of the evals file and pack them. Slow: only runs when there's no cache."""
    # Reserve all the RAM up front, then fill one row per line.
    boards = np.zeros((n, PACKED_SIZE), dtype=np.uint8)
    targets = np.zeros(n, dtype=np.float32)

    start = time.perf_counter()
    filled = 0
    # Iterating over the file reads one line at a time, so the whole file is never in RAM.
    with open(EVALS_FILE) as f:
        for line in f:
            if filled == n:
                break
            position = json.loads(line)
            boards[filled] = pack_board(chess.Board(position["fen"]))
            targets[filled] = eval_target(position)
            filled += 1
            if filled % 100_000 == 0:
                print(f"  {filled:,} / {n:,}  ({time.perf_counter() - start:.0f}s)")

    # If the file ran out early, keep only the rows that were filled.
    return boards[:filled], targets[:filled]


def load_positions(n: int = NUM_POSITIONS) -> tuple[np.ndarray, np.ndarray]:
    """Packed boards (n, 98) uint8 and targets (n,) float32, from the cache if it exists."""
    cache = DATA_DIR / f"packed_{n}.npz"
    if cache.exists():
        data = np.load(cache)
        return data["boards"], data["targets"]

    print(f"No cache, packing {n:,} positions from {EVALS_FILE.name}")
    boards, targets = build_positions(n)
    # Saved for next run, so this slow step only ever happens once.
    np.savez(cache, boards=boards, targets=targets)
    return boards, targets


if __name__ == "__main__":
    start = time.perf_counter()
    boards, targets = load_positions()
    print(f"boards {boards.shape} {boards.dtype}, targets {targets.shape} {targets.dtype}")
    print(f"{boards.nbytes / 1e6:.0f} MB in RAM, loaded in {time.perf_counter() - start:.1f}s")
