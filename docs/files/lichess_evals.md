# datasets/lichess_evals.py

[`src/datasets/lichess_evals.py`](../../src/datasets/lichess_evals.py): the
[Lichess evaluation database](https://database.lichess.org/#evals), positions
from Lichess games each evaluated by Stockfish. Used by the
[supervised](supervised.md) method.

```bash
make data DATASET=lichess_evals    # build the cache (slow once, then under a second)
```

## Imports

```python
import json
import time

import numpy as np
import chess

from config import CP_SCALE, DATA_DIR, EVALS_FILE, NUM_POSITIONS
from encoding import PACKED_SIZE, pack_board
```

`json` reads each line of the dataset, `chess` turns its FEN into a board, and
`pack_board` turns that into 98 bytes ([encoding.md](encoding.md)).

## The target

```python
def eval_target(position: dict) -> float:
    deepest = max(position["evals"], key=lambda e: e["depth"])
    best = deepest["pvs"][0]  # pvs[0] is Stockfish's best line

    if "mate" in best:
        return 1.0 if best["mate"] > 0 else -1.0  # forced mate: as good (or bad) as it gets
    return float(np.tanh(best["cp"] / CP_SCALE))  # cp = centipawns, 100 = one pawn up
```

**Does:** one parsed line of the file → one target score in -1 .. +1. Takes the
deepest evaluation, its best line (`pvs[0]`), and returns `tanh(cp / CP_SCALE)`,
or ±1 for a forced mate.

**Used:** by `build_positions`, once per line.

Each line looks like
`{"fen": "...", "evals": [{"pvs": [{"cp": 69, ...}], "depth": 46}, ...]}`, with
scores from white's point of view. A position can have evaluations at several
search depths; the deepest is the most accurate. `tanh` puts the target in the
same -1 .. +1 range as the model's output
([the head](../concepts/neural_network.md#the-head)). A forced mate has no
centipawn value, so it becomes ±1.

## Reading the file

```python
def build_positions(n: int) -> tuple[np.ndarray, np.ndarray]:
    boards = np.zeros((n, PACKED_SIZE), dtype=np.uint8)
    targets = np.zeros(n, dtype=np.float32)
```

**Does:** reads the first `n` lines of `EVALS_FILE` and returns
`boards (n, 98)` uint8 and `targets (n,)` float32. Slow: about 2.5 minutes for
3M positions.

**Used:** by `load_positions`, only when there's no cache yet.

It reserves all the RAM up front: 98 bytes and one target per position.

```python
    start = time.perf_counter()
    filled = 0
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
```

Reads one line at a time, so the 3.9 GB file is never all in RAM. Each line
fills one row. Prints progress every 100k positions.

```python
    return boards[:filled], targets[:filled]
```

If the file ran out before `n`, only the filled rows are returned.

## The cache

```python
def load_positions(n: int = NUM_POSITIONS) -> tuple[np.ndarray, np.ndarray]:
    cache = DATA_DIR / f"packed_{n}.npz"
    if cache.exists():
        data = np.load(cache)
        return data["boards"], data["targets"]

    print(f"No cache, packing {n:,} positions from {EVALS_FILE.name}")
    boards, targets = build_positions(n)
    np.savez(cache, boards=boards, targets=targets)
    return boards, targets
```

**Does:** returns `boards (n, 98)` and `targets (n,)`. If `data/packed_<n>.npz`
exists it loads that (under a second); if not, it calls `build_positions` and
saves the result there first.

**Used:** by `make data DATASET=lichess_evals`, and will be by the training loop
to get its data. 3M packed positions are 294 MB, so they all fit in RAM.

## Running it

```python
if __name__ == "__main__":
    start = time.perf_counter()
    boards, targets = load_positions()
    print(f"boards {boards.shape} {boards.dtype}, targets {targets.shape} {targets.dtype}")
    print(f"{boards.nbytes / 1e6:.0f} MB in RAM, loaded in {time.perf_counter() - start:.1f}s")
```

**Does:** calls `load_positions`, then prints the shapes, RAM used and time
taken.

**Used:** `make data DATASET=lichess_evals` runs it.

## Settings

`EVALS_FILE`, `NUM_POSITIONS` and `CP_SCALE` in `src/config.py`.
