# data.py

[`src/data.py`](../../src/data.py): reads the Lichess file, packs every
position with [`pack_board`](encoding.md), works out each target, and caches
the result.

```
evals_10M.jsonl ──build_positions (slow, once)──> data/packed_<N>.npz
packed_<N>.npz  ──np.load (under a second)──────> boards (N, 98), targets (N,) in RAM
                ──batches──> one batch at a time ──> GPU
```

The cache is the work saved to disk: encoding 3M positions takes about 2.5
minutes, loading the cache about 0.3 seconds. 3M packed positions are 294 MB,
so they all fit in RAM.

## `eval_target(position) -> float`

One Lichess line → the target score. Takes the **deepest** evaluation, its best
line (`pvs[0]`), and turns the centipawns into `tanh(cp / CP_SCALE)`; a forced
mate becomes ±1. Why: [loss function](../concepts/loss_function.md#targets-must-match-our-range).

## `build_positions(n) -> (boards, targets)`

Reads the first `n` lines, one at a time, so the 3.9 GB file is never all in
RAM. Fills arrays reserved up front: `boards (n, 98)` uint8 and
`targets (n,)` float32. Prints progress every 100k positions.

## `load_positions(n=NUM_POSITIONS) -> (boards, targets)`

Loads `data/packed_<n>.npz` if it exists. If not, calls `build_positions` and
saves the result there for next time.

## `batches(boards, targets, batch_size, rng)`

One epoch of shuffled batches, still packed. Training sends each batch to the
GPU and unpacks it with [`unpack_planes`](encoding.md#unpack_planespacked---jaxarray).

## Running it

`make data` (or `python src/data.py`) builds the cache if needed, then prints
the shapes and how much RAM they use.

## Data source

The [Lichess evaluation database](https://database.lichess.org/#evals): positions
from Lichess games, each evaluated by Stockfish, as one JSON object per line.
Scores are from white's point of view.
