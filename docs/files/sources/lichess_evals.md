# sources/lichess_evals.py

[`src/sources/lichess_evals.py`](../../../src/sources/lichess_evals.py): the
distilled model's data. Reads the Lichess file, packs every position with
[`pack_board`](../encoding.md), works out each target, and caches the result.

**Reads one format only:** the [Lichess evaluation
database](https://database.lichess.org/#evals), one JSON object per line:

```json
{"fen": "...", "evals": [{"pvs": [{"cp": 69, "line": "..."}, ...], "depth": 46}, ...]}
```

Each position comes from a Lichess game and is evaluated by Stockfish, possibly
at several depths. Scores (`cp` in centipawns, or `mate` in N) are from white's
point of view.

```
evals_10M.jsonl ──build_positions (slow, once)──> data/packed_<N>.npz
packed_<N>.npz  ──np.load (under a second)──────> boards (N, 98), targets (N,) in RAM
```

The cache is the work saved to disk: encoding 3M positions takes about 2.5
minutes, loading the cache about 0.3 seconds. 3M packed positions are 294 MB,
so they all fit in RAM.

## `eval_target(position) -> float`

One Lichess line → the target score. Takes the **deepest** evaluation, its best
line (`pvs[0]`), and turns the centipawns into `tanh(cp / CP_SCALE)`; a forced
mate becomes ±1. That puts the targets in the same -1 .. +1 range as the
model's tanh output.

## `build_positions(n) -> (boards, targets)`

Reads the first `n` lines, one at a time, so the 3.9 GB file is never all in
RAM. Fills arrays reserved up front: `boards (n, 98)` uint8 and
`targets (n,)` float32. Prints progress every 100k positions.

## `load_positions(n=NUM_POSITIONS) -> (boards, targets)`

Loads `data/packed_<n>.npz` if it exists. If not, calls `build_positions` and
saves the result there for next time.

## Running it

`make data` builds the cache if needed, then prints the shapes and how much RAM
they use.

## Settings

`EVALS_FILE`, `NUM_POSITIONS` and `CP_SCALE` in [`config.py`](../config.md).
