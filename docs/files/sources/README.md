# sources/

[`src/sources/`](../../../src/sources/): where training data comes from. Each
file reads **one** kind of data and returns the same two arrays:

```
boards   (N, 98)  uint8     packed positions, from pack_board
targets  (N,)     float32   what the model should output for each one, -1 .. +1
```

Training only ever sees those two arrays, so any source works with it. Adding a
new kind of data means adding a new file here, not changing training.

| File | Reads | Target | Model |
|---|---|---|---|
| [lichess_evals.py](lichess_evals.md) | Lichess evaluation database (JSONL) | Stockfish's score | distilled |
| [common.py](common.md) | (shared by all) | | |

Planned: games played on the site (target: the game's result), and self-play.
