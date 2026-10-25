# encoding.py

[`src/encoding.py`](../../src/encoding.py): turns a board into numbers the
network can read. How the board is encoded and why:
[concepts/encoding.md](../concepts/encoding.md).

Two halves, one on each side of the CPU/GPU split:

```
CPU:  chess.Board ──pack_board──> 98 bytes            (once per position, cached by data.py)
GPU:  (batch, 98) ──unpack_planes──> (batch, 8, 8, 18) (every batch, right before the model)
```

## `pack_board(board) -> np.ndarray`

One board → `(98,)` uint8 on the CPU. Copies python-chess's 12 bitboards, the
flags byte (turn + castling) and the en passant square. No maps are built here.

## `unpack_planes(packed) -> jax.Array`

`(batch, 98)` uint8 → `(batch, 8, 8, 18)` float32, compiled with `@jax.jit` to
run on the GPU:

1. bytes 0-95 → bits → the 12 piece maps,
2. each flag bit → a whole map of that value (maps 12-16),
3. byte 97 → a one-hot map with a single `1` on the en passant square (map 17).

Passing a NumPy array in copies it to the GPU first; the output stays there.

## Constants

| Name | Value | What |
|---|---|---|
| `NUM_PLANES` | 18 | input feature maps per board |
| `PACKED_SIZE` | 98 | bytes per packed board |
| `NO_EP` | 64 | the en passant byte when there's no en passant square |
