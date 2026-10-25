# encoding.py

[`src/encoding.py`](../../src/encoding.py)

A board becomes 18 input feature maps of 8x8: 12 piece maps, whose move, 4
castling rights, en passant. These are what conv1 reads:
[many channels](../concepts/convolution.md#many-channels).

## Imports

```python
import numpy as np
import jax
import jax.numpy as jnp
import chess
```

NumPy for packing (CPU), JAX for unpacking (GPU), python-chess for the board.

## Constants

```python
# 0-11 pieces, 12 whose move, 13-16 castling rights, 17 en passant square.
NUM_PLANES = 18

# Packed layout, one uint8 row per position:
#   bytes 0-95  12 piece bitboards (8 bytes each), white P N B R Q K then black
#   byte  96    flags: bit 0 white to move, bits 1-4 castling (W king, W queen, B king, B queen)
#   byte  97    en passant square 0-63, or 64 if none
PACKED_SIZE = 98
NO_EP = 64
```

`NUM_PLANES` is also conv1's number of input feature maps in `model.py`.

## Packing, performed by the CPU

```python
def pack_board(board: chess.Board) -> np.ndarray:
    packed = np.empty(PACKED_SIZE, dtype=np.uint8)
```

**Does:** one `chess.Board` → one row of 98 bytes (`uint8`), filled in the
three parts below. No feature maps are built.

**Used:** by [lichess_evals.py](lichess_evals.md), once per position, when it
builds its cache.

### Pieces: bytes 0-95

```python
    masks = np.array([board.pieces_mask(piece_type, color)
                      for color in (chess.WHITE, chess.BLACK)
                      for piece_type in range(1, 7)], dtype="<u8")
    packed[:96] = masks.view(np.uint8)
```

python-chess stores each piece type as a 64-bit number, bit *n* = a piece on
square *n* (a1 = 0, h8 = 63). There are 12 of them (6 piece types x 2 colours),
one per piece map, in the same order as maps 0-11. `.view(np.uint8)` reads
each 64-bit number as its 8 bytes, without changing anything: 12 x 8 = 96
bytes.

### Turn and castling: byte 96

```python
    packed[96] = (board.turn == chess.WHITE) \
        | board.has_kingside_castling_rights(chess.WHITE) << 1 \
        | board.has_queenside_castling_rights(chess.WHITE) << 2 \
        | board.has_kingside_castling_rights(chess.BLACK) << 3 \
        | board.has_queenside_castling_rights(chess.BLACK) << 4
```

Five yes/no answers, one bit each. `<< 1` shifts a bit into position 1, `<< 2`
into position 2, and so on; `|` combines them into one byte.

### En passant: byte 97

```python
    packed[97] = NO_EP if board.ep_square is None else board.ep_square
    return packed
```

The square number 0-63, or 64 when there's no en passant square.

## Unpacking, performed by the GPU

```python
@jax.jit
def unpack_planes(packed: jax.Array) -> jax.Array:
    n = packed.shape[0]
```

**Does:** `(n, 98)` packed boards → `(n, 8, 8, 18)` float32 feature maps, for
any `n`: 1 board, 1024, or all 3M if they fit on the GPU. It doesn't choose `n`
or split anything up. `@jax.jit` compiles it to run on the GPU. Each part below
rebuilds some of the 18 maps, for every row at once.

**Used:** will be called by the training loop on each batch from
[batches.py](batches.md), right before the model. Nothing calls it yet: that
loop isn't written.

### Maps 0-11: pieces

```python
    pieces = jnp.unpackbits(packed[:, :96], axis=1, bitorder="little")
    pieces = pieces.reshape(n, 12, 8, 8).transpose(0, 2, 3, 1)
```

96 bytes → 768 bits (one per square per piece map) → 12 maps of 8x8. The
transpose moves the map number last, `(n, rank, file, map)`, the layout the
conv layers expect.

### Maps 12-16: turn and castling

```python
    flags = (packed[:, 96:97] >> jnp.arange(5, dtype=jnp.uint8)) & 1
    flags = jnp.broadcast_to(flags[:, None, None, :], (n, 8, 8, 5))
```

Pulls out bits 0-4 of byte 96, then copies each one onto all 64 squares: a
whole map of 1s or 0s.

### Map 17: en passant

```python
    ep = jax.nn.one_hot(packed[:, 97], 65)[:, :64].reshape(n, 8, 8, 1)
```

The square number → 65 slots with a single `1`. Dropping slot 64 ("none")
means no en passant gives an all-zero map. Reshape to 8x8.

### All 18 together

```python
    return jnp.concatenate([pieces, flags, ep], axis=-1).astype(jnp.float32)
```

12 + 5 + 1 = 18 maps, as floats: `(n, 8, 8, 18)`.
