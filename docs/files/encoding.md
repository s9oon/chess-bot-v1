# encoding.py

[`src/encoding.py`](../../src/encoding.py)

A board becomes 18 input feature maps of 8x8: 12 piece maps, whose move, 4
castling rights, en passant. These are what conv1 reads:
[many channels](../concepts/convolution.md#many-channels).

## Packing, on the CPU

```python
masks = np.array([board.pieces_mask(piece_type, color)
                  for color in (chess.WHITE, chess.BLACK)
                  for piece_type in range(1, 7)], dtype="<u8")
packed[:96] = masks.view(np.uint8)
```

python-chess stores each piece type as a 64-bit number, bit *n* = a piece on
square *n* (a1 = 0). The 12 of them are the 12 piece maps, copied as 96 bytes.
Byte 96 holds turn + castling as bits, byte 97 the en passant square: 98 bytes
per board.

## Unpacking, on the GPU

```python
pieces = jnp.unpackbits(packed[:, :96], axis=1, bitorder="little")
pieces = pieces.reshape(batch, 12, 8, 8).transpose(0, 2, 3, 1)
```

A whole batch at once: bits → 12 maps of 8x8 → `(batch, 8, 8, 12)`. The flag
bits each fill a whole map, and en passant becomes a single `1`, for 18 maps.
