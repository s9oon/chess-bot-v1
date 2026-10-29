# Board encoding

A network only takes numbers. The board becomes 18 **input feature maps**
(planes) of 8x8, the input conv1 reads: `(8, 8, 18)`.

## Planes

Each plane is a yes/no question asked of every square, 1 for yes, 0 for no:

| Planes | Question |
|---|---|
| 0–11 | is there a white pawn / knight / bishop / rook / queen / king, then black's, on this square? |
| 12 | is it white to move? (all 1s or all 0s) |
| 13–16 | can white castle kingside / queenside, black kingside / queenside? (all 1s or 0s) |
| 17 | is this the en passant square? |

The piece planes are **one-hot**: a square has a 1 in at most one of planes
0–11. The board's layout is kept, 8x8, so a conv kernel sees neighbouring
squares together ([convolution](Convolution.md)). Rules that aren't about one
square, like whose turn it is, fill a whole plane so every patch sees them.

From [`encoding/planes.py`](../../src/chessbot/encoding/planes.py):

```python
# 0-11 pieces, 12 whose move, 13-16 castling rights, 17 en passant square.
NUM_PLANES = 18
```

## Packing: storing boards small

18 x 64 float32s is 4.6 KB per board; 3M boards would be 14 GB. Almost all of
it is 0s and 1s, so we store each board **packed** into 98 bytes, and only
build the planes on the GPU, one batch at a time.

A **bitboard** is a 64-bit number with one bit per square: bit *n* is 1 when
square *n* (a1 = 0 .. h8 = 63) has that piece. python-chess already stores
pieces this way, so packing just copies 12 of them, 8 bytes each:

```python
    masks = np.array([board.pieces_mask(piece_type, color)
                      for color in (chess.WHITE, chess.BLACK)
                      for piece_type in range(1, 7)], dtype="<u8")
    packed[:96] = masks.view(np.uint8)
```

Turn and castling are 5 more bits in byte 96, and the en passant square
(0–63, or 64 for none) is byte 97.

## Unpacking: bits back to planes

`unpack_planes` reverses it for a whole batch on the GPU, `(n, 98)` →
`(n, 8, 8, 18)`:

```python
    pieces = jnp.unpackbits(packed[:, :96], axis=1, bitorder="little")
    pieces = pieces.reshape(n, 12, 8, 8).transpose(0, 2, 3, 1)
```

96 bytes → 768 bits, one per square per piece plane. Square *n* is rank
*n* // 8, file *n* % 8, so reshaping to 8x8 puts each bit on its square. The
transpose moves the plane number last, `(n, rank, file, plane)`, the layout
`nnx.Conv` expects.

## Further reading

- Silver et al. (2017), [*Mastering Chess and Shogi by Self-Play with a
  General Reinforcement Learning Algorithm*][alphazero], Table S1: AlphaZero's
  input planes for chess. Same idea as ours, plus the last 8 positions and
  repetition counts.
- [Chess Programming Wiki: Bitboards][bitboards].

[alphazero]: https://arxiv.org/abs/1712.01815
[bitboards]: https://www.chessprogramming.org/Bitboards
