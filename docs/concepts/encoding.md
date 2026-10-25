# Board encoding

A network only understands numbers, so a chess position has to become an array.
Ours becomes **18 input feature maps** of 8x8: shape `(8, 8, 18)`.

## 18 input feature maps

The board is split into **18 grids of 8x8**: 12 for the pieces and 6 for the
rest of the game state.

| Maps | What |
|---|---|
| 0-11 | pieces |
| 12 | whose move |
| 13-16 | castling rights |
| 17 | en passant |

In the convolution paper's terms ([Dumoulin & Visin, 2016][conv-paper]) these are
the network's **input feature maps**, also called **input channels**. The
paper's figures usually draw a single feature map; we have 18 stacked together.

### Piece maps (0-11)

One per piece type and colour. Each grid has a `1` where that piece is and `0`
everywhere else.

![Each map marks where one piece type is; the maps are stacked into one array](images/encoding.png)

Left to right: the printed maps for white pawns (map 0) and black pawns
(map 6), what those `1`s mean on the board, and all the maps stacked into one
array. The network gets the whole stack at once.

| Map | Piece | Map | Piece |
|---|---|---|---|
| 0 | white pawn | 6 | black pawn |
| 1 | white knight | 7 | black knight |
| 2 | white bishop | 8 | black bishop |
| 3 | white rook | 9 | black rook |
| 4 | white queen | 10 | black queen |
| 5 | white king | 11 | black king |

### Game state maps (12-17)

The pieces alone don't tell the whole story: the same position can be winning
or losing depending on these.

| Map | Meaning |
|---|---|
| 12 | whose move: **all 1s** if white to move, **all 0s** if black |
| 13 | white can castle kingside: all 1s if yes |
| 14 | white can castle queenside: all 1s if yes |
| 15 | black can castle kingside: all 1s if yes |
| 16 | black can castle queenside: all 1s if yes |
| 17 | en passant: a single `1` on the square a pawn could capture onto, if any |

Maps 12-16 are a yes/no for the whole board, so the whole map is filled with the
same value. That way every kernel sees it, wherever it sits on the board.

## The shape: `(8, 8, 18)`

```
planes[row, col, map]
       │    │    └─ which map (0-17)
       │    └────── file: 0 = a ... 7 = h
       └─────────── rank: 0 = rank 1 ... 7 = rank 8
```

For example, `planes[1, 4, 0]` is the white-pawn map on e2: `1.0` in the
starting position.

The network takes boards in batches, which adds a dimension at the front:
`(batch, 8, 8, 18)`.

## Why one map per piece?

A single grid with pawn = 1, knight = 2 ... would make the network treat a
knight as "two pawns". Separate maps keep each piece type independent. This is
called **one-hot encoding**: each square has at most one `1` across the 12
piece maps, and an empty square is all `0`s.

## Packing: 98 bytes instead of 18 maps

Stored as float32, one board's 18 maps are 8 x 8 x 18 x 4 = 4,608 bytes, so 10M
positions would be about 46 GB. Most of that is waste:

- every value is a 0 or a 1, so it needs **1 bit**, not 32;
- maps 12-16 hold **1 bit of information** each, copied onto all 64 squares;
- map 17 marks **at most one square**.

So each position is stored as the information alone, **98 bytes**:

| Bytes | What |
|---|---|
| 0-95 | 12 **bitboards**, 8 bytes each: one 64-bit number per piece map, bit *n* = 1 means a piece on square *n* |
| 96 | flags: bit 0 white to move, bits 1-4 the four castling rights |
| 97 | en passant square 0-63, or 64 for none |

A bitboard is just a piece map written as one 64-bit number instead of 64
separate values. Squares are numbered a1 = 0, b1 = 1 ... h8 = 63, so bit *n*
is rank *n* // 8, file *n* % 8.

The CPU only copies these numbers out of python-chess, which already stores
the board as bitboards. The GPU then **unpacks** a whole batch back into
`(batch, 8, 8, 18)` maps right before the network sees it. The network gets
exactly the same 18 maps either way.

## Further reading

- **Input feature maps:** [Dumoulin & Visin (2016)][conv-paper], section 1.1
  ("Discrete convolutions"), where the input is a stack of feature maps.
- **Bitboards:** the [Chess Programming Wiki][bitboards] page on bitboards. For
  the a1 = 0 square numbering we use, see its
  [square mapping page][square-mapping] ("Little-Endian Rank-File Mapping").
- **How AlphaZero encodes a chess board:** Silver et al. (2017),
  [*Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning
  Algorithm*][alphazero], the "Representation" part of the Methods section,
  and its input features table. Same idea as ours (one map per piece type, plus
  whole maps for castling and side to move), with extra maps for past positions.

[conv-paper]: https://arxiv.org/abs/1603.07285
[bitboards]: https://www.chessprogramming.org/Bitboards
[square-mapping]: https://www.chessprogramming.org/Square_Mapping_Considerations
[alphazero]: https://arxiv.org/abs/1712.01815
