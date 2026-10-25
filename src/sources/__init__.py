"""Training data sources. Each file reads one kind of data and returns the same two arrays:

    boards   (N, 98)  uint8    packed positions, from encoding.pack_board
    targets  (N,)     float32  what the model should output for each one, -1 .. +1

Training only sees those two arrays, so any source works with it.
"""
