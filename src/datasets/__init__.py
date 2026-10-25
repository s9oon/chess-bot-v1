"""Datasets we support, one file each. Every dataset returns the same two arrays:

    boards   (N, 98)  uint8    packed positions, from encoding.pack_board
    targets  (N,)     float32  what the model should output for each one, -1 .. +1

Prepare one with `make data DATASET=<file name>`.
"""
