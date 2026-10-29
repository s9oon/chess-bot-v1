"""Board -> numbers, on the CPU. Turning packed boards into planes is done by whatever runs the network
(model/inputs.py for JAX)."""
import numpy as np
import chess

# 0-11 pieces, 12 whose move, 13-16 castling rights, 17 en passant square.
NUM_PLANES = 18

# Packed layout, one uint8 row per position:
#   bytes 0-95  12 piece bitboards (8 bytes each), white P N B R Q K then black
#   byte  96    flags: bit 0 white to move, bits 1-4 castling (W king, W queen, B king, B queen)
#   byte  97    en passant square 0-63, or 64 if none
PACKED_SIZE = 98
NO_EP = 64


def pack_board(board: chess.Board) -> np.ndarray:
    """Turn a board into 98 bytes on the CPU. No planes are built here."""
    packed = np.empty(PACKED_SIZE, dtype=np.uint8)

    # pieces_mask is a 64-bit int: bit n = 1 means that piece is on square n (a1 = 0, h8 = 63).
    # "<u8" = little-endian uint64, so byte 0 holds squares 0-7.
    masks = np.array([board.pieces_mask(piece_type, color)
                      for color in (chess.WHITE, chess.BLACK)
                      for piece_type in range(1, 7)], dtype="<u8")
    packed[:96] = masks.view(np.uint8)

    packed[96] = (board.turn == chess.WHITE) \
        | board.has_kingside_castling_rights(chess.WHITE) << 1 \
        | board.has_queenside_castling_rights(chess.WHITE) << 2 \
        | board.has_kingside_castling_rights(chess.BLACK) << 3 \
        | board.has_queenside_castling_rights(chess.BLACK) << 4

    packed[97] = NO_EP if board.ep_square is None else board.ep_square
    return packed
