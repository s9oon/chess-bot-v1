import numpy as np
import chess

NUM_MOVES = 64 * 64  # policy head outputs: from-square x to-square


def move_index(move: chess.Move) -> int:
    """Where a move sits in the policy's 4096 outputs: from-square x 64 + to-square. e2e4 -> 12 * 64 + 28 = 796."""
    return move.from_square * 64 + move.to_square


def legal_priors(policy: np.ndarray, board: chess.Board) -> dict[chess.Move, float]:
    """One board's policy scores (4096,) -> {legal move: probability}, probabilities summing to 1."""
    # Underpromotions are left out: from-square x to-square can't tell them apart from the queen promotion.
    legal = [move for move in board.legal_moves if move.promotion in (None, chess.QUEEN)]

    scores = policy[[move_index(move) for move in legal]]

    # Softmax over the legal moves only, so illegal moves never get a probability.
    # Subtracting the max first stops exp() overflowing; it doesn't change the result.
    exps = np.exp(scores - scores.max())
    probs = exps / exps.sum()
    return dict(zip(legal, probs.tolist()))


def policy_target(shares: dict[chess.Move, float]) -> np.ndarray:
    """The reverse of legal_priors: {move: probability} -> (4096,), laid out like the policy head, 0 elsewhere."""
    target = np.zeros(NUM_MOVES, dtype=np.float32)
    target[[move_index(move) for move in shares]] = list(shares.values())
    return target
