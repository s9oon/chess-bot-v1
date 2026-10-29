import numpy as np
import chess


def legal_priors(policy: np.ndarray, board: chess.Board) -> dict[chess.Move, float]:
    """One board's policy scores (4096,) -> {legal move: probability}, probabilities summing to 1."""
    # Underpromotions are left out: from-square x to-square can't tell them apart from the queen promotion.
    legal = [move for move in board.legal_moves if move.promotion in (None, chess.QUEEN)]

    # Each move's score sits at from-square x 64 + to-square.
    scores = policy[[move.from_square * 64 + move.to_square for move in legal]]

    # Softmax over the legal moves only, so illegal moves never get a probability.
    # Subtracting the max first stops exp() overflowing; it doesn't change the result.
    exps = np.exp(scores - scores.max())
    probs = exps / exps.sum()
    return dict(zip(legal, probs.tolist()))
