"""MCTS (PUCT, as in AlphaZero): search the game tree, guided by the network's policy and value.

Knows nothing about JAX or the model: it only calls `evaluate(board) -> (priors, value)`,
where value is from the side to move's view, -1 .. +1. Whatever runs the network provides it:
training/self_play.py for the JAX model, an ONNX one later.
"""
import math
from typing import Callable

import numpy as np
import chess

from chessbot.config import C_PUCT, DIRICHLET_ALPHA, NOISE_FRACTION, NUM_SIMULATIONS

Evaluate = Callable[[chess.Board], tuple[dict[chess.Move, float], float]]


class Node:
    """One position in the tree, reached by one move from its parent."""

    def __init__(self, prior: float):
        self.prior = prior    # the policy's probability for the move into this node
        self.visits = 0       # how many simulations went through it
        self.value_sum = 0.0  # their values, from the view of the player who made the move into it
        self.children: dict[chess.Move, Node] = {}  # empty until expanded

    def q(self) -> float:
        """Average value so far: how good the move into this node has been for the player who made it."""
        return self.value_sum / self.visits if self.visits else 0.0


def search(board: chess.Board, evaluate: Evaluate, num_simulations: int = NUM_SIMULATIONS,
           rng: np.random.Generator | None = None) -> tuple[dict[chess.Move, int], float]:
    """Run the simulations from `board`: {legal move: visit count}, and the root's value (side to move's view).

    Pass `rng` to add exploration noise to the root's priors, for self-play.
    """
    # The first simulation always just expands the root, so do it here: noise needs the root's children.
    root = Node(prior=1.0)
    backup([root], expand(root, board, evaluate))
    if rng is not None and root.children:
        add_noise(root, rng)

    for _ in range(num_simulations - 1):
        # 1. Select: walk down through expanded nodes, pushing each move onto the board.
        node, path = root, [root]
        while node.children:
            move, node = select_child(node)
            board.push(move)
            path.append(node)

        # 2. Expand: the network scores the new position and gives its moves' priors.
        value = expand(node, board, evaluate)

        # 3. Backup: pass the value up the path, then undo the moves.
        backup(path, value)
        for _ in range(len(path) - 1):
            board.pop()

    visits = {move: child.visits for move, child in root.children.items()}
    return visits, -root.q()  # root.q() is the view of whoever moved into the root: flip it


def add_noise(node: Node, rng: np.random.Generator) -> None:
    """Mix random Dirichlet noise into the children's priors, so self-play tries moves the policy would overlook."""
    noise = rng.dirichlet([DIRICHLET_ALPHA] * len(node.children))
    for child, n in zip(node.children.values(), noise):
        child.prior = (1 - NOISE_FRACTION) * child.prior + NOISE_FRACTION * n


def select_child(node: Node) -> tuple[chess.Move, Node]:
    """The child with the best PUCT score: its average value so far, plus a bonus for a high prior and few visits."""
    sqrt_visits = math.sqrt(node.visits)

    def puct(child: Node) -> float:
        return child.q() + C_PUCT * child.prior * sqrt_visits / (1 + child.visits)

    return max(node.children.items(), key=lambda item: puct(item[1]))


def expand(node: Node, board: chess.Board, evaluate: Evaluate) -> float:
    """Give the node one child per legal move; return the position's value, side to move's view."""
    # A finished game needs no network: checkmate means the side to move lost, anything else is a draw.
    if board.is_game_over():
        return -1.0 if board.is_checkmate() else 0.0

    priors, value = evaluate(board)
    node.children = {move: Node(prior) for move, prior in priors.items()}
    return value


def backup(path: list[Node], value: float) -> None:
    """Add the leaf's value to every node on the path, flipping its sign each ply: good for one side is bad for the other."""
    # `value` is from the leaf's side to move; the leaf node itself stores the view of whoever moved into it.
    for node in reversed(path):
        value = -value
        node.visits += 1
        node.value_sum += value
