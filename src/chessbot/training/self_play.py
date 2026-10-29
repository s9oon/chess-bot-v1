"""Self-play: learn only from games against itself, with MCTS choosing the moves (AlphaZero).

Each round: play games with search -> store (board, search's visit shares, game result) -> train on them.
"""
import time

import numpy as np
import jax
import jax.numpy as jnp
import optax
import chess
from flax import nnx

from chessbot.config import (BATCH_SIZE, BUFFER_SIZE, GAMES_PER_ROUND, LEARNING_RATE, MAX_GAME_PLIES,
                             NUM_ROUNDS, STEPS_PER_ROUND, TEMPERATURE_PLIES, WEIGHTS_DIR)
from chessbot.encoding.moves import NUM_MOVES, legal_priors, policy_target
from chessbot.encoding.planes import PACKED_SIZE, pack_board, unpack_planes
from chessbot.mcts.search import search
from chessbot.model.checkpoint import load_model, save_model
from chessbot.model.network import ChessCNN

WEIGHTS_FILE = WEIGHTS_DIR / "self_play.npz"


# --- Playing: the JAX model -> search ---

@nnx.jit
def forward(model: ChessCNN, packed):
    """Packed boards (n, 98) -> (policy, value), compiled to run on the GPU."""
    return model(unpack_planes(packed))


def make_evaluate(model: ChessCNN):
    """The `evaluate(board) -> (priors, value)` search needs, running `model` on one board at a time."""
    def evaluate(board: chess.Board) -> tuple[dict[chess.Move, float], float]:
        policy, value = forward(model, jnp.asarray(pack_board(board)[None]))  # a batch of one
        priors = legal_priors(np.asarray(policy[0]), board)
        # The value head scores from white's view; search wants the side to move's view.
        white_view = float(value[0])
        return priors, white_view if board.turn == chess.WHITE else -white_view

    return evaluate


def play_game(evaluate, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, float]:
    """One game against itself: packed boards (n, 98), search's visit shares pi (n, 4096), result (white's view)."""
    board = chess.Board()
    boards, pis = [], []

    while not board.is_game_over() and board.ply() < MAX_GAME_PLIES:
        visits, _ = search(board, evaluate, rng=rng)  # rng: noise on the root, so games differ
        moves = list(visits)
        shares = np.array([visits[move] for move in moves], dtype=np.float32)
        shares /= shares.sum()

        # pi: search's visit shares laid out like the policy head, the training target for it.
        boards.append(pack_board(board))
        pis.append(policy_target(dict(zip(moves, shares))))

        # Early on, pick in proportion to visits so openings vary; after that, play the best move.
        if board.ply() < TEMPERATURE_PLIES:
            move = moves[rng.choice(len(moves), p=shares)]
        else:
            move = moves[int(shares.argmax())]
        board.push(move)

    # "1-0" white won, "0-1" black won; a draw, or a game stopped at MAX_GAME_PLIES ("*"), is 0.
    result = {"1-0": 1.0, "0-1": -1.0}.get(board.result(), 0.0)
    return np.array(boards), np.array(pis), result


class Buffer:
    """The most recent positions to train on. Once full, new positions overwrite the oldest."""

    def __init__(self, size: int):
        self.boards = np.zeros((size, PACKED_SIZE), dtype=np.uint8)
        self.pis = np.zeros((size, NUM_MOVES), dtype=np.float32)
        self.zs = np.zeros(size, dtype=np.float32)  # the game's result, white's view, for every position in it
        self.size, self.next, self.filled = size, 0, 0

    def add(self, boards: np.ndarray, pis: np.ndarray, result: float) -> None:
        idx = (self.next + np.arange(len(boards))) % self.size  # wraps round to the start
        self.boards[idx], self.pis[idx], self.zs[idx] = boards, pis, result
        self.next = (self.next + len(boards)) % self.size
        self.filled = min(self.filled + len(boards), self.size)

    def sample(self, n: int, rng: np.random.Generator):
        idx = rng.integers(0, self.filled, n)
        return self.boards[idx], self.pis[idx], self.zs[idx]


# --- Learning: backpropagation on what search found ---

def loss_fn(model, boards, pis, zs):
    logits, values = model(unpack_planes(boards))
    # Value: predict who won (white's view, like the value head). Mean squared error.
    value_loss = jnp.mean((zs - values) ** 2)
    # Policy: predict search's visit shares. Cross-entropy: -sum(pi * log p), averaged over the batch.
    policy_loss = -jnp.mean(jnp.sum(pis * jax.nn.log_softmax(logits), axis=-1))
    return value_loss + policy_loss, (value_loss, policy_loss)


@nnx.jit
def train_step(model: ChessCNN, optimizer: nnx.Optimizer, boards, pis, zs):
    # Backpropagation through both heads and the shared body at once.
    (loss, parts), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, boards, pis, zs)
    optimizer.update(model, grads)
    return parts


def train() -> None:
    rng = np.random.default_rng()
    # Carry on from the last run if there is one.
    model = load_model(WEIGHTS_FILE) if WEIGHTS_FILE.exists() else ChessCNN(rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(LEARNING_RATE), wrt=nnx.Param)
    evaluate = make_evaluate(model)  # always the same model object, so games use the latest weights
    buffer = Buffer(BUFFER_SIZE)

    for round_num in range(1, NUM_ROUNDS + 1):
        start = time.perf_counter()

        # 1. Play. Batchnorm uses its running averages, so each board's score stands alone.
        model.eval()
        results, plies = [], 0
        for _ in range(GAMES_PER_ROUND):
            boards, pis, result = play_game(evaluate, rng)
            buffer.add(boards, pis, result)
            results.append(result)
            plies += len(boards)

        # 2. Learn from random positions out of the recent games.
        model.train()
        for _ in range(STEPS_PER_ROUND):
            value_loss, policy_loss = train_step(model, optimizer, *buffer.sample(BATCH_SIZE, rng))

        save_model(model, WEIGHTS_FILE)
        print(f"round {round_num}: {results.count(1.0)}W {results.count(0.0)}D {results.count(-1.0)}L, "
              f"{plies / GAMES_PER_ROUND:.0f} plies/game, buffer {buffer.filled:,}, "
              f"value loss {float(value_loss):.3f}, policy loss {float(policy_loss):.3f}, "
              f"{time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    train()
