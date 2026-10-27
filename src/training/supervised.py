"""Supervised: learn to predict a dataset's target scores, e.g. Stockfish's evaluations."""
import time

import numpy as np
import jax.numpy as jnp
import optax
from flax import nnx

from config import LEARNING_RATE
from model import ChessCNN


def loss_fn(model, boards, targets):
    _, scores = model(boards)  # only the value head: the dataset has no move targets
    return jnp.mean((scores - targets) ** 2)


def create_optimizer(model: ChessCNN) -> nnx.Optimizer:
    """Adam: the most popular version of gradient descent. Only the weights (nnx.Param) get nudged."""
    return nnx.Optimizer(model, optax.adam(LEARNING_RATE), wrt=nnx.Param)


@nnx.jit
def train_step(model: ChessCNN, optimizer: nnx.Optimizer, boards, targets):
    # Backpropagation: the loss, plus which way to nudge every weight (the gradients).
    loss, grads = nnx.value_and_grad(loss_fn)(model, boards, targets)
    # Gradient descent: nudge every weight a small step against its gradient (in GPU memory).
    optimizer.update(model, grads)
    return loss


if __name__ == "__main__":
    print("This isnt implemented yet")
    pass