"""Shared by every source: turning (boards, targets) into batches for training."""
import numpy as np


def batches(boards: np.ndarray, targets: np.ndarray, batch_size: int, rng: np.random.Generator):
    """One epoch of shuffled batches. Still packed: training unpacks them on the GPU."""
    order = rng.permutation(len(boards))
    for start in range(0, len(order) - batch_size + 1, batch_size):
        idx = order[start:start + batch_size]
        yield boards[idx], targets[idx]
