from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from config import CHANNELS, HIDDEN
from encoding import NUM_PLANES


class ChessCNN(nnx.Module):
    """A simple convolutional neural network for chess board evaluation."""

    def __init__(self, channels: int = CHANNELS, hidden: int = HIDDEN, *, rngs: nnx.Rngs):
        # Kept so save_model can record which size this model is.
        self.channels = channels
        self.hidden = hidden

        # conv2 and conv3 read the previous conv's output feature maps, so input = output count.
        input_maps = output_maps = channels

        # conv1 reads every input feature map from encoding.py (18: pieces, turn, castling, en passant).
        self.conv1 = nnx.Conv(NUM_PLANES, output_maps, (3, 3), padding="SAME", rngs=rngs)
        self.conv2 = nnx.Conv(input_maps, output_maps, (3, 3), padding="SAME", rngs=rngs)
        self.conv3 = nnx.Conv(input_maps, output_maps, (3, 3), padding="SAME", rngs=rngs)

        # Batch normalization after each conv: one per output feature map.
        # Use model.train() while training and model.eval() when playing.
        self.bn1 = nnx.BatchNorm(channels, rngs=rngs)
        self.bn2 = nnx.BatchNorm(channels, rngs=rngs)
        self.bn3 = nnx.BatchNorm(channels, rngs=rngs)

        # Two heads read the same patterns the convs found.
        # Value head: who's winning. value1 is a hidden layer: 8 x 8 x channels flattened values -> hidden combinations.
        # value2 is its output layer: hidden combinations -> 1 score.
        self.value1 = nnx.Linear(8 * 8 * channels, hidden, rngs=rngs)
        self.value2 = nnx.Linear(hidden, 1, rngs=rngs)
        # Policy head: one score per move, from-square x to-square (64 x 64 = 4096, most illegal in any position).
        # No hidden layer: squeezing through `hidden` (64) on the way to 4096 moves would be a bottleneck.
        self.policy = nnx.Linear(8 * 8 * channels, 64 * 64, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.bn1(self.conv1(x)))  # conv1, then batchnorm, then ReLU
        x = nnx.relu(self.bn2(self.conv2(x)))
        x = nnx.relu(self.bn3(self.conv3(x)))

        x = x.reshape(x.shape[0], -1)  # flatten: (n, 8, 8, CHANNELS) -> (n, 8 * 8 * CHANNELS)

        value = nnx.relu(self.value1(x))  # (n, HIDDEN): ReLU switches each combination on/off
        value = self.value2(value)        # (n, 1): the score
        value = jnp.tanh(value)[:, 0]     # (n,): squished into -1 (black winning) .. +1 (white winning)

        policy = self.policy(x)  # (n, 4096): raw move scores (logits); masking and softmax happen outside
        return policy, value


def save_model(model: ChessCNN, path: Path) -> None:
    """Save every weight and batchnorm's running averages by name, plus the model's size."""
    # Flat state: ("conv1", "kernel") -> array, for every Param and BatchStat in the model.
    arrays = {"/".join(map(str, name)): np.asarray(variable.get_value())
              for name, variable in nnx.to_flat_state(nnx.state(model))}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, channels=model.channels, hidden=model.hidden, **arrays)


def load_model(path: Path) -> ChessCNN:
    """Build a ChessCNN the size the file was saved at, then overwrite its random weights with the saved ones."""
    saved = np.load(path)
    model = ChessCNN(int(saved["channels"]), int(saved["hidden"]), rngs=nnx.Rngs(0))

    flat = nnx.to_flat_state(nnx.state(model))
    for name, variable in flat:
        key = "/".join(map(str, name))
        # Catches files from an older architecture (a layer added, removed or resized).
        if key not in saved:
            raise ValueError(f"{path.name} has no weights for {key}")
        if saved[key].shape != variable.get_value().shape:
            raise ValueError(f"{path.name}: {key} is {saved[key].shape}, model expects {variable.get_value().shape}")
        variable.set_value(jnp.asarray(saved[key]))

    nnx.update(model, nnx.from_flat_state(flat))
    return model


if __name__ == "__main__":
    print("JAX", jax.__version__, "devices:", jax.devices())