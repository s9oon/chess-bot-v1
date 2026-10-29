from pathlib import Path

import numpy as np
import jax.numpy as jnp
from flax import nnx

from chessbot.model.network import ChessCNN


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
