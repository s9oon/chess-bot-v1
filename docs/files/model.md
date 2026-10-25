# model.py

[`src/model.py`](../../src/model.py)

## Imports

```python
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from config import CHANNELS, HIDDEN
from encoding import NUM_PLANES
```

Flax's `nnx` provides the layers. `CHANNELS` and `HIDDEN` are the default sizes;
`NUM_PLANES` (18) is how many input feature maps a board has.

## Building the model

```python
class ChessCNN(nnx.Module):
    def __init__(self, channels: int = CHANNELS, hidden: int = HIDDEN, *, rngs: nnx.Rngs):
        self.channels = channels
        self.hidden = hidden
```

**Does:** builds the network's layers (below) with random starting weights.
`channels` and `hidden` set its size and default to `config.py`; they're stored
on the model. `rngs` provides the random numbers for the starting weights.

**Used:** training will create one to train. `load_model` creates one at the
size a saved file records.

### The conv layers

```python
        input_maps = output_maps = channels

        self.conv1 = nnx.Conv(NUM_PLANES, output_maps, (3, 3), padding="SAME", rngs=rngs)
        self.conv2 = nnx.Conv(input_maps, output_maps, (3, 3), padding="SAME", rngs=rngs)
        self.conv3 = nnx.Conv(input_maps, output_maps, (3, 3), padding="SAME", rngs=rngs)
```

- `output_maps` is the number of kernels, one output feature map each:
  [what a convolution is](../concepts/convolution.md#what-a-convolution-is).
- `(3, 3)` and `padding="SAME"` keep the board 8x8:
  [our settings](../concepts/convolution.md#our-settings-in-the-papers-terms).
- conv1 reads 18 input feature maps, conv2 and conv3 read 64, which is why
  `input_maps = output_maps`: [many channels](../concepts/convolution.md#many-channels).

### Batchnorm

```python
        self.bn1 = nnx.BatchNorm(channels, rngs=rngs)
        self.bn2 = nnx.BatchNorm(channels, rngs=rngs)
        self.bn3 = nnx.BatchNorm(channels, rngs=rngs)
```

One per conv, with one scale and bias per output feature map:
[what it does](../concepts/batch_normalization.md#what-it-does).

### The head

```python
        self.linear1 = nnx.Linear(8 * 8 * channels, hidden, rngs=rngs)
        self.linear2 = nnx.Linear(hidden, 1, rngs=rngs)
```

`linear1` takes the flattened feature maps (8 x 8 x 64 = 4096) down to
`hidden` (64); `linear2` takes those down to one score:
[the head](../concepts/neural_network.md#the-head), and how linear differs from
conv: [conv and linear](../concepts/neural_network.md#conv-and-linear).

## Scoring boards: the forward pass

```python
    def __call__(self, x):
        x = nnx.relu(self.bn1(self.conv1(x)))  # conv1, then batchnorm, then ReLU
        x = nnx.relu(self.bn2(self.conv2(x)))
        x = nnx.relu(self.bn3(self.conv3(x)))
```

**Does:** `(n, 8, 8, 18)` feature maps → `(n,)` scores in -1 .. +1, for any `n`,
all at once. Called as `model(x)`.

**Used:** inside `loss_fn` on each batch while training. Playing will call it
on the positions the bot is choosing between.

Each conv → batchnorm → ReLU: `(n, 8, 8, 18)` → `(n, 8, 8, 64)`. In
`model.train()` batchnorm normalizes with the statistics of the `n` boards it
was given; in `model.eval()` it uses its running averages instead:
[train vs eval](../concepts/batch_normalization.md#train-mode-vs-eval-mode).

```python
        x = x.reshape(x.shape[0], -1)  # flatten: (n, 8, 8, CHANNELS) -> (n, 8 * 8 * CHANNELS)
        x = nnx.relu(self.linear1(x))
        x = self.linear2(x)
        x = jnp.tanh(x)
        return x[:, 0]
```

Flatten to `(n, 4096)`, run the head, squish into -1 .. +1 with tanh. The
last line turns `(n, 1)` into `(n,)`: one score per board.

## Saving

```python
def save_model(model: ChessCNN, path: Path) -> None:
    arrays = {"/".join(map(str, name)): np.asarray(variable.get_value())
              for name, variable in nnx.to_flat_state(nnx.state(model))}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, channels=model.channels, hidden=model.hidden, **arrays)
```

**Does:** writes every array in the model to a `.npz` at `path`, each under its
name (`conv1/kernel`, `bn1/mean`, ...), plus `channels` and `hidden`. That's the
learned weights and batchnorm's running averages. Creates the folder if it
doesn't exist.

**Used:** training will save to `data/weights/`. Nothing calls it yet.

## Loading

```python
def load_model(path: Path) -> ChessCNN:
    saved = np.load(path)
    model = ChessCNN(int(saved["channels"]), int(saved["hidden"]), rngs=nnx.Rngs(0))
```

**Does:** reads a `.npz` written by `save_model` and returns a `ChessCNN` with
those weights. Raises an error naming the layer if the file doesn't match the
model.

**Used:** to load a trained model for playing, or to continue training it.
Nothing calls it yet.

First it builds a model at the size stored in the file, whatever `config.py`
says now. Its weights start random.

```python
    flat = nnx.to_flat_state(nnx.state(model))
    for name, variable in flat:
        key = "/".join(map(str, name))
        if key not in saved:
            raise ValueError(f"{path.name} has no weights for {key}")
        if saved[key].shape != variable.get_value().shape:
            raise ValueError(f"{path.name}: {key} is {saved[key].shape}, model expects {variable.get_value().shape}")
        variable.set_value(jnp.asarray(saved[key]))

    nnx.update(model, nnx.from_flat_state(flat))
    return model
```

For every array the model has, finds the one with the same name in the file and
checks its shape. A missing or wrong-shaped array (e.g. a file saved before a
layer was added) stops with the error. Otherwise the saved values replace the
random ones.

## Running it

```python
if __name__ == "__main__":
    print("JAX", jax.__version__, "devices:", jax.devices())
```

**Does:** prints the JAX version and which devices (GPU/CPU) JAX can see.

**Used:** `python src/model.py`, a quick check that the GPU is found.
