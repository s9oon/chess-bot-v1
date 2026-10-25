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

## The model's size

```python
class ChessCNN(nnx.Module):
    def __init__(self, channels: int = CHANNELS, hidden: int = HIDDEN, *, rngs: nnx.Rngs):
        self.channels = channels
        self.hidden = hidden
```

`channels` and `hidden` default to `config.py`, but can be passed in, so
`load_model` can build a model the size a file was saved at. They're kept on the
model so `save_model` can record them. `rngs` provides the random numbers for
the starting weights.

## The conv layers

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

## Batchnorm

```python
        self.bn1 = nnx.BatchNorm(channels, rngs=rngs)
        self.bn2 = nnx.BatchNorm(channels, rngs=rngs)
        self.bn3 = nnx.BatchNorm(channels, rngs=rngs)
```

One per conv, with one scale and bias per output feature map:
[what it does](../concepts/batch_normalization.md#what-it-does).

## The head

```python
        self.linear1 = nnx.Linear(8 * 8 * channels, hidden, rngs=rngs)
        self.linear2 = nnx.Linear(hidden, 1, rngs=rngs)
```

`linear1` takes the flattened feature maps (8 x 8 x 64 = 4096) down to
`hidden` (64); `linear2` takes those down to one score:
[the head](../concepts/neural_network.md#the-head), and how linear differs from
conv: [conv and linear](../concepts/neural_network.md#conv-and-linear).

## The forward pass

```python
    def __call__(self, x):
        x = nnx.relu(self.bn1(self.conv1(x)))  # conv1, then batchnorm, then ReLU
        x = nnx.relu(self.bn2(self.conv2(x)))
        x = nnx.relu(self.bn3(self.conv3(x)))
```

Each conv → batchnorm → ReLU: `(batch, 8, 8, 18)` → `(batch, 8, 8, 64)`.
Batchnorm uses the batch's statistics in `model.train()` and its running
averages in `model.eval()`: [train vs eval](../concepts/batch_normalization.md#train-mode-vs-eval-mode).

```python
        x = x.reshape(x.shape[0], -1)  # flatten: (batch, 8, 8, CHANNELS) -> (batch, 8 * 8 * CHANNELS)
        x = nnx.relu(self.linear1(x))
        x = self.linear2(x)
        x = jnp.tanh(x)
        return x[:, 0]
```

Flatten to `(batch, 4096)`, run the head, squish into -1 .. +1 with tanh. The
last line turns `(batch, 1)` into `(batch,)`: one score per board.

## Saving

```python
def save_model(model: ChessCNN, path: Path) -> None:
    arrays = {"/".join(map(str, name)): np.asarray(variable.get_value())
              for name, variable in nnx.to_flat_state(nnx.state(model))}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, channels=model.channels, hidden=model.hidden, **arrays)
```

`nnx.state(model)` is every array in the model: the learned weights and
batchnorm's running averages. Each is saved under its name, like
`conv1/kernel` or `bn1/mean`, plus `channels` and `hidden`. `mkdir` creates the
folder if it doesn't exist yet.

## Loading

```python
def load_model(path: Path) -> ChessCNN:
    saved = np.load(path)
    model = ChessCNN(int(saved["channels"]), int(saved["hidden"]), rngs=nnx.Rngs(0))
```

Builds a model at the size stored in the file, whatever `config.py` says now.
Its weights start random.

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
layer was added) stops with an error naming the layer. Otherwise the saved
values replace the random ones.

## Running it

```python
if __name__ == "__main__":
    print("JAX", jax.__version__, "devices:", jax.devices())
```

`python src/model.py` prints the JAX version and which devices (GPU/CPU) JAX can
see. A quick check that the GPU is found.
