# training.py

[`src/training.py`](../../src/training.py). The training loop isn't written
yet; these are the pieces it will use.

## Imports

```python
import time

import numpy as np
import jax.numpy as jnp
import optax
from flax import nnx

from config import LEARNING_RATE
from model import ChessCNN
```

Optax provides the optimizer (Adam). `time` and `np` aren't used yet; they're
for the loop.

## The loss

```python
def loss_fn(model, boards, targets):
    scores = model(boards)
    return jnp.mean((scores - targets) ** 2)
```

Mean squared error: run the model on a batch, square each score's error, and
average into one number. Training tries to make it as small as possible.

## The optimizer

```python
def create_optimizer(model: ChessCNN) -> nnx.Optimizer:
    return nnx.Optimizer(model, optax.adam(LEARNING_RATE), wrt=nnx.Param)
```

Adam, with the step size from `config.py` (`LEARNING_RATE` is still 0, so
nothing would learn yet; 0.001 is the usual start). `wrt=nnx.Param` means it only
updates the learned weights, not batchnorm's running averages:
[train vs eval](../concepts/batch_normalization.md#train-mode-vs-eval-mode).

## One training step

```python
@nnx.jit
def train_step(model: ChessCNN, optimizer: nnx.Optimizer, boards, targets):
    loss, grads = nnx.value_and_grad(loss_fn)(model, boards, targets)
    optimizer.update(model, grads)
    return loss
```

`@nnx.jit` compiles it to run on the GPU.

1. `value_and_grad` runs `loss_fn` and also works out a gradient for every
   weight: which way, and how much, it should move to lower the loss
   (backpropagation).
2. `optimizer.update` nudges every weight against its gradient (gradient
   descent).

Returns the loss for that batch, to track whether training is working.

## Running it

```python
if __name__ == "__main__":
    print("This isnt implemented yet")
    pass
```

Placeholder for the training loop.
