# training/supervised.py

[`src/training/supervised.py`](../../src/training/supervised.py): learn to
predict a dataset's target scores, e.g. Stockfish's evaluations from
[lichess_evals](lichess_evals.md). The training loop isn't written yet; these
are the pieces it will use.

```bash
make train METHOD=supervised
```

The loop will connect everything in this order (not wired up yet):

```
load_positions()   all packed boards → RAM                    (lichess_evals.py, already packed once by pack_board)
batches()          RAM → one packed batch at a time            (batches.py)
unpack_planes()    that batch → (1024, 8, 8, 18) on the GPU     (encoding.py)
train_step()       learn from it, then the next batch          (this file)
```

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

**Does:** runs the model on the boards it's given (any number), squares each
score's error against its target, and averages them into one number: mean
squared error.

**Used:** inside `train_step`, on each batch. Training tries to make it as small
as possible.

## The optimizer

```python
def create_optimizer(model: ChessCNN) -> nnx.Optimizer:
    return nnx.Optimizer(model, optax.adam(LEARNING_RATE), wrt=nnx.Param)
```

**Does:** returns an Adam optimizer for `model`, with the step size
`LEARNING_RATE` from `config.py`. `wrt=nnx.Param` means it only updates the
learned weights, not batchnorm's running averages:
[train vs eval](../concepts/batch_normalization.md#train-mode-vs-eval-mode).

**Used:** once, at the start of training. `LEARNING_RATE` is still 0, so
nothing would learn yet; 0.001 is the usual start.

## One training step

```python
@nnx.jit
def train_step(model: ChessCNN, optimizer: nnx.Optimizer, boards, targets):
    loss, grads = nnx.value_and_grad(loss_fn)(model, boards, targets)
    optimizer.update(model, grads)
    return loss
```

**Does:** one update of the model's weights from the boards and targets it's
given, compiled by `@nnx.jit` to run on the GPU:

1. `value_and_grad` runs `loss_fn` and also works out a gradient for every
   weight: which way, and how much, it should move to lower the loss
   (backpropagation).
2. `optimizer.update` nudges every weight against its gradient (gradient
   descent).

Returns the loss for those boards.

**Used:** by the loop, once per batch from [batches.py](batches.md). The
returned loss is how you track whether training is working.

## Running it

```python
if __name__ == "__main__":
    print("This isnt implemented yet")
    pass
```

**Does:** prints a placeholder message.

**Used:** `make train METHOD=supervised` runs it. The training loop goes here.
