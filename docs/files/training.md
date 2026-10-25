# training.py

[`src/training.py`](../../src/training.py): teaches the network to match
Stockfish's scores. The training loop isn't written yet; the pieces it will use
are.

One training step:

```
packed batch ──unpack_planes──> boards ──model──> scores
scores vs targets ──loss_fn──> loss ──backprop──> gradients ──Adam──> new weights
```

## `loss_fn(model, boards, targets)`

Mean squared error between the network's scores and the targets:
[loss function](../concepts/loss_function.md).

## `create_optimizer(model)`

Adam with `LEARNING_RATE` from [`config.py`](config.md), updating only the
learned weights (`nnx.Param`): [optimizer](../concepts/optimizer.md).

## `train_step(model, optimizer, boards, targets)`

Compiled with `@nnx.jit`, runs on the GPU:

1. `nnx.value_and_grad(loss_fn)` gives the loss and a gradient for every
   weight: [backpropagation](../concepts/back_propagation.md).
2. `optimizer.update` nudges every weight against its gradient:
   [gradient descent](../concepts/back_propagation.md#gradient-descent).

Returns the loss for that batch.

## Still to do

- The loop: [`load_positions`](data.md), then for each batch from
  [`batches`](data.md), `unpack_planes` → `train_step`.
- Save the model with [`save_model`](model.md) to `data/weights/` at the end,
  and every so often during training.
- Set `LEARNING_RATE` in `config.py` (it's 0, so nothing would learn yet).
