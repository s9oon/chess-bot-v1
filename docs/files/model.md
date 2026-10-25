# model.py

[`src/model.py`](../../src/model.py): the network, and saving/loading its
weights. How it all fits together:
[concepts/neural_network.md](../concepts/neural_network.md).

## `ChessCNN(channels=CHANNELS, hidden=HIDDEN, *, rngs)`

```
(batch, 8, 8, 18)
  conv1 → bn1 → ReLU      (batch, 8, 8, 64)
  conv2 → bn2 → ReLU      (batch, 8, 8, 64)
  conv3 → bn3 → ReLU      (batch, 8, 8, 64)
  flatten                 (batch, 4096)
  linear1 → ReLU          (batch, 64)
  linear2 → tanh          (batch,)   one score per board, -1 .. +1
```

| Part | Concept |
|---|---|
| `conv1`-`conv3` | [convolution](../concepts/convolution.md) |
| `bn1`-`bn3` | [batch normalization](../concepts/batch_normalization.md) |
| `relu`, `tanh` | [activations](../concepts/activations.md) |
| `linear1`, `linear2` | [the head](../concepts/neural_network.md#the-head) |

`channels` and `hidden` default to the values in [`config.py`](config.md). They
can be passed in so `load_model` can build a model the size a file was saved
at.

Call `model.train()` while training and `model.eval()` when playing: batchnorm
behaves differently in each ([train vs eval](../concepts/batch_normalization.md#train-mode-vs-eval-mode)).

## `save_model(model, path)`

Saves every array in the model to a `.npz`, **by name** (`conv1/kernel`,
`bn1/mean`, ...), plus `channels` and `hidden`. That's the learned weights
(`Param`) and batchnorm's running averages (`BatchStat`), which eval mode needs.

## `load_model(path) -> ChessCNN`

1. Reads `channels` and `hidden` from the file and builds a `ChessCNN` that
   size, whatever `config.py` says now.
2. For every array the model expects, checks the file has one with the same
   name and shape. If not, it stops with an error naming the layer, e.g. a file
   saved before a layer was added.
3. Overwrites the new model's random weights with the saved ones.

## Running it

`python src/model.py` prints the JAX version and which devices (GPU/CPU) it can
see.
