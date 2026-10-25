# model.py

[`src/model.py`](../../src/model.py): the network, and saving/loading its
weights.

## `ChessCNN(channels=CHANNELS, hidden=HIDDEN, *, rngs)`

```
(batch, 8, 8, 18)
  conv1 → bn1 → ReLU      (batch, 8, 8, 64)      convolution, batchnorm
  conv2 → bn2 → ReLU      (batch, 8, 8, 64)
  conv3 → bn3 → ReLU      (batch, 8, 8, 64)
  flatten                 (batch, 4096)
  linear1 → ReLU          (batch, 64)            the head
  linear2 → tanh          (batch,)               one score per board, -1 .. +1
```

Concepts: [the network](../concepts/neural_network.md),
[convolution](../concepts/convolution.md),
[batch normalization](../concepts/batch_normalization.md).

`channels` and `hidden` default to [`config.py`](config.md), and can be passed
in so `load_model` can build a model the size a file was saved at.

Call `model.train()` while training and `model.eval()` when playing
([why](../concepts/batch_normalization.md#train-mode-vs-eval-mode)).

## `save_model(model, path)`

Saves every array to a `.npz` **by name** (`conv1/kernel`, `bn1/mean`, ...),
plus `channels` and `hidden`: the learned weights and batchnorm's running
averages.

## `load_model(path) -> ChessCNN`

1. Builds a `ChessCNN` at the `channels` and `hidden` stored in the file.
2. Checks every array the model expects is in the file, with the same shape.
   If not, it stops with an error naming the layer.
3. Overwrites the new model's random weights with the saved ones.

## Running it

`python src/model.py` prints the JAX version and which devices it can see.
