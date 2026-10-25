# model.py

[`src/model.py`](../../src/model.py)

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

x = nnx.relu(self.bn1(self.conv1(x)))  # conv1, then batchnorm, then ReLU
```

One scale and bias per output feature map, between the conv and ReLU:
[what it does](../concepts/batch_normalization.md#what-it-does). This is why
you call `model.train()` / `model.eval()`:
[train vs eval](../concepts/batch_normalization.md#train-mode-vs-eval-mode).

## The head

```python
self.linear1 = nnx.Linear(8 * 8 * channels, hidden, rngs=rngs)
self.linear2 = nnx.Linear(hidden, 1, rngs=rngs)

x = x.reshape(x.shape[0], -1)  # flatten: (batch, 8, 8, 64) -> (batch, 4096)
x = nnx.relu(self.linear1(x))
x = self.linear2(x)
x = jnp.tanh(x)
```

Flatten, then linear layers that see the whole board and judge it into one
score in -1 .. +1: [the head](../concepts/neural_network.md#the-head), and how
linear differs from conv: [conv and linear](../concepts/neural_network.md#conv-and-linear).

## Saving and loading

`save_model` writes every array by name (`conv1/kernel`, `bn1/mean`, ...) plus
`channels` and `hidden`. `load_model` builds a model at the file's size, checks
every layer's name and shape, then loads the weights.
