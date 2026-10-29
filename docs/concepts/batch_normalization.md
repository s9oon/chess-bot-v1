# Batch normalization

Batch normalization (**batchnorm**) keeps the numbers flowing through the
network in a steady range. It sits after each conv, before ReLU. From
[`model/network.py`](../../src/chessbot/model/network.py):

```python
        x = nnx.relu(self.bn1(self.conv1(x)))  # conv1, then batchnorm, then ReLU
```

## What it does

For each **output feature map**, over every board in the batch and every
square:

1. **Normalize:** subtract the mean, divide by the standard deviation, so the
   values sit around 0 with a spread of about 1.
2. **Rescale** with two learned numbers, a **scale** and a **bias**, so the
   network can still pick whatever range works best.

```
output = scale * (x - mean) / sqrt(variance + ε) + bias      ε ≈ 1e-5 avoids dividing by 0
```

There's one scale and one bias per output feature map, so each batchnorm gets
the number of kernels in the conv before it:

```python
        self.bn1 = nnx.BatchNorm(channels, rngs=rngs)   # 64: one per conv1 output feature map
```

The shape doesn't change: `(n, 8, 8, 64)` in, `(n, 8, 8, 64)` out.

**Why:** without it, each layer's outputs drift as the weights change, and
every following layer has to keep re-adjusting. Training gets slow and
unstable. With it, training is faster, more stable, and works with higher
learning rates.

## Train mode vs eval mode

| Mode | Normalizes with | When |
|---|---|---|
| `model.train()` | the **current batch's** mean and variance | while training |
| `model.eval()` | **running averages** collected during training | when playing |

In eval mode a board's score doesn't depend on which other boards are in the
batch, so always call `model.eval()` before playing.

The running averages (`bn1/mean`, `bn1/var`) come from the data, not from
gradients, so the optimizer must leave them alone. That's what `wrt=nnx.Param`
does in [`training/supervised.py`](../../src/chessbot/training/supervised.py):
only learned weights (`nnx.Param`) get updated, not running averages
(`nnx.BatchStat`).

```python
    return nnx.Optimizer(model, optax.adam(LEARNING_RATE), wrt=nnx.Param)
```

They're still part of the trained model, so
[`model/checkpoint.py`](../../src/chessbot/model/checkpoint.py) saves every
array in the model's state, running averages included.

## Further reading

- Ioffe & Szegedy (2015), [*Batch Normalization: Accelerating Deep Network
  Training by Reducing Internal Covariate Shift*][paper]:
  - Algorithm 1: the normalize-then-rescale steps above.
  - Section 3.1: train vs eval mode.
  - Section 3.2: batchnorm for conv layers, over the batch and every position.
  - Section 3.3: why it allows higher learning rates.
- Goodfellow, Bengio & Courville, [*Deep Learning*][dlbook], section 8.7.1.

[paper]: https://arxiv.org/abs/1502.03167
[dlbook]: https://www.deeplearningbook.org/
