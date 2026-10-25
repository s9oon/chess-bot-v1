# Batch normalization

Batch normalization (**batchnorm**) keeps the numbers flowing through the
network in a steady range. It sits after each conv, before ReLU:

```
conv → batchnorm → ReLU
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

There's one scale and one bias per output feature map: `nnx.BatchNorm(64)`
after a conv with 64 kernels. The shape doesn't change.

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
gradients, so the optimizer never touches them. They're saved with the weights.

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
