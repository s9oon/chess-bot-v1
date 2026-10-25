# Batch normalization

## What it does

Batch normalization (**batchnorm**) keeps the numbers flowing through the
network in a healthy range. After a conv layer, it takes each **output feature
map** across the whole batch of boards and:

1. **Normalizes** it: subtracts the average and divides by the spread
   (standard deviation), so the values sit around 0 with a spread of about 1.
2. **Rescales** it with two learned numbers: a **scale** (multiply) and a
   **bias** (add), so the network can still pick whatever range works best.

```
normalized = (x - mean) / sqrt(variance + ε)
output     = scale * normalized + bias
```

`ε` is a tiny number (around 1e-5) that stops a division by zero when a map's
values are all the same.

The shape doesn't change: 64 output feature maps go in, 64 come out.

## One per feature map

`nnx.BatchNorm(64)` has one scale and one bias **per output feature map**:

```
bn1.scale   shape (64,)
bn1.bias    shape (64,)
```

That's why the `64` must match the conv layer's output channels.

The mean and variance for a feature map are taken over **every board in the
batch and every square** of that map. With a batch of 1024 boards, each
feature map's mean comes from 1024 x 64 = 65,536 numbers. A kernel finds the
same pattern everywhere on the board, so its whole output feature map gets
normalized the same way.

## Why we need it

Without it, the numbers coming out of each layer can drift very large or very
small as the weights change during training. Every layer then has to keep
adjusting to the previous layer's shifting outputs, and training becomes slow
and unstable.

Batchnorm resets each layer's output to a steady range, so:

- training is **faster** and more **stable**,
- higher learning rates work,
- deeper networks become practical.

## Where it goes

After the conv, **before** ReLU:

```
conv → batchnorm → ReLU
```

ReLU then sees values centred around 0, so roughly half get switched off
(negative → 0) and half pass through.

## Train mode vs eval mode

Batchnorm behaves differently depending on the mode:

| Mode | Uses | When |
|---|---|---|
| `model.train()` | the mean and spread of the **current batch** | while training |
| `model.eval()` | **running averages** collected during training | when playing / evaluating |

In eval mode the score for a board doesn't depend on which other boards happen
to be in the batch. So always call `model.eval()` before using the network to
play.

The running averages are updated every training step: each new batch's mean
moves them a little (Flax keeps 99% of the old average, adds 1% of the new).
They're stored in the model as `BatchStat` (`bn1/mean`, `bn1/var`), next to the
learned `Param`s, and saved with the weights. The optimizer never touches them:
they come from the data, not from gradients.

## Further reading

- Ioffe & Szegedy (2015), [*Batch Normalization: Accelerating Deep Network
  Training by Reducing Internal Covariate Shift*][paper]: the original paper.
  - Algorithm 1: the normalize-then-rescale steps above.
  - Section 3.1: train vs eval mode (batch statistics vs running averages).
  - Section 3.2: batchnorm for conv layers, normalizing over the batch and every
    position.
  - Section 3.3: why it allows higher learning rates.
- Goodfellow, Bengio & Courville, [*Deep Learning*][dlbook], section 8.7.1.

[paper]: https://arxiv.org/abs/1502.03167
[dlbook]: https://www.deeplearningbook.org/
