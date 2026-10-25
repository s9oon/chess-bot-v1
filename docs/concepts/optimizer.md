# Optimizer

The optimizer is the part that actually changes the weights, using the
gradients from [backpropagation](back_propagation.md). We use **Adam**.

For a great explanation behind the Adam optimizer, refer to
[this video at 6:50](https://www.youtube.com/watch?v=vKdKtxpfvIc&t=410s).

## Adam vs plain gradient descent

Plain gradient descent uses one learning rate for every weight and only looks
at the current batch's gradient. Adam adds two running averages **per weight**:

| | Average of | Does |
|---|---|---|
| **m** (momentum) | the gradients | keeps moving in a direction that's been consistent, and smooths out noisy batches |
| **v** | the gradients **squared** | measures how big this weight's gradients usually are |

Each weight's step is then:

```
step = learning_rate × m / (sqrt(v) + ε)
```

Dividing by `sqrt(v)` gives every weight its own step size: weights with big,
jumpy gradients take smaller steps, weights with small, steady gradients take
bigger ones.

At the start `m` and `v` are 0, so early averages are too small. Adam corrects
for that (**bias correction**) during the first steps.

## Settings

The paper's defaults, which `optax.adam` uses:

| Setting | Default | Meaning |
|---|---|---|
| learning rate | 0.001 | step size (set in `config.py` as `LEARNING_RATE`) |
| β1 | 0.9 | how much of the old `m` to keep each step |
| β2 | 0.999 | how much of the old `v` to keep each step |
| ε | 1e-8 | stops a division by zero |

`0.001` is the usual starting learning rate for Adam.

## Only the weights

`wrt=nnx.Param` means Adam only updates the learned weights. Batchnorm's running
averages (`BatchStat`) aren't learned by gradients, so the optimizer leaves them
alone: see [batch_normalization.md](batch_normalization.md).

Adam's `m` and `v` are stored too, two extra numbers for every weight. They're
only needed to carry on training, not to play.

## Further reading

- Kingma & Ba (2014), [*Adam: A Method for Stochastic Optimization*][paper]:
  Algorithm 1 is the whole optimizer on half a page, with the default settings
  above. Section 3 explains the bias correction.
- Goodfellow, Bengio & Courville, [*Deep Learning*][dlbook], section 8.5.3.

[paper]: https://arxiv.org/abs/1412.6980
[dlbook]: https://www.deeplearningbook.org/
