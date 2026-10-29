# Training

Training adjusts the network's weights so its outputs get closer to
**targets**: the outputs it should have given. This page uses
**supervised** training, the value head learning Stockfish's evaluations
([`training/supervised.py`](../../src/chessbot/training/supervised.py)); its
loop isn't written yet. The same ideas run [self-play](SelfPlay.md), where the
targets come from the bot's own games instead.

## Targets

The dataset ([`datasets/lichess_evals.py`](../../src/chessbot/datasets/lichess_evals.py))
gives Stockfish's score in **centipawns** (100 = one pawn up). The value head
outputs -1 .. +1, so the target is squished the same way, with tanh:

```python
    if "mate" in best:
        return 1.0 if best["mate"] > 0 else -1.0  # forced mate: as good (or bad) as it gets
    return float(np.tanh(best["cp"] / CP_SCALE))  # cp = centipawns, 100 = one pawn up
```

With `CP_SCALE = 400`, four pawns up is tanh(1) = 0.76.

## Loss

The **loss** is one number for how wrong the network is. Ours is **mean
squared error**: each score's error against its target, squared, averaged over
the batch.

```python
def loss_fn(model, boards, targets):
    _, scores = model(boards)  # only the value head: the dataset has no move targets
    return jnp.mean((scores - targets) ** 2)
```

Squaring makes every error positive and punishes big errors more than small
ones.

## Gradients and gradient descent

The **gradient** of the loss says, for every weight, which way to move it to
make the loss bigger, and how strongly. **Backpropagation** computes all of
them at once, working backwards from the loss through every layer.

**Gradient descent** moves every weight a small step the other way, to make
the loss smaller. The step size is the **learning rate**.

```python
@nnx.jit
def train_step(model: ChessCNN, optimizer: nnx.Optimizer, boards, targets):
    loss, grads = nnx.value_and_grad(loss_fn)(model, boards, targets)  # backpropagation
    optimizer.update(model, grads)                                     # gradient descent
    return loss
```

`@nnx.jit` compiles the whole step to run on the GPU.

## Adam

We use **Adam**, a version of gradient descent that keeps a running average of
each weight's recent gradients, and gives each weight its own step size from
them. It usually trains faster and needs less tuning than plain gradient
descent.

```python
    return nnx.Optimizer(model, optax.adam(LEARNING_RATE), wrt=nnx.Param)
```

`LEARNING_RATE` in `config.py` is 0.001, the usual starting point for Adam.

## Batches and epochs

Each step learns from a **batch** of positions (`BATCH_SIZE = 1024`), not one
at a time and not all at once: one is noisy and slow, all at once doesn't fit
on the GPU. One pass over the whole dataset is an **epoch**. From
[`training/batches.py`](../../src/chessbot/training/batches.py):

```python
    order = rng.permutation(len(boards))
    for start in range(0, len(order) - batch_size + 1, batch_size):
        idx = order[start:start + batch_size]
        yield boards[idx], targets[idx]
```

The positions are **shuffled** every epoch, so each batch is a random mix and
the network can't learn the file's order.

## Further reading

- Goodfellow, Bengio & Courville, [*Deep Learning*][dlbook]: section 4.3 for
  gradient descent, 6.5 for backpropagation, 8.1.3 for batches, 8.5.3 for Adam.
- Kingma & Ba (2014), [*Adam: A Method for Stochastic Optimization*][adam],
  Algorithm 1.

[dlbook]: https://www.deeplearningbook.org/
[adam]: https://arxiv.org/abs/1412.6980
