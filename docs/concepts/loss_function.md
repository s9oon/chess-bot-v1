# Loss function

The loss function is the arithmetic used to define the difference between the
correct answer (in this case Stockfish's evaluation) and our answer (the
network's score). Training tries to make this number as small as possible.

## Mean squared error (MSE)

For each board: take the error, **square** it, then **average** over the batch.

```
ours    Stockfish   error   squared
 0.20     0.60      -0.40     0.16
-0.10    -0.50       0.40     0.16
 0.70     0.80      -0.10     0.01
                             ─────
                      loss = 0.11
```

- **Squaring** makes every error positive, so a too-high and a too-low guess
  can't cancel out, and big mistakes cost much more than small ones.
- **Averaging** turns the whole batch into one number.

## Targets must match our range

The network outputs -1 .. +1 (tanh), so Stockfish's centipawn scores are
squashed into the same range first:

```
target = tanh(cp / CP_SCALE)        CP_SCALE = 400
```

| cp (pawns up) | target |
|---|---|
| 0 (equal) | 0.00 |
| 100 (1 pawn) | 0.24 |
| 400 (4 pawns) | 0.76 |
| 900 (a queen) | 0.98 |

`CP_SCALE` decides how many centipawns count as "nearly winning". 400 keeps
the range that matters most in real games, roughly 0 to 5 pawns, in the part of
tanh that still changes (see [activations.md](activations.md)). A forced mate
has no centipawn value, so it becomes -1 or +1.

Each position has evaluations at several search depths; the target uses the
**deepest** one, and its **best line**.

## Why a batch of positions

Backpropagation works from this one loss number: it traces back how much each
weight contributed to it. One position on its own gives a noisy, unreliable
direction. Averaging over a batch of many positions, each with its own
Stockfish score, gives a much more reliable direction to nudge the weights.

## Further reading

- Goodfellow, Bengio & Courville, [*Deep Learning*][dlbook], section 5.5.1:
  why MSE is the natural loss for predicting a number. Section 8.1.3: why
  training uses batches (minibatches) rather than one example or the whole
  dataset.

[dlbook]: https://www.deeplearningbook.org/
