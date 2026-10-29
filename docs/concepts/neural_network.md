# The neural network

From board to a move policy and a score. Each part has its own page:
[board encoding](board_encoding.md), [convolution](convolution.md),
[batch normalization](batch_normalization.md),
[policy and value](policy_and_value.md).

```
input (8, 8, 18)                  the encoded board
  → conv → batchnorm → ReLU   ┐
  → conv → batchnorm → ReLU   ├ the body: find patterns
  → conv → batchnorm → ReLU   ┘   (8, 8, 64) output feature maps
  → flatten                       (4096,)
     ├→ linear → ReLU → linear → tanh     value head: one score, -1 .. +1
     └→ linear                            policy head: 4096 move scores
```

The whole network is one class, built from layer objects. From
[`model/network.py`](../../src/chessbot/model/network.py):

```python
class ChessCNN(nnx.Module):
    def __init__(self, channels: int = CHANNELS, hidden: int = HIDDEN, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(NUM_PLANES, output_maps, (3, 3), padding="SAME", rngs=rngs)
        ...
        self.value1 = nnx.Linear(8 * 8 * channels, hidden, rngs=rngs)
        self.value2 = nnx.Linear(hidden, 1, rngs=rngs)
        self.policy = nnx.Linear(8 * 8 * channels, 64 * 64, rngs=rngs)
```

`__init__` builds the layers with random weights. `__call__` is the **forward
pass**: it runs any number of boards through them at once, `(n, 8, 8, 18)` →
`(policy, value)`.

## Conv and linear

Both are layers doing the same affine transformation, `W · x + b`. They differ
in what `x` is:

| | Conv | Linear |
|---|---|---|
| `x` is | one small **patch**, at every position | the whole **flattened** input, once |
| weights | **shared**: the same kernel everywhere | **its own** weight per input |
| sees | a local area | the whole board |

A linear layer is like a conv whose kernel covers the entire input, with no
sliding.

## ReLU

Between layers, **ReLU** keeps positive numbers and turns negatives into 0.
Without it, layers of `W · x + b` stacked on each other collapse into a single
affine transformation, and depth would add nothing.

```python
        x = nnx.relu(self.bn1(self.conv1(x)))  # conv1, then batchnorm, then ReLU
```

## The heads

Conv layers have two blind spots for judging a position:

1. **They only see locally.** After three 3x3 convs, each number knows about a
   7x7 area, not the whole board.
2. **They treat every square the same.** A shared kernel can't tell that a pawn
   on the 7th rank matters more than one on the 2nd.

So we **flatten** the output feature maps and add linear layers, which connect
every input to the output with its own weight:

```python
        x = x.reshape(x.shape[0], -1)  # flatten: (n, 8, 8, CHANNELS) -> (n, 8 * 8 * CHANNELS)
```

Two **heads** read that same flattened vector: the value head (who's winning)
and the policy head (which moves look good). The body runs once and both share
it, so the patterns it learns have to serve both. What each head outputs:
[policy and value](policy_and_value.md).

## Size

17,128,257 weights. The policy head is 98% of them (4096 x 4096 = 16.8M); the
body is about 85k, the value head about 262k.

## Further reading

- Goodfellow, Bengio & Courville, [*Deep Learning*][dlbook]: chapter 6 for
  feed-forward networks, hidden layers and ReLU, chapter 9 for CNNs.
- Silver et al. (2017), [*Mastering the game of Go without human
  knowledge*][alphago-zero], "Neural network architecture" in Methods: one
  body with a policy head and a value head, the same layout as ours, just much
  deeper.

[dlbook]: https://www.deeplearningbook.org/
[alphago-zero]: https://www.nature.com/articles/nature24270
