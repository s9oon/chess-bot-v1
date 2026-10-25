# The neural network

From board to score. The conv layers and batchnorm have their own pages:
[convolution](convolution.md), [batch normalization](batch_normalization.md).

```
input (8, 8, 18)                  the encoded board
  → conv → batchnorm → ReLU   ┐
  → conv → batchnorm → ReLU   ├ hidden layers: find patterns
  → conv → batchnorm → ReLU   ┘   (8, 8, 64) output feature maps
  → flatten                       (4096,)
  → linear → ReLU             ┐
  → linear                    ├ the head: judge the patterns into a score
  → tanh                      ┘   -1 (black winning) .. +1 (white winning)
```

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

## The head

Conv layers have two blind spots for judging a position:

1. **They only see locally.** After three 3x3 convs, each number knows about a
   7x7 area, not the whole board.
2. **They treat every square the same.** A shared kernel can't tell that a pawn
   on the 7th rank matters more than one on the 2nd.

So we **flatten** the output feature maps and add linear layers, which connect
every input to the output with its own weight. Ours is 4096 → 64 → 1, with ReLU
between. **tanh** then squishes the final number into -1 .. +1.

Of the network's 346,945 weights, about three quarters are in the first linear
layer (4096 x 64).

## Further reading

- Goodfellow, Bengio & Courville, [*Deep Learning*][dlbook]: chapter 6 for
  feed-forward networks and hidden layers, chapter 9 for CNNs.
- Silver et al. (2017), [*Mastering the game of Go without human
  knowledge*][alphago-zero], "Neural network architecture" in Methods:
  AlphaGo Zero's **value head** is the same idea as ours (convs, then linear,
  ReLU, linear to one number, tanh), just much deeper.

[dlbook]: https://www.deeplearningbook.org/
[alphago-zero]: https://www.nature.com/articles/nature24270
