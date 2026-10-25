# The neural network

How the pieces fit together, from board to score. Details on each part:
[encoding](encoding.md), [convolution](convolution.md),
[batch normalization](batch_normalization.md), [activations](activations.md),
[loss function](loss_function.md), [backpropagation](back_propagation.md),
[optimizer](optimizer.md).

## The layers

```
input (8, 8, 18)                  the encoded board
  → conv → batchnorm → ReLU   ┐
  → conv → batchnorm → ReLU   ├ hidden layers (conv)
  → conv → batchnorm → ReLU   ┘   (8, 8, 64) output feature maps
  → flatten                       (4096,) in one row
  → linear → ReLU             ┐
  → linear                    ├ the head: turns the features into the score
  → tanh                      ┘   the score, -1 (black winning) to +1 (white winning)
```

In a CNN like ours, the **hidden layers are conv layers** and **linear comes at
the end**. That end part is called the **head**.

Our head has **two** linear layers with ReLU between them (4096 → 64 → 1). The
middle linear layer is also a hidden layer, one that sits in the head.

## Conv and linear are both layers

Both do the same affine transformation, `W · x + b`. They differ in what `x`
is:

| | Conv (`nnx.Conv`) | Linear (`nnx.Linear`) |
|---|---|---|
| `x` is | one small **patch**, repeated at every position | the whole **flattened** input, done once |
| weights | **shared**: the same kernel everywhere | every input has **its own** weight |
| sees | a local area | the whole board |
| job | find patterns | judge them into a score |

A linear layer is like a conv whose kernel covers the entire input, with no
sliding.

## The head

After the conv layers we **flatten** the output feature maps and apply
**linear layers**: weight matrices (plus biases) that turn them into the score.

```
score = W · x + b
        │   │   └─ bias
        │   └───── the flattened output feature maps (4096 numbers)
        └───────── the linear layer's weight matrix
```

The **weights** live inside `W`: learned and fixed. What comes out is a
**value**: the score for this board.

Finally, **tanh** takes that score and squishes it into a number between
**-1 and +1**: -1 means black is winning, 0 is even, +1 means white is winning.

### Why a linear head?

Conv layers are good at **spotting patterns**, but not at **evaluating how
important they are**. That's why we stack linear layers on top:

```
conv layers  →  detect patterns    (what's going on, and where)
linear head  →  evaluate them      (so who's winning?)
```

Conv layers have two blind spots:

1. **They only see locally.** After three conv layers each number only knows
   about a 7x7 area, but an evaluation is about the whole board.
2. **They treat every square the same.** A kernel uses the same weights
   everywhere, so it can't tell that a pawn on the 7th rank matters more than
   one on the 2nd.

The linear layer connects every input to the output and gives each one its own
weight, so it sees the whole board and can learn that *where* something is
matters.

## Where the weights are

```
conv1      10,432
conv2      36,928
conv3      36,928
bn1-3         384   (scale + bias, 64 each)
linear1   262,208   (4096 x 64 + 64)
linear2        65
         ────────
          346,945 weights
```

Over three quarters of them are in `linear1`, the layer that gives every square
of every feature map its own weight.

## CNN vs FNN

| | Layers |
|---|---|
| **FNN** (feed-forward / fully connected network) | **only** linear layers: every layer, hidden ones included |
| **CNN** | conv layers to find patterns, usually **plus** a small linear head |

## Further reading

- Goodfellow, Bengio & Courville, [*Deep Learning*][dlbook]: chapter 6 for
  feed-forward networks and hidden layers, chapter 9 for CNNs.
- Silver et al. (2017), [*Mastering the game of Go without human
  knowledge*][alphago-zero], the "Neural network architecture" part of Methods:
  AlphaGo Zero's **value head** is the same idea as ours (conv layers, then a
  linear layer, ReLU, a linear layer to one number, and tanh), just much
  deeper.

[dlbook]: https://www.deeplearningbook.org/
[alphago-zero]: https://www.nature.com/articles/nature24270
