# Activation functions

## ReLU

**ReLU is an activation function applied to all output feature maps.** After
each conv layer (and its batchnorm), it goes over every number in every output
feature map and turns the negatives into 0:

```
relu(x) = max(0, x)

feature map before ReLU:   [ 2.0, -1.3,  0.7, -0.4, ... ]
feature map after ReLU:    [ 2.0,  0.0,  0.7,  0.0, ... ]
```

The shape doesn't change: 64 output feature maps of 8x8 go in, 64 come out.
Then they become the next conv layer's input feature maps:

```
conv1 -> 64 output feature maps -> ReLU -> into conv2
conv2 -> 64 output feature maps -> ReLU -> into conv3
conv3 -> 64 output feature maps -> ReLU -> into the head
```

## Why we need it

Each conv layer is an affine transformation (`W · x + b`). Stacked on their
own, affine transformations collapse into a single one:

```
W2 · (W1 · x + b1) + b2  =  W · x + b
```

So without ReLU, three conv layers would be no better than one.

ReLU is **non-linear**, which stops the collapse. Each layer can then build
on the one before: conv1 spots simple patterns, conv2 combines them into
bigger ones, and conv3 into bigger ones still.

It also acts like an on/off switch: a negative value means "this pattern isn't
here", so it becomes 0.

### Dead ReLUs

A value that's switched off (negative → 0) also has a gradient of 0: nudging
its weights a little changes nothing, so backpropagation can't move them. If a
kernel's output ends up negative for **every** board, that kernel stops
learning for good: a **dead ReLU**. Batchnorm before ReLU makes this rarer,
since it keeps values centred around 0.

## tanh

The network's very last step is **tanh**, which squishes the score into
**-1 to +1**: -1 black winning, 0 even, +1 white winning.

```
tanh(0)  = 0.00
tanh(1)  = 0.76
tanh(2)  = 0.96
tanh(3)  = 0.995
```

tanh changes fast near 0 and flattens out past about ±2. That flat part
matters for training: its gradient (`1 - tanh²`) is close to 0 there, so a
score stuck near ±1 learns slowly. That's why the targets are scaled with
`tanh(cp / CP_SCALE)`, keeping most positions in the part that still changes:
see [loss_function.md](loss_function.md).

## Other activation functions

| Function | Range | Used for |
|---|---|---|
| **ReLU** | 0 to ∞ | between layers (ours) |
| **tanh** | -1 to +1 | a final score (ours) |
| **sigmoid** | 0 to 1 | a final probability |

ReLU is the standard choice between layers because it trains faster in deep
networks than tanh or sigmoid: it never flattens out for positive values.

## Further reading

- Krizhevsky, Sutskever & Hinton (2012), [*ImageNet Classification with Deep
  Convolutional Neural Networks*][alexnet], section 3.1 ("ReLU Nonlinearity"):
  the paper that made ReLU mainstream, with a graph of how much faster it
  trains than tanh.
- Glorot, Bordes & Bengio (2011), [*Deep Sparse Rectifier Neural
  Networks*][glorot]: why ReLU works well in deep networks.
- Goodfellow, Bengio & Courville, [*Deep Learning*][dlbook], section 6.3.1
  ("Rectified Linear Units and Their Generalizations").

[alexnet]: https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html
[glorot]: https://proceedings.mlr.press/v15/glorot11a.html
[dlbook]: https://www.deeplearningbook.org/
