# Convolution layers

Terms and figures follow *A guide to convolution arithmetic for deep
learning* ([Dumoulin & Visin, 2016][paper]).

## What a convolution is

![A 3x3 kernel sliding over a padded input to build the output](https://raw.githubusercontent.com/vdumoulin/conv_arithmetic/master/gif/same_padding_no_strides.gif)

*Blue: input feature map. Shaded: the patch under the kernel. Green: output
feature map. Figure from [vdumoulin/conv_arithmetic][repo] (MIT licence).*

The **input feature map** is the entire blue grid above, and the shaded blue
area is a **patch**. Each patch is flattened into a **patch vector**.

A **kernel** is a matrix of weights that is applied to each flattened
**patch** vector on the input feature map. Applying it and adding a bias is
an **affine transformation**, which gives one number:

```
output value = W · x + b
               │   │   └─ bias
               │   └───── the flattened patch vector
               └───────── the kernel's weights
```

Applying one kernel to **every** patch vector gives 64 numbers, laid out 8x8:
one **output feature map**. Each kernel makes its own, so 64 kernels give 64
output feature maps.

## Our conv layers

From [`model/network.py`](../../src/chessbot/model/network.py):

```python
        input_maps = output_maps = channels

        self.conv1 = nnx.Conv(NUM_PLANES, output_maps, (3, 3), padding="SAME", rngs=rngs)
        self.conv2 = nnx.Conv(input_maps, output_maps, (3, 3), padding="SAME", rngs=rngs)
        self.conv3 = nnx.Conv(input_maps, output_maps, (3, 3), padding="SAME", rngs=rngs)
```

Each argument in the paper's terms:

| Code | Paper's term | Ours |
|---|---|---|
| first number | input channels (input feature maps) | 18 (`NUM_PLANES`) for conv1, 64 after |
| second number | output channels, one per kernel | 64 (`CHANNELS`) |
| `(3, 3)` | kernel size *k* | 3 |
| `strides` (default) | unit strides, *s* = 1 | moves 1 square at a time |
| `padding="SAME"` | half (same) padding, *p* = 1 | one ring of zeros around the board |

The figure above is exactly our case, **half padding with unit strides**: the
output is the same size as the input, so the board stays 8x8 in every layer.

```
board (8, 8, 18) → conv1 → (8, 8, 64) → conv2 → (8, 8, 64) → conv3 → (8, 8, 64)
```

Each layer's input channels must match the previous layer's output channels,
which is why conv2 and conv3 have `input_maps = output_maps`.

## Many channels

The paper's figures show **one** input feature map and **one** kernel. Ours
has more of both: **each kernel spans all input channels**. conv1's kernels are
3 x 3 x 18 = **162** weights; conv2's and conv3's are 3 x 3 x 64 = **576**.

You can see it in the saved weights' shapes, `(k, k, input channels, kernels)`:

```
conv1/kernel  (3, 3, 18, 64)    64 kernels of 3 x 3 x 18
conv2/kernel  (3, 3, 64, 64)    64 kernels of 3 x 3 x 64
```

## How far each layer sees

One 3x3 conv sees a 3x3 area, two see **5x5**, three see **7x7**: the
**receptive field**. No single number sees the whole 8x8 board, which is one
reason the network ends with linear heads ([NeuralNetwork.md](NeuralNetwork.md#the-heads)).

## Further reading

- [Dumoulin & Visin (2016)][paper]: section 1.1 for discrete convolutions with
  several input feature maps, and section 2.2.1 for half padding, our case.
  For padding and stride in general, chapter 2 and the animations in
  [vdumoulin/conv_arithmetic][repo] explain it extremely well.
- Goodfellow, Bengio & Courville, [*Deep Learning*][dlbook]: section 9.1 for
  the convolution operation, section 9.2 for why local, shared weights work.

[paper]: https://arxiv.org/abs/1603.07285
[repo]: https://github.com/vdumoulin/conv_arithmetic
[dlbook]: https://www.deeplearningbook.org/
