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

Applying one kernel to **every** patch vector gives one number per patch.
Laid out 8x8, those numbers are one **output feature map**:

```
kernel 0 · patch 0  + b  ->  1 number
kernel 0 · patch 1  + b  ->  1 number
...
kernel 0 · patch 63 + b  ->  1 number
                             64 numbers = output feature map 0 (8x8)
```

Each kernel makes its own output feature map, so 64 kernels give 64 output
feature maps. That whole process is the convolution.

Two things make it a convolution rather than a normal (fully connected)
layer:

- **Local:** each output only looks at a small patch, not the whole input.
- **Shared weights:** the same kernel is used at every position.

## Our settings, in the paper's terms

| Code | Paper's term | Ours |
|---|---|---|
| first number | input channels (input feature maps) | 18 for conv1, 64 after |
| second number | output channels, one per kernel | 64 (`CHANNELS`) |
| `(3, 3)` | kernel size *k* | 3 |
| `strides` (default) | unit strides, *s* = 1 | moves 1 square at a time |
| `padding="SAME"` | half (same) padding, *p* = 1 | one ring of zeros around the board |

The paper's figure above is exactly our case: **half padding, unit strides**,
where the output is the same size as the input.

## Padding and stride

For padding and stride, refer to the animations in
[vdumoulin/conv_arithmetic][repo] and chapter 2 of [the paper][paper]. They
explain it extremely well. Our case, half padding, is section 2.2.1.

We use stride 1 and `padding="SAME"`, so the board stays 8x8 in every layer.

## Many channels

The paper's figures show **one** input feature map and **one** kernel. Ours
has more of both:

- **Each kernel spans all input channels.** conv1's kernels are 3 x 3 x 18 =
  **162** weights; conv2's and conv3's are 3 x 3 x 64 = **576**.
- **Every kernel is applied to every patch.** 64 kernels give 64 output
  feature maps.

As one matrix multiply, for conv1:

```
patches (64, 162)  @  kernels (162, 64)  =  output (64, 64)  ->  (8, 8, 64)
   │                     │
   │                     └─ one column per kernel
   └─ one row per position (8 x 8), each a flattened 3x3x18 patch
```

## Shapes through the layers

```
board   (8, 8, 18)
  ↓ conv1   18 -> 64
        (8, 8, 64)
  ↓ conv2   64 -> 64
        (8, 8, 64)
  ↓ conv3   64 -> 64
        (8, 8, 64)
```

Each layer's input channels must match the previous layer's output channels.
That's why conv2 and conv3 have `input_maps = output_maps`.

## How far each layer sees

One 3x3 conv sees a 3x3 area. Stack a second one and each of its outputs reads
3x3 values that each saw 3x3, so it sees **5x5**. A third sees **7x7**. This
area is called the **receptive field**.

So after three layers no single number sees the whole 8x8 board. That's one
reason the network ends with a linear head: see
[neural_network.md](neural_network.md).

## How many weights

```
conv1   3 x 3 x 18 x 64  + 64 biases  =  10,432
conv2   3 x 3 x 64 x 64  + 64 biases  =  36,928
conv3   3 x 3 x 64 x 64  + 64 biases  =  36,928
```

Compare a linear layer from the same `(8, 8, 64)` input to the same output size:
4,096 x 4,096 = **16.7 million** weights. Sharing one small kernel across every
position is what keeps convs small.

After each conv layer comes batch normalization, then ReLU: see
[batch_normalization.md](batch_normalization.md) and
[activations.md](activations.md).

## Further reading

- [Dumoulin & Visin (2016)][paper]: section 1.1 for discrete convolutions with
  several input feature maps, and section 2.2.1 for half padding with unit
  strides, our case.
- Goodfellow, Bengio & Courville, [*Deep Learning*][dlbook], chapter 9: section
  9.1 for the convolution operation, and section 9.2 for why local
  connections and shared weights work (sparse interactions, parameter sharing).

[paper]: https://arxiv.org/abs/1603.07285
[repo]: https://github.com/vdumoulin/conv_arithmetic
[dlbook]: https://www.deeplearningbook.org/
