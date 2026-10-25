# Docs

- [setup.md](setup.md): install, get the data, run it.

## files/

One page per source file: what it does and what each function does, linking to
the concepts behind it.

- [config.md](files/config.md): every setting
- [encoding.md](files/encoding.md): board → 98 bytes → 18 input feature maps
- [data.md](files/data.md): the Lichess file → packed cache → batches
- [model.md](files/model.md): the network, saving and loading weights
- [training.md](files/training.md): loss, optimizer, one training step

## concepts/

The ideas behind the code, with sources to read further.

- [neural_network.md](concepts/neural_network.md): how the pieces fit together (start here)
- [encoding.md](concepts/encoding.md): input feature maps, one-hot, bitboards
- [convolution.md](concepts/convolution.md): kernels, patches, feature maps
- [batch_normalization.md](concepts/batch_normalization.md)
- [activations.md](concepts/activations.md): ReLU and tanh
- [loss_function.md](concepts/loss_function.md): MSE and the targets
- [back_propagation.md](concepts/back_propagation.md): gradients and gradient descent
- [optimizer.md](concepts/optimizer.md): Adam
