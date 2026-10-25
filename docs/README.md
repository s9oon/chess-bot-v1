# Docs

- [setup.md](setup.md): install, get the data, run it.

## files/

One page per source file: what it does and what each function does, linking to
the concepts behind it.

- [config.md](files/config.md): every setting
- [encoding.md](files/encoding.md): board → 98 bytes → 18 input feature maps
- [sources/](files/sources/README.md): where training data comes from, one file per kind of data
  - [lichess_evals.md](files/sources/lichess_evals.md): the Lichess evals file → packed cache
  - [common.md](files/sources/common.md): batches, shared by every source
- [model.md](files/model.md): the network, saving and loading weights
- [training.md](files/training.md): loss, optimizer, one training step

## concepts/

The ideas behind the code, with sources to read further.

- [neural_network.md](concepts/neural_network.md): how the pieces fit together (start here)
- [convolution.md](concepts/convolution.md): kernels, patches, feature maps
- [batch_normalization.md](concepts/batch_normalization.md)
