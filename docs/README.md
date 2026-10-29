# Docs

- [setup.md](setup.md): install, get the data, run it.

## concepts/

The ideas behind the code, each shown with the code that does it, and sources
to read further.

- [neural_network.md](concepts/neural_network.md): how the pieces fit together (start here)
- [board_encoding.md](concepts/board_encoding.md): board → input feature maps (`encoding/planes.py`)
- [convolution.md](concepts/convolution.md): kernels, patches, feature maps (`model/network.py`)
- [batch_normalization.md](concepts/batch_normalization.md) (`model/network.py`)
- [policy_and_value.md](concepts/policy_and_value.md): the two outputs, move indexes, softmax (`model/network.py`, `encoding/moves.py`)
- [training.md](concepts/training.md): targets, loss, gradients, Adam, batches (`training/`, `datasets/`)
