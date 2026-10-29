# Docs

- [Setup.md](Setup.md): install, get the data, run it.

## concepts/

The ideas behind the code, each shown with the code that does it, and sources
to read further.

- [NeuralNetwork.md](concepts/NeuralNetwork.md): how the pieces fit together (start here)
- [BoardEncoding.md](concepts/BoardEncoding.md): board → input feature maps (`encoding/planes.py`)
- [Convolution.md](concepts/Convolution.md): kernels, patches, feature maps (`model/network.py`)
- [BatchNormalization.md](concepts/BatchNormalization.md) (`model/network.py`)
- [PolicyAndValue.md](concepts/PolicyAndValue.md): the two outputs, move indexes, softmax (`model/network.py`, `encoding/moves.py`)
- [MCTS.md](concepts/MCTS.md): tree search with the policy and value, PUCT, visit counts (`mcts/search.py`)
- [Training.md](concepts/Training.md): targets, loss, gradients, Adam, batches (`training/`, `datasets/`)
- [SelfPlay.md](concepts/SelfPlay.md): learning from its own games, π and z, temperature, buffer, the two-part loss (`training/self_play.py`)
