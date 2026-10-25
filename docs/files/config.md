# config.py

[`src/config.py`](../../src/config.py): every setting you might want to tweak,
in one place. The other files import from here.

## Data

| Setting | Default | What |
|---|---|---|
| `ROOT` | repo root | everything else is relative to this |
| `DATA_DIR` | `data/` | dataset, cache and weights (gitignored) |
| `EVALS_FILE` | `data/evals_10M.jsonl` | Lichess positions with Stockfish evaluations |
| `NUM_POSITIONS` | 3,000,000 | how many positions to load and train on |
| `CP_SCALE` | 400 | targets = `tanh(cp / CP_SCALE)`: see [loss function](../concepts/loss_function.md#targets-must-match-our-range) |
| `WEIGHTS_DIR` | `data/weights/` | saved models |

## Model

| Setting | Default | What |
|---|---|---|
| `CHANNELS` | 64 | kernels per conv layer = output feature maps per layer: see [convolution](../concepts/convolution.md) |
| `HIDDEN` | 64 | numbers in the head's hidden linear layer: see [the head](../concepts/neural_network.md#the-head) |

## Training

| Setting | Default | What |
|---|---|---|
| `LEARNING_RATE` | 0 (not set yet) | Adam's step size; 0.001 is the usual start: see [optimizer](../concepts/optimizer.md#settings) |
| `BATCH_SIZE` | 1024 | positions per `train_step`: see [why a batch](../concepts/loss_function.md#why-a-batch-of-positions) |
