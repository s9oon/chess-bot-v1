# config.py

[`src/config.py`](../../src/config.py): every setting in one place. The other
files import from here.

## Data

| Setting | Default | What |
|---|---|---|
| `ROOT` | repo root | everything else is relative to this |
| `DATA_DIR` | `data/` | datasets, caches and weights (gitignored) |
| `WEIGHTS_DIR` | `data/weights/` | saved models |
| `EVALS_FILE` | `data/evals_10M.jsonl` | [Lichess evals](sources/lichess_evals.md) only: the file to read |
| `NUM_POSITIONS` | 3,000,000 | Lichess evals only: how many positions to load |
| `CP_SCALE` | 400 | Lichess evals only: targets = `tanh(cp / CP_SCALE)`, 400 cp (4 pawns) → 0.76 |

## Model

| Setting | Default | What |
|---|---|---|
| `CHANNELS` | 64 | kernels per conv layer = output feature maps per layer ([convolution](../concepts/convolution.md)) |
| `HIDDEN` | 64 | numbers in the head's hidden linear layer ([the head](../concepts/neural_network.md#the-head)) |

## Training

| Setting | Default | What |
|---|---|---|
| `LEARNING_RATE` | 0 (not set yet) | Adam's step size; 0.001 is the usual start |
| `BATCH_SIZE` | 1024 | positions per `train_step` |
