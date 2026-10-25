# Setup

Needs Linux or WSL and an NVIDIA GPU.

```bash
git clone https://github.com/s9oon/chess-bot.git
cd chess-bot
bash scripts/install.sh
```

This creates `.venv`, installs `requirements.txt`, and prints the command to download the data: the first 10M positions of the [Lichess evaluation database](https://database.lichess.org/#evals), saved as `data/evals_10M.jsonl`.

## Running

```bash
make data    # pack positions into data/packed_<N>.npz (slow once, then loads in under a second)
make train   # train the model
```

Without `make`: `.venv/bin/python src/data.py` and `.venv/bin/python src/training.py`.

Settings are in `src/config.py`.
