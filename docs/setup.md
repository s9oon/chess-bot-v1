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
make data DATASET=lichess_evals    # prepare a dataset (pack it into data/, slow once)
make train METHOD=supervised       # train with a method
```

`DATASET` is a file in `src/datasets/`, `METHOD` a file in `src/training/`;
leave either out to see the list.

Without `make`: `PYTHONPATH=src .venv/bin/python -m datasets.lichess_evals` and
`PYTHONPATH=src .venv/bin/python -m training.supervised`.

Settings are in `src/config.py`.
