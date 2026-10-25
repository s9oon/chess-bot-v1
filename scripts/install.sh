#!/usr/bin/env bash
# Set up chess-bot: a virtual environment with every package in requirements.txt.
# Usage (from anywhere):  bash scripts/install.sh
# Use an existing venv instead:  VENV=../.venv bash scripts/install.sh
set -euo pipefail  # stop at the first error

cd "$(dirname "$0")/.."  # repo root, wherever this was run from
VENV="${VENV:-.venv}"

if [ ! -d "$VENV" ]; then
    echo "Creating virtual environment in $VENV"
    python3 -m venv "$VENV"
fi

"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install -r requirements.txt

mkdir -p data checkpoints

echo
echo "Installed. Next steps:"
if [ ! -f data/evals_10M.jsonl ]; then
    # Lichess's Stockfish eval database; head stops the download after 10M positions.
    echo "  Get the data (needs zstd):"
    echo "    curl -L https://database.lichess.org/lichess_db_eval.jsonl.zst | zstd -dc | head -n 10000000 > data/evals_10M.jsonl"
fi
if ! command -v make > /dev/null; then
    echo "  Optional, for make data / make train:  sudo apt install make"
fi
echo "  make data    (or $VENV/bin/python data.py)"
echo "  make train   (or $VENV/bin/python training.py)"
