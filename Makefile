# Shortcuts: `make data DATASET=lichess_evals`, `make train METHOD=supervised`. No need to activate a venv first.
# Uses the repo's own .venv (scripts/install.sh) if it exists, else the shared one a folder up.
PYTHON ?= $(or $(abspath $(firstword $(wildcard .venv/bin/python ../.venv/bin/python))),python3)

# What each flag can be: a file in src/chessbot/datasets/ or src/chessbot/training/ (batches.py is a helper, not a method).
PKG := src/chessbot
DATASETS := $(filter-out __init__,$(basename $(notdir $(wildcard $(PKG)/datasets/*.py))))
METHODS := $(filter-out __init__ batches,$(basename $(notdir $(wildcard $(PKG)/training/*.py))))
RUN = PYTHONPATH=src $(PYTHON) -m

.PHONY: data train

# Prepare a dataset (e.g. pack it into a cache in data/).
data:
	@test -n "$(DATASET)" || { echo "Set DATASET to one of: $(DATASETS)"; exit 1; }
	@test -f $(PKG)/datasets/$(DATASET).py || { echo "No dataset '$(DATASET)'. Choose from: $(DATASETS)"; exit 1; }
	$(RUN) chessbot.datasets.$(DATASET)

# Train a model with a method.
train:
	@test -n "$(METHOD)" || { echo "Set METHOD to one of: $(METHODS)"; exit 1; }
	@test -f $(PKG)/training/$(METHOD).py || { echo "No method '$(METHOD)'. Choose from: $(METHODS)"; exit 1; }
	$(RUN) chessbot.training.$(METHOD)
