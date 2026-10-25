# Shortcuts: `make data DATASET=lichess_evals`, `make train METHOD=supervised`. No need to activate a venv first.
# Uses the repo's own .venv (scripts/install.sh) if it exists, else the shared one a folder up.
PYTHON ?= $(or $(abspath $(firstword $(wildcard .venv/bin/python ../.venv/bin/python))),python3)

# What each flag can be: a file in src/datasets/ or src/training/.
DATASETS := $(filter-out __init__,$(basename $(notdir $(wildcard src/datasets/*.py))))
METHODS := $(filter-out __init__,$(basename $(notdir $(wildcard src/training/*.py))))
RUN = PYTHONPATH=src $(PYTHON) -m

.PHONY: data train

# Prepare a dataset (e.g. pack it into a cache in data/).
data:
	@test -n "$(DATASET)" || { echo "Set DATASET to one of: $(DATASETS)"; exit 1; }
	@test -f src/datasets/$(DATASET).py || { echo "No dataset '$(DATASET)'. Choose from: $(DATASETS)"; exit 1; }
	$(RUN) datasets.$(DATASET)

# Train a model with a method.
train:
	@test -n "$(METHOD)" || { echo "Set METHOD to one of: $(METHODS)"; exit 1; }
	@test -f src/training/$(METHOD).py || { echo "No method '$(METHOD)'. Choose from: $(METHODS)"; exit 1; }
	$(RUN) training.$(METHOD)
