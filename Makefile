# Shortcuts: `make data`, `make train`. No need to activate a venv first.
# Uses the repo's own .venv (scripts/install.sh) if it exists, else the shared one a folder up.
PYTHON ?= $(or $(abspath $(firstword $(wildcard .venv/bin/python ../.venv/bin/python))),python3)

.PHONY: data train

# Build (or check) the packed position cache in data/.
data:
	$(PYTHON) src/data.py

# Train the model.
train:
	$(PYTHON) src/training.py