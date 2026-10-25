# chess-bot

A work-in-progress chess AI, built from scratch in [JAX](https://github.com/jax-ml/jax) and [Flax](https://github.com/google/flax).

The plan is a set of models, each trained a different way:

- **Distilled:** taught from other chess engines' evaluations in public datasets.
- **Self-play:** learns only by playing against itself, with no human or engine knowledge.
- **Human games:** learns purely from games people play against it on a public site (link coming soon).

## Setup

Needs Linux or WSL and an NVIDIA GPU.

```bash
git clone https://github.com/s9oon/chess-bot.git
cd chess-bot
bash scripts/install.sh
```

Settings are in `config.py`.

## Documentation

For development, training and more, see [`docs/`](docs/).
