# chess-bot

A work-in-progress chess AI, built from scratch in [JAX](https://github.com/jax-ml/jax) and [Flax](https://github.com/google/flax).

**Play online at:** coming soon

## Plans

A set of models, each trained a different way, all playable online:

- **Distilled:** taught from other chess engines' evaluations in public datasets.
- **Self-play:** learns only by playing against itself, with no human or engine knowledge.
- **Human games:** learns purely from games people play against it online.

## Models

Trained weights for each model and version will be published under [Releases](https://github.com/s9oon/chess-bot/releases).

## Documentation

For setup, development, training and more, see [`docs/`](docs/).
