# Self-play

Learning only from games against itself, no human or engine knowledge: the
AlphaZero method. In
[`training/self_play.py`](../../src/chessbot/training/self_play.py), run with
`make train METHOD=self_play`.

## The loop

Search with the network plays better than the network alone
([MCTS](MCTS.md)). So the network learns to match search, search gets better
with the better network, and so on:

```
model → play games with search → (board, π, z) per position → train on them → better model → ...
```

Each **round** does both halves:

```python
        # 1. Play. Batchnorm uses its running averages, so each board's score stands alone.
        model.eval()
        for _ in range(GAMES_PER_ROUND):
            boards, pis, result = play_game(evaluate, rng)
            buffer.add(boards, pis, result)

        # 2. Learn from random positions out of the recent games.
        model.train()
        for _ in range(STEPS_PER_ROUND):
            value_loss, policy_loss = train_step(model, optimizer, *buffer.sample(BATCH_SIZE, rng))
```

The weights are saved to `data/weights/self_play.npz` after every round, and
the next run carries on from them.

## Playing: the model to search

Search only calls `evaluate(board) -> (priors, value)`. `make_evaluate` builds
that from the JAX model being trained:

```python
        policy, value = forward(model, jnp.asarray(pack_board(board)[None]))  # a batch of one
        priors = legal_priors(np.asarray(policy[0]), board)
        white_view = float(value[0])
        return priors, white_view if board.turn == chess.WHITE else -white_view
```

It's the same `model` object training updates, so every game uses the latest
weights.

## The targets: π and z

For every position in a game, `play_game` stores:

| | What | Trains |
|---|---|---|
| board | the position, packed | input |
| **π** | search's visit shares: each move's visits / total visits | policy head |
| **z** | the game's result, white's view: +1 white won, -1 black won, 0 draw | value head |

```python
        shares /= shares.sum()
        pis.append(policy_target(dict(zip(moves, shares))))
```

`policy_target` lays the shares out like the policy head
([the reverse of legal_priors](PolicyAndValue.md#the-reverse-a-training-target)).
z is only known when the game ends, so every position in a game gets the same
z. It's white's view, like the value head, so no flipping is needed.

## Choosing moves: temperature

For the first `TEMPERATURE_PLIES` (30) plies, a move is picked at random **in
proportion to its visits**, so games open differently. After that, the most
visited move is played:

```python
        if board.ply() < TEMPERATURE_PLIES:
            move = moves[rng.choice(len(moves), p=shares)]
        else:
            move = moves[int(shares.argmax())]
```

Together with [root noise](MCTS.md#root-noise), this keeps the games varied,
so the network sees more than one line.

Early on the network is random and games can drag on, so a game stops at
`MAX_GAME_PLIES` (300) and counts as a draw.

## The buffer

Training samples random positions from the most recent `BUFFER_SIZE` (20,000),
not just the last game:

- positions from the same game are very alike, and training on them in a row
  pulls the network towards that one game;
- old games were played by a weaker network, so they're dropped as new ones
  arrive.

## The loss

Two parts, one per head, added together:

```python
    value_loss = jnp.mean((zs - values) ** 2)
    policy_loss = -jnp.mean(jnp.sum(pis * jax.nn.log_softmax(logits), axis=-1))
    return value_loss + policy_loss, (value_loss, policy_loss)
```

- **Value:** mean squared error against z, as in [training](Training.md#loss).
- **Policy:** **cross-entropy** against π: `-Σ π · log p`, where p is the
  softmax of the policy's scores. It's smallest when p matches π. Illegal
  moves have π = 0, so the network learns to give them low scores too.

One `train_step` backpropagates both through their heads and the shared body
([gradients](Training.md#gradients-and-gradient-descent)).

## Speed

About 0.5 s per move at 100 simulations on a GTX 1660, so a 300-ply game takes
a few minutes. AlphaZero played 44 million games on 5,000 TPUs; expect slow
progress here. Fewer simulations (`NUM_SIMULATIONS`) means faster, weaker games.

## Further reading

- Silver et al. (2017), [*Mastering the game of Go without human
  knowledge*][alphago-zero], "Self-play training pipeline" and "Self-play" in
  Methods: π, z, temperature, Dirichlet noise and the loss.
- Silver et al. (2017), [*Mastering Chess and Shogi by Self-Play*][alphazero]:
  the same method for chess, with Dirichlet α = 0.3.
- Goodfellow, Bengio & Courville, [*Deep Learning*][dlbook], section 3.13:
  cross-entropy.

[alphago-zero]: https://www.nature.com/articles/nature24270
[alphazero]: https://arxiv.org/abs/1712.01815
[dlbook]: https://www.deeplearningbook.org/
