# Policy and value

The network's two outputs, from the two heads
([NeuralNetwork.md](NeuralNetwork.md#the-heads)):

| Head | Answers | Output | Used by search to |
|---|---|---|---|
| **value** | who's winning here? | one score, -1 .. +1 | score the positions it reaches |
| **policy** | which moves are worth trying? | 4096 move scores | pick which moves to explore |

From [`model/network.py`](../../src/chessbot/model/network.py):

```python
        value = nnx.relu(self.value1(x))  # (n, HIDDEN)
        value = self.value2(value)        # (n, 1): the score
        value = jnp.tanh(value)[:, 0]     # (n,): squished into -1 (black winning) .. +1 (white winning)

        policy = self.policy(x)  # (n, 4096): raw move scores (logits)
        return policy, value
```

## Value: tanh

**tanh** squishes any number into -1 .. +1, so the score always has the same
meaning: +1 white is winning, -1 black is, 0 even. Training targets are put in
the same range ([training](Training.md#targets)).

## Policy: one score per move

A layer has a fixed number of outputs, and each output has its own weights, so
it must mean the same move every time. But legal moves change with every
position. So the head outputs a score for **every possible** move, always in
the same order, and we pick out the legal ones afterwards.

We number moves by **from-square x to-square**: 64 x 64 = 4096. A move's
score is at index `from_square * 64 + to_square`, e.g. e2→e4 is
12 x 64 + 28 = 796. Most of the 4096 are illegal in any given position.

```python
def move_index(move: chess.Move) -> int:
    return move.from_square * 64 + move.to_square
```

Everything that reads or writes the policy goes through
[`move_index`](../../src/chessbot/engine/moves.py), so the order is defined
in one place.

From-square x to-square can't tell promotions apart (e7→e8 is one index for
queen, rook, bishop and knight), so only queen promotions are used.

These raw scores are **logits**: any size, not yet probabilities.

## Softmax: scores to probabilities

**Softmax** turns logits into probabilities that add up to 1: `exp` makes every
score positive, dividing by the sum makes them add to 1. A higher score means a
higher probability.

```
p_i = exp(s_i) / Σ_j exp(s_j)
```

Taking the softmax over only the legal moves' scores gives illegal moves no
probability at all. That's
[`engine/moves.py`](../../src/chessbot/engine/moves.py):

```python
    legal = [move for move in board.legal_moves if move.promotion in (None, chess.QUEEN)]
    scores = policy[[move_index(move) for move in legal]]

    exps = np.exp(scores - scores.max())
    probs = exps / exps.sum()
```

Subtracting the largest score first keeps `exp` from overflowing and doesn't
change the result. The probabilities are the **priors** MCTS uses: how
promising each move looks before searching it ([MCTS](MCTS.md)).

## The reverse: a training target

Training goes the other way: `{move: probability}` → a 4096 array laid out like
the policy head, 0 for every other move. That's `policy_target`:

```python
    target = np.zeros(NUM_MOVES, dtype=np.float32)
    target[[move_index(move) for move in shares]] = list(shares.values())
```

It uses the same `move_index` as `legal_priors`, or the network would learn
scores for the wrong moves.

## Further reading

- Silver et al. (2017), [*Mastering Chess and Shogi by Self-Play*][alphazero],
  "Domain Knowledge" and Table S2: AlphaZero's policy for chess, 8 x 8 x 73 =
  4672 moves (it also covers underpromotions), with illegal moves masked out.
- Goodfellow, Bengio & Courville, [*Deep Learning*][dlbook], section 6.2.2.3:
  softmax.

[alphazero]: https://arxiv.org/abs/1712.01815
[dlbook]: https://www.deeplearningbook.org/
