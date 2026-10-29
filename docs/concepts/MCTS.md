# Monte Carlo tree search (MCTS)

The network gives a first guess: which moves look good (policy) and who's
winning (value) ([policy and value](PolicyAndValue.md)). **Search** improves on
that guess by looking ahead: it builds a **tree** of positions, one move per
branch, and spends its time on the moves that look most promising.

Ours is **PUCT**, the version AlphaZero uses, in
[`engine/search.py`](../../src/chessbot/engine/search.py).

## The tree

Each **node** is a position, reached by one move from its parent:

```python
class Node:
    def __init__(self, prior: float):
        self.prior = prior    # the policy's probability for the move into this node
        self.visits = 0       # how many simulations went through it
        self.value_sum = 0.0  # their values, from the view of the player who made the move into it
        self.children: dict[chess.Move, Node] = {}  # empty until expanded
```

**Q** is a node's average value so far, `value_sum / visits`: how good that
move has turned out to be.

## One simulation

Search runs `NUM_SIMULATIONS` of these (100, in `config.py`). Each adds one new
position to the tree:

```python
        # 1. Select: walk down through expanded nodes, pushing each move onto the board.
        node, path = root, [root]
        while node.children:
            move, node = select_child(node)
            board.push(move)
            path.append(node)

        # 2. Expand: the network scores the new position and gives its moves' priors.
        value = expand(node, board, evaluate)

        # 3. Backup: pass the value up the path, then undo the moves.
        backup(path, value)
```

1. **Select:** from the root, keep choosing a child until you reach a position
   that hasn't been expanded yet.
2. **Expand:** run the network once on that position. Its **value** scores the
   position, and its **policy**, through `legal_priors`, becomes the priors of
   its new children. A finished game needs no network: checkmate is -1 for the
   side to move, anything else is a draw, 0.
3. **Backup:** add the value to every node on the path back to the root.

## Choosing a child: PUCT

```
score = Q + C_PUCT * prior * sqrt(parent visits) / (1 + visits)
        │   └──────────────────── U ─────────────────────────┘
        └ exploitation: how good it's been so far
```

```python
        return child.q() + C_PUCT * child.prior * sqrt_visits / (1 + child.visits)
```

- **Q** pulls search towards moves that have scored well: **exploitation**.
- **U** pulls it towards moves the policy likes but search hasn't tried much:
  **exploration**. It shrinks as a move gets visited, so no move is trusted on
  its prior alone for long.

`C_PUCT` (1.25) sets the balance. Higher trusts the priors more, lower trusts
the values found so far more.

## Whose view: flipping the sign

A position that's good for white is bad for black by the same amount. So on the
way back up, the value's sign flips every ply:

```python
    for node in reversed(path):
        value = -value
        node.visits += 1
        node.value_sum += value
```

Each node stores values from the view of the player who **made the move into
it**. That way, when a player chooses among their children, a higher Q always
means better for them.

The value head scores from white's view, so whatever runs the network for search
must flip it when black is to move: the value given to search is always the
side to move's view.

## The result: visit counts

```python
    visits, value = search(board, evaluate)   # {move: visit count}, value for the side to move
```

The move to play is the one with the **most visits**, not the highest Q. Visits
are what search settled on after weighing everything, while a high Q on a few
visits can be luck. The visit counts, turned into probabilities, are also the
policy's training target in self-play: search's improved version of the
network's guess.

The counts add up to one less than the number of simulations, because the first
simulation expands the root itself.

Search only needs `evaluate(board) -> (priors, value)`, so it doesn't know
whether a JAX model, an ONNX file or anything else is behind it. Self-play
gives it one that runs the JAX model ([self-play](SelfPlay.md#playing-the-model-to-search)).

## Root noise

In self-play, every game would start the same way if search always trusted the
policy. So when given an `rng`, search mixes random **Dirichlet noise** into
the root's priors, before any other simulation:

```python
    noise = rng.dirichlet([DIRICHLET_ALPHA] * len(node.children))
    for child, n in zip(node.children.values(), noise):
        child.prior = (1 - NOISE_FRACTION) * child.prior + NOISE_FRACTION * n
```

Dirichlet noise is a random set of probabilities that add up to 1. With
`DIRICHLET_ALPHA = 0.3` most of it lands on a few moves, so each game search
takes a closer look at a few moves the policy would have overlooked. The priors
become 75% policy, 25% noise. Only the root gets noise, and only in self-play.

Even with an untrained network, 200 simulations find a mate in one: the mating
move scores +1 every time it's visited, so search keeps choosing it.

## Further reading

- Silver et al. (2017), [*Mastering the game of Go without human
  knowledge*][alphago-zero], "Search algorithm" in Methods: select, expand and
  back up, with the PUCT formula.
- Silver et al. (2017), [*Mastering Chess and Shogi by Self-Play*][alphazero]:
  the same search for chess.
- Browne et al. (2012), [*A Survey of Monte Carlo Tree Search
  Methods*][survey], section 3: MCTS in general, before neural networks.

[alphago-zero]: https://www.nature.com/articles/nature24270
[alphazero]: https://arxiv.org/abs/1712.01815
[survey]: https://ieeexplore.ieee.org/document/6145622
