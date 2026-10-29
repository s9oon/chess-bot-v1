"""The engine: everything needed to play, with no JAX or model code. A port to another language
reimplements exactly these files.

encoding.py  board -> 98 packed bytes (the network's input format)
moves.py     the policy's 4096 outputs <-> legal moves
search.py    MCTS, calling whatever `evaluate(board) -> (priors, value)` it's given
"""
