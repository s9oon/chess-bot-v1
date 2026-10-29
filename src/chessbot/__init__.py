"""The chess bot. Each folder is one part, importing only from the ones before it:

    engine  ->  model  ->  datasets  ->  training

engine    playing: board encoding, move indexes, MCTS. No JAX, no model: a port reimplements this.
model     the JAX network, its input planes, saving/loading its weights.
datasets  positions to learn from.
training  ways to train the model. Self-play also runs the model for search (its make_evaluate).
"""
