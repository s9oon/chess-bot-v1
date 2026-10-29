"""The chess bot. Each folder is one part, importing only from the ones before it:

    encoding  ->  model  ->  datasets  ->  training
    encoding  ->  mcts   ------------->  training
    (search never imports the model: training gives it an evaluate function)

encoding  board <-> numbers (input planes, move indexes). No ML.
model     the network and saving/loading its weights.
mcts      Monte Carlo tree search, guided by the network's policy and value.
datasets  positions to learn from.
training  ways to train the model.
"""
