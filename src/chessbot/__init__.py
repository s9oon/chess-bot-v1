"""The chess bot. Each folder is one part, importing only from the ones before it:

    encoding  ->  model  ->  datasets, training

encoding  board <-> numbers (input planes, move indexes). No ML.
model     the network and saving/loading its weights.
datasets  positions to learn from.
training  ways to train the model.
"""
