
import os
import sys
# Adds higher directory to python modules path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from TrainerDL.Network.MultiLayerPerceptron import MultiLayerPerceptron
import random

for _ in range(5) :
    shape = random.uniform(0.1, 10)
    n_hidden = random.randint(0, 10)
    n_inputs = random.randint(4, 64)
    n_outputs = random.randint(4, 64)
    MLP = MultiLayerPerceptron(shape=shape, n_hidden=n_hidden, n_inputs=n_inputs, n_outputs=n_outputs)
    print(MLP)

MLP = MultiLayerPerceptron(shape=1, n_hidden=5, n_inputs=16, n_outputs=48)
print(MLP)
batch = MLP.generateFakeBatch()
print(f"Batch.shape : {batch.shape}")
print(f"First element of batch : {batch[0]}")
results = MLP(batch)
print(f"Results.shape = {results.shape}")
print(f"First result : {results[0]}")
