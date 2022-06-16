
import os
import sys
# Adds higher directory to python modules path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from TrainerDL.Network.MLP import MLP
import random
import optuna

for _ in range(5) :
    shape = random.uniform(0.1, 10)
    n_hidden = random.randint(0, 10)
    n_inputs = random.randint(4, 64)
    n_outputs = random.randint(4, 64)
    mlp = MLP(n_inputs=n_inputs, n_outputs=n_outputs, n_hidden=n_hidden, shape=shape)
    print(mlp)

mlp = MLP(n_inputs=16, n_outputs=48, n_hidden=5, shape=1)
print(mlp)
batch = mlp.generateFakeBatch()
print(f"Batch.shape : {batch.shape}")
print(f"First element of batch : {batch[0]}")
results = mlp(batch)
print(f"Results.shape = {results.shape}")
print(f"First result : {results[0]}")

print("\n\nTest Sampling")
study = optuna.create_study(direction='minimize')
def objective(trial) :
    mlp = MLP.sample(trial)
    # print(mlp)
    return mlp.shape
study.optimize(objective, n_trials=3)
