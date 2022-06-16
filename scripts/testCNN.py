
import os
import sys
# Adds higher directory to python modules path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from TrainerDL.Network.CNN import ConvolutionnalBlock, FeatureExtractor, CNN, ReshapeCNN
import random
import optuna

print("Test convblock")
convBlock = ConvolutionnalBlock(prior_n_channels=3, posterior_n_channels=6, n_convolutions=2, poolingKernel=(2, 2))
print(convBlock)
batch = convBlock.generateRandomBatch()
print(batch.shape)
res = convBlock(batch)
print(res.shape)

print("\n\nTest resizing")
reshape = ReshapeCNN(width=res.shape[2], height=res.shape[3], n_channels=res.shape[1], n_outputs=128)
print(reshape)
res = reshape(res)
print(res.shape)

batch = reshape.generateRandomBatch()
print(batch.shape)
res = reshape(batch)
print(res.shape)


print("\n\nTest Feature extractor Sampling")
study = optuna.create_study(direction='minimize')
def objective(trial) :
    fe = FeatureExtractor.sample(trial)
    # print(fe)
    return fe.n_convBlocks
study.optimize(objective, n_trials=3)


print("\n\nTest CNN")
cnn = CNN(
        width=8,
        height=8,
        n_channels=3,
        n_convPerBlock=1,
        n_convBlocks=10,
        n_convFeatures=256,
        n_outputs=64,
        n_hidden=5,
        shape=2.5,
        dropoutRate=0.5
)
print(cnn)
batch = cnn.generateRandomBatch()
print(f"Batch.shape : {batch.shape}")
print(f"First element of batch : {batch[0]}")
results = cnn(batch)
print(f"Results.shape = {results.shape}")
print(f"First result : {results[0]}")


print("\n\nTest Sampling")
study = optuna.create_study(direction='minimize')
def objective(trial) :
    cnn = CNN.sample(trial)
    # print(cnn)
    return cnn.shape
study.optimize(objective, n_trials=3)
