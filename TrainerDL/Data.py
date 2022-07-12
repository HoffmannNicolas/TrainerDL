
import random
import torch
import numpy as np

""" A set of methods to simplify dev """

class myDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, Xs, Ys):
        'Initialization'
        self.Xs = Xs
        self.Ys = Ys

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.Ys)

  def __getitem__(self, index):
        'Generates one sample of data'
        X = self.Xs[index]
        Y = self.Ys[index]
        return np.array(X), np.array(Y)


def splitData(xs, ys, trainProp=0.7, validProp=0.2, shuffle=True) :
    if (shuffle) :
        seed = random.random()
        random.Random(seed).shuffle(xs)
        random.Random(seed).shuffle(ys)

    n_data = len(xs)
    i_trainValidSplit = round(n_data * trainProp)
    i_validTestSplit = round(n_data * (trainProp + validProp))
    xs_train = xs[:i_trainValidSplit]
    ys_train = ys[:i_trainValidSplit]
    xs_valid = xs[i_trainValidSplit:i_validTestSplit]
    ys_valid = ys[i_trainValidSplit:i_validTestSplit]
    xs_test = xs[i_validTestSplit:]
    ys_test = ys[i_validTestSplit:]

    params = {
      'batch_size': 3,
      'shuffle': True,
      'num_workers': 2
    }

    # Training DataLoader
    training_dataset = myDataset(xs_train, ys_train)
    training_generator = torch.utils.data.DataLoader(training_dataset, **params)

    # Validation DataLoader
    validation_dataset = myDataset(xs_valid, ys_valid)
    validation_generator = torch.utils.data.DataLoader(validation_dataset, **params)

    # Test DataLoader
    test_dataset = myDataset(xs_test, ys_test)
    test_generator = torch.utils.data.DataLoader(test_dataset, **params)

    return training_dataset, validation_dataset, test_dataset, training_generator, validation_generator, test_generator



if __name__ == "__main__" :
    xs = list(range(0, 10))
    ys = list(range(10, 20))
    print(xs)
    print(ys)

    training_dataset, validation_dataset, test_dataset, training_generator, validation_generator, test_generator = splitData(xs, ys, shuffle=False)

    for batch in training_generator :
        print(batch)
