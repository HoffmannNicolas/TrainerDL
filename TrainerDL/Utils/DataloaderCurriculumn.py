
import torch
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler, WeightedRandomSampler
from typing import Iterator, List

class CurriculumBatchSampler(BatchSampler) :
    # Source : https://github.com/pytorch/pytorch/blob/master/torch/utils/data/sampler.py
    def __init__(self, batch_size: int = 16, drop_last: bool = False, num_data: int = 32) -> None:
        self.numBatches = 0
        self.num_data = num_data
        self.samplingWeights = [1e3] * num_data # High initial expected loss to nudge the sampler to test all data at least once
        self.indexOccurance = [0] * num_data
        self.lastPicked = [0] * num_data
        sampler = WeightedRandomSampler(self.samplingWeights, batch_size, replacement=True)
        super().__init__(sampler, batch_size, drop_last)

    def __iter__(self) -> Iterator[List[int]]:
        for batch in super().__iter__() :
            self.indicesChosen = list(batch)
            for index in self.indicesChosen :
                self.indexOccurance[index] += 1
                self.lastPicked[index] = self.numBatches
            self.numBatches += 1
            yield self.indicesChosen

    def updateSamplingWeights(self, losses) :
        assert len(losses) == len(self.indicesChosen)
        for (index, loss) in zip(self.indicesChosen, losses) :
            self.samplingWeights[index] = loss
        self.sampler.weights = torch.as_tensor(self.samplingWeights, dtype=torch.double)


if __name__ == "__main__" :
    bs = 2
    numData = 100

    curriculumBatcher = CurriculumBatchSampler(batch_size=bs, drop_last=True, num_data=numData)

    print("Testing Batch Sampler")
    print(curriculumBatcher.samplingWeights)
    print(curriculumBatcher.indexOccurance)

    for batchIndex, batch in enumerate(curriculumBatcher) :
        print("batch ", batchIndex,  ":", list(batch))
        curriculumBatcher.updateSamplingWeights([0.1] * bs)
    print(curriculumBatcher.samplingWeights)
    print(curriculumBatcher.indexOccurance)

    print("Testing Dataloader")

    from torch.utils.data import Dataset
    import random
    class MyLittleDataset(Dataset):
        def __init__(self, Xs, Ys):
            self.Xs = Xs
            self.Ys = Ys

        def __len__(self):
            return len(self.Xs)

        def __getitem__(self, index):
            return Xs[index], Ys[index]

    Xs = []
    Ys = []
    for _ in range(numData) :
        x = int(10000 * random.random()) / 10000
        y = 0
        Xs.append(x)
        Ys.append(y)

    dataset = MyLittleDataset(Xs, Ys)
    curriculumBatcher = CurriculumBatchSampler(batch_size=bs, drop_last=True, num_data=numData)
    dataloader = DataLoader(dataset, batch_sampler=curriculumBatcher, num_workers=4)

    for epoch in range(3) :
        print("Epoch ", epoch)
        # print(dataloader.batch_sampler.samplingWeights)
        # print(dataloader.batch_sampler.indexOccurance)
        epochSum = 0
        for batchIndex in range(numData // bs) :
            for batch in dataloader :
                break

            preds = [float(val) for val in list(batch[0])]
            print("\r\tpreds ", batchIndex, "= ", preds, end="")
            dataloader.batch_sampler.updateSamplingWeights(preds)
            epochSum += sum(preds)
        print()
        print(epochSum)

        print(dataloader.batch_sampler.samplingWeights)
        print(dataloader.batch_sampler.indexOccurance)
