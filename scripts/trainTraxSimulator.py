


import os
import sys
# Adds higher directory to python modules path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
import matplotlib.pyplot as plt
import math
import copy

import optuna
import csv


from TrainerDL.Data import splitData
from TrainerDL.Saver import Saver
from TrainerDL.Network.MLP import MLP




def main() :

    study = optuna.create_study(direction='minimize')

    # Data
    xs, ys = loadData()

    training_dataset, validation_dataset, test_dataset, training_generator, validation_generator, test_generator = splitData(xs, ys, shuffle=False)

    study.optimize(trainWrapper(training_generator, validation_generator, test_generator), n_trials=3)


def trainWrapper(training_generator, validation_generator, test_generator) :
    """ Wrapper to provide the generators to the train function """

    print("training_generator.dataset.Xs : ", training_generator.dataset.Xs)
    def train(trial) :

        # Network
        model = MLP.sample(
            trial,
            n_inputs=len(training_generator.dataset.Xs[0]),
            n_outputs=len(training_generator.dataset.Ys[0])
        )
        print(model)
        print("model.parameters() : ", model.parameters())

        def MSE(prediction, target):
            loss = torch.mean((prediction-target)**2)
            return loss
        loss = trial.suggest_categorical("Loss", ["MSE"])#, "BCE"])
        if (loss == "MSE") : loss = MSE
        if (loss == "BCE") : loss = torch.nn.BCELoss()

        learningRate = trial.suggest_float("LearningRate", 1e-4, 1e-1, log=True)
        optimizer = trial.suggest_categorical("Optimizer", ["adam", "sgd", "adadelta"])
        if (optimizer == "adam"): optimizer = torch.optim.Adam(model.parameters(), lr=learningRate) #, amsgrad=True)
        if (optimizer == "sgd"): optimizer = torch.optim.SGD(model.parameters(), lr=learningRate, momentum=0.9)
        if (optimizer == "adadelta"): optimizer = torch.optim.Adadelta(model.parameters(), lr=learningRate)

        maxEpochs = trial.suggest_int("Epoch", 512, 512)
        batchSize = trial.suggest_int("BatchSize", 8, 512, log=True)

        saver = Saver(saveFolderPath="temp/")

        for i_epoch in range(maxEpochs):
            epochLoss = 0
            model.train()
            trainLoss = torch.tensor(0).type(torch.DoubleTensor).to(device) # Compute the training loss in the GPU. In CPU, is slows down training by 3x
            for i_batch, (xBatch, yBatch) in enumerate(training_generator):
                batchNumber = i_batch + 1
                    # Prepare data
                xBatch = xBatch.type(torch.FloatTensor).to(device)
                yBatch = yBatch.type(torch.FloatTensor).to(device)
                    # Optimize weigths
                optimizer.zero_grad()
                predictions = model(xBatch)
                print("predictions : ", predictions)
                print("yBatch : ", yBatch)
                batchLoss = loss(predictions, yBatch)
                print(f"Batch {batchNumber}\t\tbatchLoss={batchLoss}")
                batchLoss.backward()
                optimizer.step()
                trainLoss += batchLoss

            # Compute Train and Validation losses
            trainLoss = trainLoss.item() / batchNumber
            saver.saveEpoch(
                model,
                epochTrainingLoss=trainLoss,
                epochTrainingAccuracy=None,
                epochValidationLoss=None,
                epochValidationAccuracy=None,
                epochTestLoss=None,
                epochTestAccuracy=None,
                verbose=True
            )
        minTrainLoss = min(trainLosses)
        return minTrainLoss

    return train



def loadData() :
    print("load data")
    xs = []
    ys = []
    with open('data/data.csv', newline='') as csvfile:
        data = csv.reader(csvfile, delimiter=',')
        for i_row, row_t in enumerate(data):

            if (i_row == 0) :
                names = row_t
                continue

            row_t = [float(value) for value in row_t]
            row_t = normalizeRow(row_t, names)

            if (i_row > 1) :
                x = defineInput(row_tm1, row_t, names)
                xs.append(x)
                y = defineOutput(row_tm1, row_t, names)
                ys.append(y)

            row_tm1 = row_t

            if (i_row > 10) : break
    return xs, ys

def degToRad(angle_deg) :
    return angle_deg * math.pi / 180


def defineInput(row_tm1, row_t, names) :

    x = []

    # State
    x.append(row_tm1[names.index("arm_position")])
    x.append(row_tm1[names.index("arm_speed")])
    x.append(row_tm1[names.index("bucket_position")])
    x.append(row_tm1[names.index("bucket_speed")])
    x.append(row_tm1[names.index("trax_linear_speed")])
    x.append(row_tm1[names.index("trax_angular_speed")])
    roll_deg = row_tm1[names.index("roll_deg")]
    roll_rad = degToRad(roll_deg)
    x.append(math.cos(roll_rad))
    x.append(math.sin(roll_rad))
    pitch_deg = row_tm1[names.index("pitch_deg")]
    pitch_rad = degToRad(pitch_deg)
    x.append(math.cos(pitch_rad))
    x.append(math.sin(pitch_rad))
    yaw_deg = row_tm1[names.index("yaw_deg")]
    yaw_rad = degToRad(yaw_deg)
    x.append(math.cos(yaw_rad))
    x.append(math.sin(yaw_rad))

    # Action
    x.append(row_tm1[names.index("arm_command")])
    x.append(row_tm1[names.index("bucket_command")])
    x.append(row_tm1[names.index("trax_linear_command")])
    x.append(row_tm1[names.index("trax_angular_command")])

    # Delta t
    x.append(row_t[names.index("Time")] - row_tm1[names.index("Time")])

    return x

def defineOutput(row_tm1, row_t, names) :
    def delta(name) :
        return row_t[names.index(name)] - row_tm1[names.index(name)]
    def longitudeDegreesToMeter(deltaLongitude) :
        latitude_rad = degToRad(row_t[names.index("latitude")])
        return deltaLongitude * 111139 * math.cos(latitude_rad)
    def latitudeDegreesToMeter(deltaLatitude) :
        # Each latitude degree is 111km139m
        return deltaLatitude * 111139

    y = []
    y.append(delta("arm_position"))
    y.append(delta("arm_speed"))
    y.append(delta("bucket_position"))
    y.append(delta("bucket_speed"))
    y.append(delta("trax_linear_speed"))
    y.append(delta("trax_angular_speed"))

    y.append(delta("longitude"))
    y[-1] = longitudeDegreesToMeter(y[-1])
    y.append(delta("latitude"))
    y[-1] = latitudeDegreesToMeter(y[-1])
    y.append(delta("altitude"))

    y.append(delta("roll_deg"))
    y.append(delta("pitch_deg"))
    y.append(delta("yaw_deg"))

    return y


def normalizeRow(row_t, names) :
    row_t[names.index("arm_position")] /= 100
    row_t[names.index("arm_speed")] /= 30

    row_t[names.index("bucket_position")] /= 100
    row_t[names.index("bucket_speed")] /= 30

    row_t[names.index("trax_linear_speed")] /= 2
    row_t[names.index("trax_angular_speed")] /= 90

    row_t[names.index("arm_command")] /= 100
    row_t[names.index("bucket_command")] /= 100
    row_t[names.index("trax_linear_command")] /= 100
    row_t[names.index("trax_angular_command")] /= 100
    return row_t


if __name__ == "__main__" :
    main()
