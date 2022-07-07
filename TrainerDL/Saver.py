
import datetime
from pathlib import Path
import json
import pickle
import random
import torch
import math

import matplotlib.pyplot as plt

from Network.MLP import MLP

class Saver :

    """ Records and saves training logs and results """

    def __init__(self, saveFolderPath="data", deepSaveFactor=2) :

        self.nextDeepSaveEpoch = 1 # When to save all graphs and network, additionnaly to last and best
        self.deepSaveFactor = deepSaveFactor

        now = datetime.datetime.now()
        date = f"{now.year}y{str(now.month).zfill(2)}m{str(now.day).zfill(2)}d_{str(now.hour).zfill(2)}h{str(now.minute).zfill(2)}m{str(now.second).zfill(2)}s{str(int(now.microsecond/1e3)).zfill(3)}ms{str(int(now.microsecond%1e3)).zfill(3)}ys"
        self.trainingFolderPath = f"{saveFolderPath}/{date}"
        Path(self.trainingFolderPath).mkdir(parents=True, exist_ok=True)

        self.trainLosses = []
        self.trainAccuracies = []
        self.validationLosses = []
        self.validationAccuracies = []
        self.testLosses = []
        self.testAccuracies = []


    def saveDict(self, dictToSave, pathInTrainingFolder="parameters.json") :
        # Save json in training folder
        jsonPath = f"{self.trainingFolderPath}/{pathInTrainingFolder}"
        jsonFolder = '/'.join(jsonPath.split('/')[:-1])
        Path(jsonFolder).mkdir(parents=True, exist_ok=True)
        with open(f"{jsonPath}", "w") as jsonFile :
            json.dump(dictToSave, jsonFile, indent=4)


    def saveEpoch(self,
        model,
        epochTrainingLoss=None,
        epochTrainingAccuracy=None,
        epochValidationLoss=None,
        epochValidationAccuracy=None,
        epochTestLoss=None,
        epochTestAccuracy=None,
        verbose=True
    ) :
        """ Records what is provided and plots it """
        self.trainLosses.append(epochTrainingLoss)
        self.trainAccuracies.append(epochTrainingAccuracy)
        self.validationLosses.append(epochValidationLoss)
        self.validationAccuracies.append(epochValidationAccuracy)
        self.testLosses.append(epochTestLoss)
        self.testAccuracies.append(epochTestAccuracy)

        # Training message
        if (verbose) :
            n_epoch = len(self.trainLosses)
            toPrint = f"[Saver] :: Epoch : {n_epoch}"
            if (epochTrainingLoss is not None) : toPrint += f" ; Train Loss = {round(epochTrainingLoss, 5)}"
            if (epochTrainingAccuracy is not None) : toPrint += f" ; Train Acc = {round(epochTrainingAccuracy, 3)}"
            if (epochValidationLoss is not None) : toPrint += f" ; Valid Loss = {round(epochValidationLoss, 5)}"
            if (epochValidationAccuracy is not None) : toPrint += f" ; Valid Acc = {round(epochValidationAccuracy, 3)}"
            if (epochTestLoss is not None) : toPrint += f" ; Test Loss = {round(epochTestLoss, 5)}"
            if (epochTestAccuracy is not None) : toPrint += f" ; Test Acc = {round(epochTestAccuracy, 3)}"
            print(f"{toPrint}.")

        # Always save the last results
        self.saveResultsToFolder(model, saveFolderPath=f"{self.trainingFolderPath}/last")

        n_epoch = len(self.trainLosses)
        if (n_epoch >= self.nextDeepSaveEpoch) :
            # Every once in a logarithmic while, permanently save intermediary results
            self.saveResultsToFolder(model, saveFolderPath=f"{self.trainingFolderPath}/epoch_{str(n_epoch).zfill(5)}")
            self.nextDeepSaveEpoch = n_epoch * self.deepSaveFactor

        if (epochValidationLoss is not None) and self._listIsValid(self.validationLosses) :
            minValidationLoss = min(self.validationLosses)
            if (epochValidationLoss <= minValidationLoss) :
                self.saveResultsToFolder(model, saveFolderPath=f"{self.trainingFolderPath}/best")


    def saveResultsToFolder(self, model, saveFolderPath=f"temp") :
        Path(saveFolderPath).mkdir(parents=True, exist_ok=True)
        n_epoch = len(self.trainLosses)
        torch.save(model.state_dict(), f"{saveFolderPath}/model.pt")
        self.saveGraphs(saveFolderPath=saveFolderPath)

    def saveGraphs(self, saveFolderPath="temp") :

        self.saveGraph(
            toPlot={
                "Train Loss" : self.trainLosses,
                "Train Accuracy" : self.trainAccuracies,
                "Validation Loss" : self.validationLosses,
                "Validation Accuracy" : self.validationAccuracies,
                "Test Loss" : self.testLosses,
                "Test Accuracy" : self.testAccuracies,
            },
            graphPath=f"{saveFolderPath}/All.png"
        )

        self.saveGraph(
            toPlot={
                "Train Loss" : self.trainLosses,
                "Validation Loss" : self.validationLosses,
                "Test Loss" : self.testLosses,
            },
            graphPath=f"{saveFolderPath}/Losses.png"
        )

        self.saveGraph(
            toPlot={
                "Train Accuracy" : self.trainAccuracies,
                "Validation Accuracy" : self.validationAccuracies,
                "Test Accuracy" : self.testAccuracies,
            },
            graphPath=f"{saveFolderPath}/Accuracies.png"
        )

        self.saveGraph(
            toPlot={
                "Train Loss" : self.trainLosses,
                "Train Accuracy" : self.trainAccuracies,
            },
            graphPath=f"{saveFolderPath}/Train.png"
        )

        self.saveGraph(
            toPlot={
                "Validation Loss" : self.validationLosses,
                "Validation Accuracy" : self.validationAccuracies,
            },
            graphPath=f"{saveFolderPath}/Validation.png"
        )

        self.saveGraph(
            toPlot={
                "Test Loss" : self.testLosses,
                "Test Accuracy" : self.testAccuracies,
            },
            graphPath=f"{saveFolderPath}/Test.png"
        )


    def _listIsValid(self, l) :
        for element in l :
            if element is None :
                return False
        return True

    def saveGraph(self, toPlot={}, graphPath="temp/graph.png", log=True) :
        keys = list(toPlot.keys())


        if (len(keys) == 0) : return
        if (log) :
            self.saveGraph(toPlot=toPlot, graphPath=graphPath, log=False)
            graphPath = f"{graphPath.split('.')[0]}_log.png" # Add '_log' to the name

        plot = plt.figure()

        savePlot = False # Stay False if no valid list was plotted
        xAxis = list(range(len(self.trainLosses)))
        for label, values in toPlot.items() :
            if (self._listIsValid(values)) :
                savePlot = True
                if (log) :
                    plt.plot(xAxis, [math.log(0.0001 if (value <= 0) else value) for value in values], "-o", label=label)
                else :
                    plt.plot(xAxis, values, "-o", label=label)
            # plt.plot(xAxis, validationLosses, "b--o", label="ValidationLoss")
        if (savePlot) :
            plt.grid(linestyle='--', linewidth=0.15)
            plt.legend()
            plt.savefig(f"{graphPath}")
        plt.close()



if __name__ == "__main__" :
    saver = Saver(saveFolderPath="temp/", deepSaveFactor=2)

    model = mlp = MLP(n_inputs=4, n_outputs=2, n_hidden=3, shape=1)

    trainLoss = 1
    trainAcc = 0
    validLoss = 1
    validAcc = 0
    testLoss = 1
    testAcc = 0

    for epoch in range(150) :
        trainLoss *= random.uniform(0.90, 0.99)
        trainAcc = max(0.001, 1 - trainLoss * random.uniform(0.8, 1))
        validLoss *= random.uniform(0.90, 0.99)
        validAcc = max(0.001, 0.9 - validLoss * random.uniform(0.8, 1))
        testLoss *= random.uniform(0.90, 0.99)
        testAcc = max(0.001, 0.8 - testLoss * random.uniform(0.8, 1))

        saver.saveEpoch(
            model,
            epochTrainingLoss=trainLoss,
            epochTrainingAccuracy=trainAcc,
            epochValidationLoss=validLoss,
            epochValidationAccuracy=0,
            epochTestLoss=None,
            epochTestAccuracy=None,
            verbose=True
        )
