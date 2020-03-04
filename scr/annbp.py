from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import TanhLayer, SigmoidLayer, MDLSTMLayer
from pybrain.tools.customxml import NetworkWriter, NetworkReader
import math as mt
import dataUtil as dtu

import numpy as np


class NeuralNetwork:

    def __init__(self, num, traindata, inputdata, hiddenNN, type='Tanh', maxepochs=2000, ifprint=False, toload = False,
                 lRate = 0.0001, moment = 0.005):

        self.dataSet = self.createDataSet(traindata[0], traindata[1]) #trainIn trainOut
        if (toload==True):
            self.net = NetworkReader.readFrom('nets/newnettemplate{}.xml'.format(num))
            print(self.net)
        else:
            self.net = self.createNet(inputdata.shape[1], hiddenNN, type, ifprint=True)
            self.trainer = self.trainTrainer(maxepochs, lRate, moment, ifprint)
            NetworkWriter.writeToFile(self.net, 'nets/newnettemplatetest{}.xml'.format(num))

        self.predictedVals = None
        self.predictedX = None
        self.resultTrainVals = None


    def createDataSet(self, trainInput, trainOut):
        ds = SupervisedDataSet(trainInput.shape[1], 1)
        # adhoc - no first input element
        # adding all train samples to dataset
        for x in range(len(trainInput)): #for x in range(len(trainInput)-1):
            ds.addSample(trainInput[x], trainOut[x])  # ds.addSample(trainInput[x + 1], trainOut[x])
        return ds


    def updateDataSet(self, numSamples, trainInput, trainOut):
        for x in range(numSamples):  # for x in range(90):
            self.dataSet.addSample(trainInput[x], trainOut[x])  # ds.addSample(trainNpInput[x],trainNpOut[x])
        return self.dataSet


    def createNet(self, inputDim, hiddenNN, type, ifprint):
        net = None
        if (type == 'Tanh'):
            net = buildNetwork(inputDim, hiddenNN, 1, hiddenclass=TanhLayer)
        if (type == 'Sig'):
            net = buildNetwork(inputDim, hiddenNN, 1, hiddenclass=SigmoidLayer)
        if (type =='MDLSTM'):
            net = buildNetwork(inputDim, hiddenNN, 1, hiddenclass=MDLSTMLayer)
        #net.sortModules()
        # hiddenclass =  SigmoidLayer TanhLayer MDLSTMLayer
        if (ifprint == True):
            print(net['in'])
            print(net['hidden0'])
            print(net['out'])
            print(net)
        return net


    def trainTrainer(self, epochs, lRate, moment, ifprint):  # momentum(gradient of the last timestep)
        # net.reset()
        plotRMSE = []
        newtrainer = BackpropTrainer(self.net, learningrate=lRate, momentum=moment, verbose=ifprint) #weightdecay=0.01
        newtrainer.trainUntilConvergence(self.dataSet, maxEpochs=epochs, continueEpochs=100)  # validationProportion=0.1
        #newtrainer.trainOnDataset(self.dataSet, epochs)
        #newtrainer.testOnData(self.dataSet, verbose=False)

        #TRAINING ERROR---------------------------------------
        errorEp = newtrainer.trainingErrors
        for num,x in enumerate(errorEp):
            n = len(errorEp[:num+1]) #(num+1) due to /0
            sumsq = 0.0

            for num2, x2 in enumerate(errorEp[:num+1]):
                sumsq += mt.pow(errorEp[num2], 2)

            plotRMSE.append(mt.sqrt(sumsq / n))

        #print (plotRMSE)
        dtu.buildGraph("Training RMSE errors", True,
                   ['Epochs', 'Training error (RMSE) distribution'], [[plotRMSE, 'r--', 1.0, 2.0]],
                   ['Error (RMSE) distribution'])
        # TRAINING ERROR END---------------------------------------

        return newtrainer

    def activateNet(self, npInput):
        resultData = []
        for num, input in enumerate(npInput):
            result = self.net.activate(input)
            resultData.append(result)
        return dtu.toListFromLArr(resultData)
        #return resultData

    def testNetOnData(self, testInData): #train or test data
        resultDataArr = self.activateNet(testInData)
        result = dtu.toListFromLArr(resultDataArr)
        return result