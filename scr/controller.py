import annbp as nn
import dataUtil as dtu
import math as mt
import numpy as np

class Controller():

    def __init__(self, window, testdays, data, nnlist = None, treelist = None, rftreelist = None, adalist = None ):
        self.window = window # usually 8 7 or 3
        self.testDays = testdays #splitting index for testing
        self.allData = data
        self.netList = nnlist #netlist
        self.treeList = treelist
        self.rfTreeList = rftreelist
        self.adaList = adalist

        self.futureDays = None

    def sendNotification(self):
        return True

    def netPerformanceEval(self, trainin, testin): # trainout testout
        for numnet, exNet in enumerate(self.netList):
            print("-----------------------------------------------")
            print ("Evaluating of net #{}:".format(numnet+1))
            # test on train data
            resultTrainData = exNet.activateNet(trainin)
            # test on test data
            resultTestData = exNet.activateNet(testin)

            # test day and week ahead
            unknownData = []  # futurelist = []
            restlist = testin[-self.window:] #predictionsize depends on window size

            for num, x in enumerate(restlist):
                unknownData.append(exNet.net.activate(x))

            unknownData = dtu.toListFromLArr(unknownData)
            # minmax16.transform()
            # -----------------------------------------------
            # network = network.reset()

            dataOutput = self.allData[2]
            trainOut = self.allData[4]

            #print(dataOutput[-self.testDays:])
            #print(resultTestData)
            #print("Length")
            #print(len((dataOutput[-self.testDays:])))
            #print(len(resultTestData))
            #print (resultTestData)

            resultTrainData = exNet.activateNet(trainin)
            resultTestData = exNet.activateNet(testin)
            dataOutput = self.allData[2]
            #trainOut = self.allData[4]

            #length = len(datalist1)
            # print ("Length of evaluated error list is {}". format(length))
            error = dtu.computeSimpleAvgErr(resultTestData,dataOutput[-self.testDays:])
            print("Error of actual and test data: {}".format(error))

            unknownX = np.arange(len(dataOutput)-1,  len(dataOutput) + self.window) # -1 for connection
            resultTrainTest = np.concatenate((resultTrainData, resultTestData), axis=0)
            unknownData.insert(0, resultTrainTest[-1]) #insert last element from resultTrainTest to unknown first pos
            # print (len(unknownData), len(unknownX))

            plotlistNN = [[dataOutput, 'go-', 1.0, 1.0], [resultTrainTest, 'bo-', 1.0, 1.0],
                          [unknownX, unknownData, 'ro-', 1.0, 1.0]]
            legendlistNN = ['All output data', 'Train and test data', 'Predicted data']
            dtu.buildGraph("Results of BP ANN (DFF)#{}".format(numnet+1), True, ['Days', 'Price, NOK'], plotlistNN, legendlistNN)

            dtu.analyzeResult(resultTrainTest, dataOutput, "All data")

            #for ensembling
            exNet.predictedVals = unknownData
            exNet.predictedX = unknownX
            exNet.resultTrainVals = resultTrainTest

            print("Ended evaluation of net.")
            print("-----------------------------------------------")

    def treePerformanceEval(self, datain, dataout): #testin testout

        for numtree, extree in enumerate(self.treeList):
            #extree.scoreTree(datain, dataout)
            #print("-----------------------------------------------")
            print ("Evaluating of tree #{}:".format(numtree+1))

            # test on train data
            resultData = extree.predictTree(datain)

            #for predict
            unknownData = []  # futurelist = []
            restlist = datain[-self.window:]  # predictionsize depends on window size
            for num, x in enumerate(restlist):
                unknownData.append(extree.predictTree(x.reshape(1, -1)))
            unknownData = dtu.toListFromLArr(unknownData)

            dataOutput = self.allData[2]
            unknownX = np.arange(len(dataOutput) - 1, len(dataOutput) + self.window)  # -1 for connection
            unknownData.insert(0, resultData[-1])  # insert last element from resultTrainTest to unknown first pos
            resultDataX = np.arange(len(dataOutput) - len(resultData), len(dataOutput))

            plotlistNN = [[resultDataX, resultData, 'go-', 1.0, 1.0], #[resultDataX, dataout, 'co-', 1.0, 1.0]
                          [unknownX, unknownData, 'ro-', 1.0, 1.0], [dataOutput, 'bo-', 0.5, 1.0]]
            legendlistNN = ['Result Test data', 'Predicted data', "All data"] #'Actual test data'
            dtu.buildGraph("Results of regression tree #{}".format(numtree+1), True, ['Days', 'Price, NOK'], plotlistNN, legendlistNN)

            dtu.analyzeResult(resultData, dataout, "test dataset")
            print("Ended evaluation of tree.")

    def rftreePerformanceEval(self, datain, dataout):
        for numrf, exrf in enumerate(self.rfTreeList):
            #exrf.scoreRFTree(datain, dataout)
            #print("-----------------------------------------------")
            print ("Evaluating of random forest predictor #{}:".format(numrf+1))

            # test on train data
            resultData = exrf.predictRFTree(datain)

            #for predict
            unknownData = []  # futurelist = []
            restlist = datain[-self.window:]  # predictionsize depends on window size
            for num, x in enumerate(restlist):
                unknownData.append(exrf.predictRFTree(x.reshape(1, -1)))
            unknownData = dtu.toListFromLArr(unknownData)

            dataOutput = self.allData[2]
            unknownX = np.arange(len(dataOutput) - 1, len(dataOutput) + self.window)  # -1 for connection
            unknownData.insert(0, resultData[-1])  # insert last element from resultTrainTest to unknown first pos
            resultDataX = np.arange(len(dataOutput) - len(resultData), len(dataOutput))

            plotlistNN = [[resultDataX, resultData, 'go-', 1.0, 1.0], [resultDataX, dataout, 'bo-', 0.5, 1.0],
                          [unknownX, unknownData, 'ro-', 1.0, 1.0]] #[dataOutput, 'bo-', 0.5, 1.0]
            legendlistNN = ['Result Test data', 'Actual test data', 'Predicted data'] #"All data"
            dtu.buildGraph("Results of random forest #{}".format(numrf+1), True, ['Days', 'Price, NOK'], plotlistNN, legendlistNN)

            dtu.analyzeResult(resultData, dataout, "test dataset")
            print("Ended evaluation of random forest predictor.")

    def adaPerformanceEval(self, datain, dataout):
        for numada, exada in enumerate(self.adaList):
            #exrf.scoreAda(datain, dataout)
            #print("-----------------------------------------------")
            print ("Evaluating of AdaBoost predictor #{}:".format(numada+1))

            # test on train data
            resultData = exada.predictAda(datain)

            #for predict
            unknownData = []  # futurelist = []
            restlist = datain[-self.window:]  # predictionsize depends on window size
            for num, x in enumerate(restlist):
                unknownData.append(exada.predictAda(x.reshape(1, -1)))
            unknownData = dtu.toListFromLArr(unknownData)

            dataOutput = self.allData[2]
            unknownX = np.arange(len(dataOutput) - 1, len(dataOutput) + self.window)  # -1 for connection
            unknownData.insert(0, resultData[-1])  # insert last element from resultTrainTest to unknown first pos
            resultDataX = np.arange(len(dataOutput) - len(resultData), len(dataOutput))

            plotlistNN = [[resultDataX, resultData, 'go-', 1.0, 1.0], [resultDataX, dataout, 'bo-', 0.5, 1.0],
                          [unknownX, unknownData, 'ro-', 1.0, 1.0]] #[dataOutput, 'bo-', 0.5, 1.0]
            legendlistNN = ['Result Test data', 'Actual test data', 'Predicted data'] #"All data"
            dtu.buildGraph("Results of AdaBoost #{}".format(numada+1), True, ['Days', 'Price, NOK'], plotlistNN, legendlistNN)

            dtu.analyzeResult(resultData, dataout, "test dataset")
            print("Ended evaluation of AdaBoost predictor.")

    def ensembleAvgNets(self, netlist):
        ensembleResPredAvg = []
        ensembleResTrainAvg = []

        allpredictedVals = []
        alltrainedVals = []

        #forming all values in one list
        for num,x in enumerate(netlist): #for each net in list
            allpredictedVals.append(x.predictedVals)
            alltrainedVals.append(x.resultTrainVals)

        #forming train data
        #print(alltrainedVals[0])
        #ensembling trained data
        for numinlistTr, xinlistTr in enumerate(alltrainedVals[0]):
            #print("Step:", numinlistTr+1)
            sumTr = 0
            for numvalTr, xvalTr in enumerate(alltrainedVals):  # for each value in predicted values
                sumTr += xvalTr[numinlistTr]
            ensembleResTrainAvg.append(sumTr / len(netlist))
            #print(sum1/len(netlist))
            #print(ensembleResTrainAvg)
        #print(ensembleResTrainAvg)

        #forming future data
        #ensembling process for future
        for numinlist, xinlist in enumerate(allpredictedVals[0]):
            #print("Step:", numinlist+1)
            sum = 0
            for numval, xval in enumerate(allpredictedVals): #for each value in predicted values
                sum += xval[numinlist]
            ensembleResPredAvg.append(sum/len(netlist))
            #print(sum/len(netlist))
            #print(ensembleResPredAvg)
        #print(ensembleResPredAvg)

        #plotting ensemple predictions

        dataOutput = self.allData[2]
        xEnsemble = np.arange(len(dataOutput) - 1, len(dataOutput) + self.window)
        # connect line for presentation purposes
        connectx = [len(dataOutput) - 1, len(dataOutput) - 1]
        connecty = [dataOutput[-1], ensembleResPredAvg[0]]

        dtu.analyzeResult(ensembleResTrainAvg[-self.testDays:], dataOutput[-self.testDays:],
                          "ensemble prediction on test data")

        # plot graph
        plotlistEns = [[xEnsemble, ensembleResPredAvg, 'ro-', 1.0, 1.0], [dataOutput, 'bo-', 1.0, 1.0],
                       [connectx, connecty, 'yo-', 0.3, 1.0], [ensembleResTrainAvg, 'co-', 1.0, 1.0]]
        legendlistEns = ['Ensemble Result Prediction', 'All data output', 'Error connection', 'Ensemble result Train data']
        dtu.buildGraph("Results of Ensemble nets for prediction", True, ['Days', 'Price, NOK'], plotlistEns,
                       legendlistEns)

        return ensembleResPredAvg

def main():
    print ("Initializing controller.")

#if __name__ == '__main__':
    #main()