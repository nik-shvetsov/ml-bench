import controller as contrl
import annbp as nn
import treeR as trg
import rftree as rft
import adaB as adab
import dataUtil as dtu
import math as mt

#need to include in PYTHONPATH scr folder
#modify excel files for needed month-dates

#preparations #change this lines for paramteres
pathToFile16 = "datafiles/data2016.xlsx" #num of lines
pathToFile15 = "datafiles/data2015.xlsx" #365-normal
pathToFile14 = "datafiles/data2014.xlsx"
pathToFile13 = "datafiles/data2013.xlsx"
pathToWeather = "datafiles/weatherdata.xlsx"
weatherListParams = [True, pathToWeather, ['Date','Max','Min','Normal']] #using Average temperature

#create nparrays for all year data

weatherData = dtu.importWeatherData(weatherListParams)
weatherListParams.append(weatherData)

window = 8 # 3, 7, 8
testDays = 16 # splitIndex
year13data = dtu.createSplitDataset(pathToFile13, window, True, weatherListParams, testDays, False)
year14data = dtu.createSplitDataset(pathToFile14, window, True, weatherListParams, testDays, False)
year15data = dtu.createSplitDataset(pathToFile15, window, True, weatherListParams, testDays, False)
year16data  = dtu.createSplitDataset(pathToFile16, window, True, weatherListParams, testDays, False)

controller = contrl.Controller(window, testDays, year16data)

#[minmax, npinput, npout, trainnpinput, trainnpout, testnpinput, testnpout]
minmaxscaler = year16data[0]
dataInput = year16data[1]
dataOutput = year16data[2]
trainIn = year16data[3]
trainOut = year16data[4]
testIn = year16data[5]
testOut = year16data[6]

#long term multiyear data
'''
dataInput = np.concatenate((npInput15, npInput16), axis = 0)
dataOutput = np.concatenate((npOut15, npOut16), axis = 0)
trainIn, trainOut, testIn, testOut = splitData(testDays, dataInput, dataOutput, False)
'''

#nets for ensemble
nNet1 = nn.NeuralNetwork(1,[trainIn,trainOut],dataInput, 51, type='Tanh', ifprint=True, toload=True)
nNet2 = nn.NeuralNetwork(2,[trainIn,trainOut],dataInput, 51, type='Sig', ifprint=True, toload=True)
nNet3 = nn.NeuralNetwork(3,[trainIn,trainOut],dataInput, 51, type='MDLSTM', ifprint=True, toload=True)

treeReg1 = trg.TreeRegression(5)
treeReg1.treeReg.fit(trainIn, trainOut)

rfTree1 = rft.RFTree(300)
rfTree1.rftreeReg.fit(trainIn, trainOut)

ada1 = adab.AdaBoostCl(8,300)
ada1.ada.fit(trainIn, trainOut)

controller.netList = [nNet1, nNet2, nNet3]
controller.treeList = [treeReg1]
controller.rfTreeList = [rfTree1]
controller.adaList = [ada1]

print("Net evaluation:")
controller.netPerformanceEval(trainIn, testIn)
print("Regression tree evaluation:")
controller.treePerformanceEval(testIn, testOut)
print("Random forest evaluation:")
controller.rftreePerformanceEval(dataInput, dataOutput)
print("AdaBoost evaluation:")
controller.adaPerformanceEval(dataInput, dataOutput)

controller.ensembleAvgNets(controller.netList)
print("End evaluation.----------------------------")
input()