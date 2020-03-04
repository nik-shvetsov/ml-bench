from sklearn import metrics
from sklearn import preprocessing
import numpy as np
import pandas as pd # used for importing from excel
from pylab import * #rand, plot, show, norm, axis, hist, xlabel, ylabel, title
import math as mt

#from openpyxl import load_workbook


def toListFromLArr(listarr):
    resultData = []
    for num, x in enumerate(listarr):
        resultData.append(x[0])
    return resultData


def importWeatherData(weatherListParams):
    rawWeatherEx = pd.read_excel(weatherListParams[1])
    for x in range(len(weatherListParams[2])):
        rawWeatherEx = rawWeatherEx.drop(weatherListParams[2][x], 1)
    exWeatherdata = rawWeatherEx.as_matrix()
    return exWeatherdata


def importDataSet(path, window, normalize, weatherListParams):
    # print ("Original data:")
    dropCols = ['Date']
    rawexcel = pd.read_excel(path)
    for x in range(len(dropCols)):
        rawexcel = rawexcel.drop(dropCols[x], 1)  # delete column unnesessary columns
    #print (rawexcel)
    #print (len(rawexcel))
    exdata = rawexcel.as_matrix()

    #print ("Weather data test:")
    if (weatherListParams[0] == True):
        exWeatherdata = weatherListParams[3]

    #cycle
    i = 0
    inputDataArr = []
    outputDataArr = []

    while (i < len(rawexcel)-window): #totalDataRows - window = num of data sets

        inputList = exdata[i:i + window]  # last element does not count (not included) 0-2
        templist = []
        for elem in inputList:
            templist.append(elem[0])
        if (weatherListParams[0] == True):
            templist.append(weatherListParams[3][i + window])  # append temperature data

        outputData = exdata[i + window][0]
        inputDataArr.append(templist)
        outputDataArr.append(outputData)
        i += 1

    if (normalize==True):
        # normalizing input data from 0-1-----
        # npInput = np.array(inputDataArr)
        npInputTemp = np.array(inputDataArr)
        min_max_scaler = preprocessing.MinMaxScaler()  # feature_range=[-1,1]
        npInput = min_max_scaler.fit_transform(npInputTemp)

        # no need to normilize output
        # npOutTemp = np.array(outputDataArr)
        # npOut = min_max_scaler.fit_transform(npOutTemp)
        npOut = np.array(outputDataArr)

    else:
        npInput = np.array(inputDataArr)
        npOut = np.array(outputDataArr)

    return npInput, npOut, min_max_scaler


def splitData(splitIndex, inData, outData, ifprint):
    #train data
    trainInput = inData[:-splitIndex]
    trainOut = outData[:-splitIndex]
    # test data
    testInput = inData[-splitIndex:]
    testOut = outData[-splitIndex:]

    if (ifprint == True):
        print("Train Input data:")
        print(trainInput)
        print("Train Output data:")
        print(trainOut)

        print("Test Input data:")
        print(testInput)
        print("Test Output data:")
        print(testOut)

    return trainInput, trainOut, testInput, testOut

def createSplitDataset(path, window, norm, weatherlist, testdays, ifprint):
    npinput, npout, minmax = importDataSet(path, window, norm, weatherlist)
    trainnpinput, trainnpout, testnpinput, testnpout = splitData(testdays, npinput, npout, ifprint)
    return [minmax, npinput, npout, trainnpinput, trainnpout, testnpinput, testnpout]


def buildGraph(titlename, ifgrid, labelsxy, plotlist, legendlist):
    figure()
    title(titlename)
    grid(ifgrid)
    xlabel(labelsxy[0])
    ylabel(labelsxy[1])
    for num, x in enumerate(plotlist):
        if (len(plotlist[num]) == 4):
            plot(x[0], x[1], alpha = x[2], label=legendlist[num], linewidth=x[3]) #plotlist,colorlist,alphalist,widthlist
        else:
            plot(x[0], x[1], x[2], alpha = x[3], label=legendlist[num], linewidth=x[4])
    legend()
    show()


def computeMAE(predicted, actual): #Mean absolute error
    n = len(predicted)
    sumfabs = 0.0

    for num,x in enumerate(predicted):
       sumfabs += mt.fabs(predicted[num] - actual[num])

    errValue = (1.0/n)*sumfabs
    return round(errValue,4)


def computeMAPE(predicted, actual): #Mean absolute percent error
    n = len(predicted)
    sumfabs = 0.0

    for num,x in enumerate(predicted):
       sumfabs += mt.fabs(actual[num] - predicted[num]) / mt.fabs(actual[num])

    perErrorValue = ((1.0 / n) * sumfabs) * 100.0
    return round(perErrorValue,4)


def computeRMSE(predicted, actual): #Root-mean-square deviation/error
    n = len(predicted)
    sumsq = 0.0

    for num,x in enumerate(predicted):
       sumsq += mt.pow(predicted[num] - actual[num], 2)

    devValue = mt.sqrt(sumsq/n)
    return round(devValue,4)


def computeSimpleAvgErr(datalist1, datalist2):
    error = 0
    length = len(datalist1)
    # print ("Length of evaluated error list is {}". format(length))
    for num, x in enumerate(datalist1):
        error += mt.fabs(datalist1[num] - (datalist2[num]))
    error /= length
    return round(error,4)


def analyzeResult(predicted, actual, method):
    print ("------------------------------------")
    print ("Accuracy mesuring for {}: ".format(method))
    print ("MAE (Mean absolute error): {}".format(computeMAE(predicted, actual)))
    print ("MAPE (Mean absolute percent error): {} %".format(computeMAPE(predicted, actual)))
    print ("RMSE (Root-mean-square deviation): {}".format(computeRMSE(predicted, actual)))
    print ("------------------------------------")


def infoGainTest(oldData, newData, actualData): #resultDataOld(without temp), resultDataNew(with temp), npOut
    listIG = []
    for num,x in actualData:
        oldvsactual = mt.fabs(oldData[num] - actualData[num])
        newvsactual = mt.fabs(newData[num] - actualData[num])
        if (oldvsactual > newvsactual):
            listIG.append(1) # if newdata is closer to actual then we gain info
        if (oldvsactual < newvsactual):
            listIG.append(-1) # if olddata is closer to actual then we gain neg info
        if (oldvsactual == newvsactual):
            listIG.append(0)

    print(listIG)
    buildGraph("Information gain graph (simple) for adding weather data", True,
               ['Data', 'Gain'], [[listIG, 'r-', 1.0, 2.0]],
               ['Information gain (simple)'])