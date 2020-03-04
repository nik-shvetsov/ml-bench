from sklearn import tree
from sklearn import metrics
from sklearn import ensemble
import dataUtil as dtu

class TreeRegression():
    def __init__(self, maxdepth): #treelist = None

        self.treeReg = tree.DecisionTreeRegressor(max_depth=maxdepth)
        self.maxDepth = maxdepth
        self.futureDays = None

        #self.treeReg.fit(traindata[0], traindata[1])

    def scoreTree(self, trainin, trainout):
        print(self.treeReg)
        print("Model score for DTReg (0.0 - 1.0): {}".format(self.treeReg.score(trainin, trainout)))

    def predictTree(self, input):
        predictedData = self.treeReg.predict(input)
        return predictedData