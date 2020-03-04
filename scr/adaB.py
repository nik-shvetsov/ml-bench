from sklearn import ensemble
from sklearn import tree
import numpy as np

class AdaBoostCl():
    def __init__(self, depthnum, estimators): #treelist = None
        self.rand = np.random.RandomState(1)  # Mersenne Twister pseudo-random number generator
        self.ada = ensemble.AdaBoostRegressor(tree.DecisionTreeRegressor(max_depth=depthnum),
                                 n_estimators=estimators, random_state=self.rand)
        self.depth = depthnum
        self.estimators = estimators
        self.futureDays = None

        #self.ada.fit(traindata[0], traindata[1])

    def scoreAda(self, trainin, trainout):
        print(self.ada)
        #print("Model score for (0.0 - 1.0): {}".format(self..score(trainin, trainout)))

    def predictAda(self, input):
        predictedData = self.ada.predict(input)
        return predictedData