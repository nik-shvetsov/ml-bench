from sklearn import ensemble

class RFTree():
    def __init__(self, estimators): #treelist = None

        self.rftreeReg = ensemble.RandomForestRegressor(n_estimators=estimators)
        self.estimators = estimators
        self.futureDays = None

        #self.rftreeReg.fit(traindata[0], traindata[1])

    def scoreRFTree(self, trainin, trainout):
        print(self.rftreeReg)
        #print("Model score for (0.0 - 1.0): {}".format(self..score(trainin, trainout)))

    def predictRFTree(self, input):
        predictedData = self.rftreeReg.predict(input)
        return predictedData