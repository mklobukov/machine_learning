#Mark Klobukov
#CS383 HW3
#Closed form lin reg
#2/7/2018

import numpy as np
import matplotlib.pyplot as plt
from util import *
from CFLinReg import *

dataFile = "x06Simple.csv"

def main():
    data = readFile2(dataFile)
    np.random.seed(0)
    shuffled = shuffleMatrix(data)
    train, test = splitData(shuffled, 0.66)
    #standardize data -- training
    std = standardizeDataExceptLast(train)
    std = np.append(np.ones([train.shape[0], 1]), std, 1)

    #standardize data -- testing
    std_test = standardizeDataExceptLast(test)
    std_test = np.append(np.ones([test.shape[0], 1]), std_test, 1)

    #num features
    d = std.shape[1]-1
    theta = findTheta(std[:,0:d], np.asmatrix(std[:,d:]))
    print "Theta = ", theta

    #extract all observations from test set
    #but don't include target value (last col)
    t = std_test[:, :-1]

    #what's the prediction for test data samples?
    result = np.dot(t, theta)
    rmse = RMSE(result, std_test[:,-1])
    for i in range(len(result)):
        print "Predicted = ", result[i], " Actual = ", std_test[i, std_test.shape[1]-1]
    print "RMSE = ", rmse



if __name__ == "__main__":
    main()
