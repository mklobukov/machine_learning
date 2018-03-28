#Mark Klobukov
#CS 383 HW3
#S-folds cross-validation
#2/7/2018

import numpy as np
import matplotlib.pyplot as plt
from CFLinReg import *
from util import *
import sys

dataFile = "x06Simple.csv"

def main():
    if len(sys.argv) < 2:
        print "Must provide an argument for S (how many folds)"
        return
    leave_one_out = False
    if sys.argv[1] == 'N':
        leave_one_out = True

    data = readFile2(dataFile)

    S = None
    if (not leave_one_out):
        S = float(sys.argv[1])
    else:
        S = data.shape[0]

    np.random.seed(0)
    shuffled = shuffleMatrix(data)

    #train model S times, compute RMSE each time,
    #and append to RMSE
    RMSE_list = list()

    sq_residuals_sum = 0
    #split matrix into n parts
    folds = np.asarray(np.array_split(shuffled, S))
    # print"shape of folds" , folds.shape
    #iterate S times. Designate a diff slice
    #as a test segment each time
    for i in range(0, int(S)):
    # for i in range(0, 1):
        #extract test data
        test = folds[i]

        #extract train data
        t1 = folds[0:i]
        t2 = folds[(i+1):int(S)]

        if leave_one_out:
            t = np.vstack((t1, t2))
        else:
            t = np.append(t1, t2)

        train = np.concatenate(t)
        means_train = getMeans(train[:,:-1])
        stds_train = getStds(train[:,:-1])
        #the test data must be standardized with the
        #mu and sigma from training data
        std_test = standardizeTestSF(test, stds_train, means_train)
        std_test = np.append(np.ones([test.shape[0], 1]), std_test, 1)

        std_train = standardizeDataExceptLast(train)
        std_train = np.append(np.ones([train.shape[0], 1]), std_train, 1)

        #num features
        d = std_train.shape[1] -1
        theta = findTheta(std_train[:,0:d], np.asmatrix(std_train[:,d:]))
        # print "This theta = ", theta

        #extract all observations from test set
        #but don't include target value (last col)
        t = std_test[:,:-1]

        # print "t: ", t
        #find predicted values
        result = np.dot(t, theta)

        # print "result: "

        sq_residuals_sum += np.sum(np.square(result - std_test[:,-1]))

        rmse = RMSE(result, std_test[:,-1])
        RMSE_list.append(rmse)
    # print "all rmses: ", RMSE_list
    rmse = pow(sq_residuals_sum / data.shape[0], 0.5)

    message = "\nRMSE with " + str(int(S)) + " folds: " + str(rmse) + "\n"
    print message









if __name__ == "__main__":
    main()
