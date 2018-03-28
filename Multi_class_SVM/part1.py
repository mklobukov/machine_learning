#Mark Klobukov
#CS383 hw 5 SVMs
#2/27/2018
import numpy as np
import matplotlib.pyplot as plt
from util import *
import sys
import random
import svm

# dataFile = "spambase.data"

def main():
    if len(sys.argv) < 2:
        print "Must provide data file name. Exiting\n"
        return
    dataFile = sys.argv[1]

    #read file, shuffle, split into 2/3 train, 1/3 test
    data = readFile2(dataFile)
    np.random.seed(0)
    shuffled = shuffleMatrix(data)
    train, test = splitData(shuffled, 0.66)

    #standardize data -- training
    std, mus, sigmas = standardizeDataExceptLast(train)

    #standardize data -- testing. Standardize
    #with the values obtained in training set
    std_test = standardizeTestSF(test, mus, sigmas)

    tp=tn=fp=fn = 0.
    print "Total test rows: ", len(test)
    #pass features and class label separately to the
    #classifier training function
    SVM_classifier = svm.trainClassifier(train[:, :-1], train[:, -1])

    for idx, t in enumerate(test):
        c = svm.classify(SVM_classifier, [ t[:-1] ])
        if idx % 100 == 0:
            print "# test rows classified:", idx, "-- " + str(float(idx)/float(len(test)) * 100) + "% done."
        #count stats. Compare last entry in this row to KNN classification
        if t[-1] == 1:
            if c == 1:
                tp +=1
            elif c == 0:
                fn +=1
        elif t[-1] == 0:
            if c == 1:
                fp +=1
            elif c == 0:
                tn += 1

    precision = div(tp, tp+fp)
    recall = div(tp, tp+fn)
    f_measure = fmeasure(precision, recall)
    accuracy = div(tp+tn, tp+tn+fp+fn)

    print "TP: ", tp, "TN: ", tn
    print "FP: ",  fp, "FN: ", fn
    print "Precision =", precision
    print "Recall =", recall
    print "f-measure =", f_measure
    print "Accuracy =", accuracy

def fmeasure(precision, recall):
    if precision == None or recall == None:
        return None
    return div(2*precision*recall, precision+recall)

def div(num, denom):
    if denom == 0 or num == None or denom == None:
        return None
    return num/denom


if __name__ == "__main__":
    main()
