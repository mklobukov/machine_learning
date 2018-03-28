#Mark Klobukov
#CS383 hw 5 SVMs -- multi class with confusion matrix
#2/27/2018
import numpy as np
import matplotlib.pyplot as plt
from util import *
import sys
import random
import svm
from itertools import combinations
from collections import Counter

# dataFile = "spambase.data"

def main():
    if len(sys.argv) < 2:
        print "Must provide data file name. Exiting\n"
        return
    dataFile = sys.argv[1]

    #read file, shuffle, split into 2/3 train, 1/3 test
    data = readFile3(dataFile)

    #get a list of all classes from dataset for 1vs1 classification
    classes = np.unique(data[:,-1])
    k = len(classes)
    np.random.seed(0)
    shuffled = shuffleMatrix(data)
    train, test = splitData(shuffled, 0.66)

    #standardize data -- training
    std, mus, sigmas = standardizeDataExceptLast(train)

    #standardize data -- testing. Standardize
    #with the values obtained in training set
    std_test = standardizeTestSF(test, mus, sigmas)

    tp=tn=fp=fn = 0.

    #train k(k-1)/2 classifiers
    numClassifiers = k * (k-1) / 2

    classifiers = list()

    pairs = combinations(classes, 2)
    for pair in pairs:
        #extract data where class is pair[0] or pair[1]
        # d = train[np.where( train[:,-1] == pair[0] or train[:,-1] == pair[1])]
        class1 = train[np.where( train[:,-1] == pair[0])]
        class2 = train[np.where( train[:,-1] == pair[1])]
        d = np.vstack([class1, class2])
        classifier = svm.trainClassifier(d[:, :-1], d[:,-1])
        classifiers.append(classifier)

    #count predictions for each class
    predictions = np.zeros([len(classes), len(classes)])
    classCounts = Counter()

    correctClassifications = 0.
    for idx, t in enumerate(test):
        if idx % 100 == 0:
            print "# test rows classified:", idx, "-- " + str(float(idx)/float(len(test)) * 100) + "% done."
        #count stats. Compare last entry in this row to KNN classification
        scores = Counter()
        for classifier in classifiers:
            c = svm.classify(classifier, [ t[:-1] ])[0]
            scores[c] += 1
        best = scores.most_common()[0]
        predictedClass = best[0]
        trueClass = t[-1]
        if predictedClass == trueClass:
            correctClassifications +=1

        classCounts[int(trueClass)-1] +=1
        predictions[int(predictedClass)-1][int(trueClass)-1] +=1
    accuracy = div(correctClassifications, len(test))
    cm = makeConfusionMatrix(predictions, len(test))
    print "Accuracy = ", accuracy
    print "Prediction counts: "
    print predictions
    print "Confusion matrix (entries are %): "
    print cm

def div(num, denom):
    if denom == 0 or num == None or denom == None:
        return None
    return float(num)/denom

def makeConfusionMatrix(preds, totalPreds):
    preds = np.copy(preds)
    preds /= totalPreds
    preds *= 100
    return preds


if __name__ == "__main__":
    main()
