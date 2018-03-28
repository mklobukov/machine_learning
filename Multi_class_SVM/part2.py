#Mark Klobukov
#CS383 hw 5 SVMs -- multi class
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

    correctClassifications = 0.
    for idx, t in enumerate(test):
        scores = Counter()
        for classifier in classifiers:
            c = svm.classify(classifier, [ t[:-1] ])[0]
            scores[c] += 1
        best = scores.most_common()[0]
        if best[0] == t[-1]:
            correctClassifications +=1
    accuracy = div(correctClassifications, len(test))
    print "Accuracy = ", accuracy

def div(num, denom):
    if denom == 0 or num == None or denom == None:
        return None
    return float(num)/denom


if __name__ == "__main__":
    main()
