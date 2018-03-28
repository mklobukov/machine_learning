import numpy as np
import csv

def standardizeData(matrix):
    #for each column, find std and mean
    #subtract mean from each entry and
    #divide by std
    matrix = np.copy(matrix)
    for i in range(0, len(matrix[0])):
        std = np.std(matrix[:,i])
        mean = np.mean(matrix[:,i])
        for j in range(0, len(matrix[:,i])):
            matrix[j][i] -= mean
            matrix[j][i] /= std
    #at this point, all data is normalized. Return matrix
    return matrix


def readFile(f):
    matrix = list()
    labels = list()
    with open(f) as dataFile:
        csvReader = csv.reader(dataFile)
        for row in csvReader:
            labels.append(float(row[0]))
            matrix.append(map(float, row[1:]))
    matrix = np.array(matrix)
    matrix = matrix.astype(np.float64)
    return matrix, labels

def L1_dist(A, B):
    summ = 0
    for (a,b) in zip(A,B):
        summ += abs(a - b)
    return summ

def L2_norm(A, B):
    if len(A) != len(B):
        return None
    summ = 0
    for a, b in zip(A, B):
        summ += pow((a-b), 2)
    return pow(summ, 0.5)

def pp(inp):
    print "\n*******"
    for row in inp:
        print row
    print "*******\n"

def extractColumns(matrix, cols):
    newMatrix = np.zeros([matrix.shape[0], len(cols)])

    for idx,col in enumerate(cols):
        newMatrix[:,idx] = matrix[:,col]
    return newMatrix
