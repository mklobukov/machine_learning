import numpy as np
import csv
import math
np.set_printoptions(suppress=True)

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

def getMeans(matrix):
    return np.mean(matrix, axis=0)

def getStds(matrix):
    return np.std(matrix, axis=0)

#standardize except nth column
def standardizeDataExceptLast(matrix_in):
    matrix = np.copy(matrix_in)
    matrix = np.delete(matrix, matrix_in.shape[1]-1, axis=1)
    for i in range(0, len(matrix[0])):
        std = np.std(matrix[:,i])
        mean = np.mean(matrix[:,i])
        for j in range(0, len(matrix[:,i])):
            matrix[j][i] -= mean
            if (std != 0):
                matrix[j][i] /= std
    #at this point, all data is normalized. Return matrix
    lastCol = np.asmatrix(matrix_in[:, matrix_in.shape[1]-1]).T
    # print "SHAPES: ", matrix.shape, lastCol.shape
    matrix = np.append(matrix, lastCol, 1)
    return matrix



def standardizeTestSF(matrix_in, sigmas, mus):
    lastCol = np.asmatrix(matrix_in[:, matrix_in.shape[1]-1]).T
    matrix = np.copy(matrix_in)
    matrix = np.delete(matrix, matrix_in.shape[1]-1, axis=1)

    for idx,mu in enumerate(mus):
        # print "This mu: ", mu
        # m = np.full((len(sigmas), 1), mu, np.float32)
        matrix[:,idx] -= mu

    for idx,sigma in enumerate(sigmas):
        if sigma != 0:
            matrix[:,idx] /= sigma

    matrix  = np.append(matrix, lastCol, 1)
    return matrix

#read file assuming class labels are last column,
#no column titles
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

#read file assuming first column is indexes
#and each col has a title
def readFile2(f):
    matrix = list()
    labels = list()
    with open(f) as dataFile:
        next(dataFile)
        csvReader = csv.reader(dataFile)
        for row in csvReader:
            matrix.append(map(float, row[1:]))
    matrix = np.array(matrix)
    matrix = matrix.astype(np.float64)
    return matrix

def RMSE(X, Y):
    # print" Operands: "
    # print X, Y
    square = np.square(X-Y)
    # print "Step0: squre: ", square

    summ = np.sum(np.square(X-Y))
    # print "Step1, sum of sq: ", summ
    summ /= len(X)
    # print "Step2, div by N: ", summ
    summ = pow(summ, 0.5)
    # print "Step3, sqrt", summ

    return summ

#split data into training and testing
#frac -- fraction to be used for training
#so 1-frac is used for testing
def splitData(matrix,frac):
    nrows = matrix.shape[0]
    train = int(math.ceil(nrows * frac))

    trainingSet = matrix[0:train, :]
    testingSet = matrix[train:, :]

    return trainingSet, testingSet

def shuffleMatrix(matrix):
    m = np.copy(matrix)
    np.random.shuffle(m)
    return m

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
