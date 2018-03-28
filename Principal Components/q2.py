#Mark Klobukov
#CS 383
#HW 1 Q2
#Dimensionality Reduction via PCA
import csv
import numpy as np
import matplotlib.pyplot as plt

dataFile = "diabetes.csv"

def readFile(f):
    matrix = list()
    labels = list()
    with open(f) as dataFile:
        csvReader = csv.reader(dataFile)
        for row in csvReader:
            labels.append(float(row[0]))
            matrix.append(map(float, row[1:]))
    matrix = np.array(matrix)
    return matrix, labels

def standardizeData(matrix):
    #for each column, find std and mean
    #subtract mean from each entry and
    #divide by std
    for i in range(0, len(matrix[0])):
        std = np.std(matrix[:,i])
        mean = np.mean(matrix[:,i])
        for j in range(0, len(matrix[:,i])):
            matrix[j][i] -= mean
            matrix[j][i] /= std
    #at this point, all data is normalized. Return matrix
    return matrix

def findIdxOfNLargest(arr, N):
    return arr.argsort()[-N:][::-1]

def reducePCA(matrix):
    #find covariance matrix
    #set rowvar argument to False because features are columns, not rows
    covmat = np.cov(matrix, rowvar=False) #8x8 matrix
    #find eigenvalues/eigenvectors
    w, v = np.linalg.eig(covmat)
    #reduce to 2D - so only pick the two highest eigenvals
    eigenvals = findIdxOfNLargest(w, 2)
    evec = list()
    for idx in eigenvals:
        evec.append(v[:,idx])

    #this multiplication by -1 helps rotate the figure
    #so that it looks like the one in the assignment PDF
    #eigenvectors can be multiplied by any scalar
    #including a negative one
    #This does not change the answer
    evec = -1*np.array(evec)
    #project matrix onto evec
    proj_matrix = np.dot(matrix, np.transpose(evec))
    return proj_matrix, w, v


def plotData(class1, class_1):
    colors = (0,0,0)
    plt.scatter(class1[:,0], class1[:,1], c='r', alpha=0.5)

    plt.scatter(class_1[:,0], class_1[:,1], c='b', alpha=0.5)
    plt.title('PCA - Data Reduced to 2D')
    plt.show()

def getDataInefficient(matrix, labels):
    class_1 = list()
    class1 = list()
    for i in range(0, len(labels)):
        if labels[i] == -1:
            class_1.append(matrix[i,:])
        elif labels[i] == 1:
            class1.append(matrix[i,:])
    return np.array(class1), np.array(class_1)

def main():
    matrix, labels = readFile(dataFile)
    matrix = standardizeData(np.copy(matrix))
    projected_matrix = reducePCA(matrix)
    projected_matrix = np.matrix(projected_matrix)
    class1, class_1 = getDataInefficient(projected_matrix, labels)
    plotData(class1, class_1)

if __name__ == "__main__":
    main()
