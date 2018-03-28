#Mark Klobukov
#CS 383
#HW 1 Q 2
#Dimensionality Reduction via PCA
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


dataFile = "diabetes.csv"
classes = []
npregnant = []
plasma = []
bp = []
tricep = []
insulin = []
bmi = []
pedigree = []
age = []

def readFile(f):
    matrix = list()
    labels = list()
    with open(f) as dataFile:
        csvReader = csv.reader(dataFile)
        for row in csvReader:
            classes.append(row[0])
            npregnant.append(row[1])
            plasma.append(row[2])
            bp.append(row[3])
            tricep.append(row[4])
            insulin.append(row[5])
            bmi.append(row[6])
            pedigree.append(row[7])
            age.append(row[8])

            labels.append(float(row[0]))
            matrix.append(map(float, row[1:9]))
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
    return proj_matrix


def plotData(class1, class_1):
    colors = (0,0,0)
    plt.scatter(class1[:,0], class1[:,1], c='r', alpha=0.5)

    plt.scatter(class_1[:,0], class_1[:,1], c='b', alpha=0.5)
    plt.title('PCA Projected Data')
    # plt.xlabel('Feature 1')
    # plt.ylabel('Feature 2')
    plt.show()

def getDataInefficient(matrix, labels):
    class_1 = list()
    class1 = list()
    for i in range(0, len(labels)):
        if labels[i] == -1:
            class_1.append(matrix[i,:])
        elif labels[i] == 1:
            class1.append(matrix[i,:])
    print "First few inputs: ", matrix[0:5,:]
    print "Class 1 zero: ", class1[0]
    print "Class -1 zero: ", class_1[0]
    return np.array(class1), np.array(class_1)

def main():
    matrix, labels = readFile(dataFile)
    matrix = standardizeData(np.copy(matrix))

    print "start fitting"
    pca = PCA(n_components=2)
    pca.fit(matrix)
    print pca.components_
    print "end fitting"

    projected_matrix = reducePCA(matrix)


    print projected_matrix[0]
    print projected_matrix[1]



    # newPCA = testPCA.project()s


    class1, class_1 = getDataInefficient(projected_matrix, labels)

    plotData(class1, class_1)




if __name__ == "__main__":
    main()
