#Mark Klobukov
#CS 383 HW2
#Basic k means clustering
import numpy as np
import matplotlib.pyplot as plt
import random
from util import *

dataFile = "diabetes.csv"
epsilon = pow(2, -23)

def makeSets(k):
    sets = list()
    for i in range(0,k):
        newSet = set()
        sets.append(newSet)
    return sets

def kmeans(matrix, k):
    #assuming each col is a var
    #each row is an observation
    #randomly select 2 instances
    #and use for initial seeds
    indices = range(0, matrix.shape[0])
    random.seed(0)
    random.shuffle(indices)

    idx = list()
    for i in range(0,k):
        idx.append(indices[i])
    del indices

    print("selected indices: "), idx
    #select data rows corresponding
    #to the idx array
    centers = np.zeros([len(idx), matrix.shape[1]])
    #put initial ref vectors in the centers matrix
    #each row == vector
    for i, _ in enumerate(centers):
        centers[i,:] = matrix[idx[i], :]
    print "Initial ref vectors: ", centers

    ####GOOD up to here

    #create a set of row indexes for each ref vector
    #these sets represent clusters
    #sets = makeSets()


    #experimentation
    # centers[0,:] = matrix[0,:]
    # centers[1,:] = matrix[1,:]
    # ####
    counter = 0
    stopLoop = False

    plt.figure(0)
    plt.scatter(matrix[:,1], matrix[:,0], c='r',marker='x')
    plt.scatter(centers[0][1], centers[0][1], c='b', marker='x')
    plt.scatter(centers[1][0], centers[1][1], c='b', marker='x')
    plt.show()
    #iterations of algorithm
    while (not stopLoop):
        #for each column
        sets = makeSets(k)
        for index_row,obs in enumerate(matrix):
            #calculate the distance between each ref vector and obs
            minDist = [float('inf'), None] #distance and index
            for index_center, center in enumerate(centers):
                dist = L2_norm(obs, center)
                if dist < minDist[0]:
                    minDist = [dist, index_center]
            #append this obs to the given ref vector's cluster
            sets[minDist[1]].add(index_row)
        counter += 1

        if counter == 0:
            c1 = list()
            for s in sets[0]:
                c1.append(matrix[s,:])

            c2 = list()
            for s in set[1]:
                c2.append(matrix[s,:])
            for c in c1:
                plt.scatter

        #end for loop
        #recalculate the cluster's center
        oldCenters = np.copy(centers)
        newCenters = findAverage(matrix, sets)
        print "Old centers, new centers"
        pp(oldCenters)
        pp(newCenters)
        stopLoop = checkTerminate(oldCenters, newCenters, epsilon)
        if not stopLoop:
            centers = np.copy(newCenters)
    #end while loop
    print "COUNTER: ", counter
    return sets


def checkTerminate(oldCenters, newCenters, threshold):
    summ = 0
    for (old, new) in zip(oldCenters, newCenters):
        summ += L1_dist(old, new)
    print "CHECK TERMINATE: Summ == ", summ
    return (summ < threshold)

def findAverage(matrix,sets):
    #assuming rows are vectors
    centers = np.zeros([len(sets), matrix.shape[1]])
    for i,s in enumerate(sets):
        setSum = 0.
        for idx in s:
            setSum += matrix[idx,:]
        if (len(s) > 0):
            setSum /= len(s)
        centers[i,:] = setSum
    return centers

def main():
    k = 2
    matrix, labels = readFile(dataFile)
    matrix = extractColumns(matrix,[5,6])
    print "MATRIX original"
    pp(matrix)
    matrix = standardizeData(np.copy(matrix))
    print "MATR STAND: ",
    pp(matrix)
    print matrix.shape
    sets = kmeans(matrix, k)
    print sets[0]
    print sets[1]
    print len(sets[0]), len(sets[1])


if __name__ == "__main__":
    main()
