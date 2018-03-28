#Mark Klobukov
#CS 383 HW2
#Basic k means clustering
import numpy as np
import matplotlib.pyplot as plt
import random
from util import *
from collections import Counter

dataFile = "diabetes.csv"
epsilon = pow(2, -23)

def makeSets(k):
    #create a set of row indexes for each ref vector
    #these sets represent clusters
    sets = list()
    for i in range(0,k):
        newSet = set()
        sets.append(newSet)
    return sets

def makeFinalPlot(matrix, sets, centers, counter):
    c1 = list()
    for s in sets[0]:
        c1.append(matrix[s,:])

    c2 = list()
    for s in sets[1]:
        c2.append(matrix[s,:])

    c1 = np.asarray(c1)
    c2 = np.asarray(c2)

    plt.figure()

    plt.scatter(c1[:,1], c1[:,0], c='b', marker='x')
    plt.scatter(c2[:,1], c2[:,0], c='r', marker='x')
    plt.scatter(centers[0][1], centers[0][0], s=100, c='b', edgecolors='k')
    plt.scatter(centers[1][1], centers[1][0], s =100,c= 'r', edgecolors='k')
    plt.suptitle('Final Clustering After ' + str(counter) + ' iterations')
    plt.show()

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

    #print("selected indices: "), idx
    #select data rows corresponding
    #to the idx array
    centers = np.zeros([len(idx), matrix.shape[1]])
    #put initial ref vectors in the centers matrix
    #each row == vector
    for i, _ in enumerate(centers):
        centers[i,:] = matrix[idx[i], :]
    #print "Initial ref vectors: ", centers

    counter = 0
    stopLoop = False

    plt.figure(0)
    plt.scatter(matrix[:,1], matrix[:,0], c='r',marker='x')
    plt.suptitle("Initial Setup (before clustering)")
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


        if counter == 0:
            #plot the initial setup
            c1 = list()
            for s in sets[0]:
                c1.append(matrix[s,:])

            c2 = list()
            for s in sets[1]:
                c2.append(matrix[s,:])

            c1 = np.asarray(c1)
            c2 = np.asarray(c2)
            plt.figure(1)

            plt.scatter(c1[:,1], c1[:,0], c='b', marker='x')
            plt.scatter(c2[:,1], c2[:,0], c='r', marker='x')
            plt.scatter(centers[0][1], centers[0][0], s=100, c='b', edgecolors='k')
            plt.scatter(centers[1][1], centers[1][0], s =100,c= 'r', edgecolors='k')
            plt.suptitle('Iteration 1')
            plt.show()

        #end for loop
        #recalculate the cluster's center
        oldCenters = np.copy(centers)
        newCenters = findAverage(matrix, sets)
        #print "Old centers, new centers"
        #pp(oldCenters)
        #pp(newCenters)
        stopLoop = checkTerminate(oldCenters, newCenters, epsilon)
        counter += 1
        if not stopLoop:
            centers = np.copy(newCenters)
    #end while loop
    return sets, centers, counter


def checkTerminate(oldCenters, newCenters, threshold):
    summ = 0
    for (old, new) in zip(oldCenters, newCenters):
        summ += L1_dist(old, new)
    #print "CHECK TERMINATE: Summ == ", summ
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

def findPurity(clusters, labels):
    #assign to each cluster a class
    #which is most frequent in the cluster
    clusterAssignments = list()

    for i,cluster in enumerate(clusters):
        c = Counter()
        for obsIndex in cluster:
            c[labels[obsIndex]] += 1
        clusterAssignments.append(c.most_common()[0])
    # print clusterAssignments

    #now cluster 0 has its clusterAssignment
    #stored in clusterAssignments[0]

    #add up all cluster assignments
    #and divide by number of observations
    summ = 0.
    for a in clusterAssignments:
        summ+= a[1]
    summ /= len(labels)
    return summ
def main():
    k = 2
    matrix, labels = readFile(dataFile)
    matrix = extractColumns(matrix,[5,6])
    #print "MATRIX original"
    #pp(matrix)
    matrix = standardizeData(np.copy(matrix))
    #print "MATR STAND: ",
    #pp(matrix)
    #print matrix.shape
    sets, centers, niter = kmeans(matrix, k)
    print k, "clusters:\n"
    for i,s in enumerate(sets):
        print "Size of cluster " + str(i+1) + ": ", len(s)
    makeFinalPlot(matrix, sets, centers, niter)
    print "Purity: ", findPurity(sets, labels)

if __name__ == "__main__":
    main()
