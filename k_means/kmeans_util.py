import numpy as np
from util import *
import matplotlib.pyplot as plt

colors = ['r', 'b', 'c', 'y', 'g', 'm', 'k']

def makePlot(matrix, sets, centers, counter, cols):
    #I don't like the space complexity of this
    #copying rows isn't most optimal
    #but i may not have time to improve this
    x = cols[0]-1
    y = cols[1]-1

    clusters = list()
    for st in sets:
        newList = list()
        for el in st:
            newList.append(matrix[el,:])
        newList = np.asarray(newList)
        clusters.append(newList)

    clusters = np.asarray(clusters)
    plt.figure()
    #plot clusters
    for idx,cluster in enumerate(clusters):
        plt.scatter(cluster[:, x], cluster[:,y], c=colors[idx], marker='x')

    #plot centers
    for idx,center in enumerate(centers):
        plt.scatter(center[x], center[y], s=200, c=colors[idx], edgecolors='k')

    plt.suptitle("Clustering for k = " + str(len(centers)) + " after " + str(counter) + " iterations. Features plotted: " + str(cols[0]) + ", " + str(cols[1]))


    #
    # c1 = list()
    # for s in sets[0]:
    #     c1.append(matrix[s,:])
    #
    # c2 = list()
    # for s in sets[1]:
    #     c2.append(matrix[s,:])
    #
    # c1 = np.asarray(c1)
    # c2 = np.asarray(c2)
    #
    # plt.figure()
    #
    # plt.scatter(c1[:,1], c1[:,0], c='b', marker='x')
    # plt.scatter(c2[:,1], c2[:,0], c='r', marker='x')
    # plt.scatter(centers[0][1], centers[0][0], s=100, c='b', edgecolors='k')
    # plt.scatter(centers[1][1], centers[1][0], s =100,c= 'r', edgecolors='k')
    # plt.suptitle('Final Clustering After ' + str(counter) + ' iterations')
    plt.show()

def checkTerminate(oldCenters, newCenters, threshold):
    summ = 0
    for (old, new) in zip(oldCenters, newCenters):
        summ += L1_dist(old, new)
    #print "CHECK TERMINATE: Summ == ", summ
    return (summ < threshold)

def makeSets(k):
    #create a set of row indexes for each ref vector
    #these sets represent clusters
    sets = list()
    for i in range(0,k):
        newSet = set()
        sets.append(newSet)
    return sets

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
    print clusterAssignments

    #now cluster 0 has its clusterAssignment
    #stored in clusterAssignments[0]

    #add up all cluster assignments
    #and divide by number of observations
    summ = 0.
    for a in clusterAssignments:
        summ+= a[1]
    summ /= len(labels)
    print "Purity: ", summ
    return summ
