#Mark Klobukov
#CS 383 HW2
#FLexible k means clustering
#allow k to be between 1 and 7
import numpy as np
import matplotlib.pyplot as plt
import random
from util import *
from kmeans_util import *
import sys

dataFile = "diabetes.csv"
epsilon = pow(2, -23)

def kmeans(matrix, k, xcol, ycol):
    #assuming each col is a var
    #each row is an observation
    #k -- number of clusters
    #xcol - column to plot on x-axis
    #ycol = column to plot on y-axis
    xcol -= 1
    ycol -= 1
    #raniomly select 2 instances
    #and use for initial seeds
    indices = range(0, matrix.shape[0])
    random.seed(0)
    random.shuffle(indices)

    idx = list()
    for i in range(0,k):
        idx.append(indices[i])
    del indices

    print("Randomly selected indices for ref vectors: "), idx
    #select data rows corresponding
    #to the idx array
    centers = np.zeros([k, matrix.shape[1]])
    #put initial ref vectors in the centers matrix
    #each row == vector
    for i, _ in enumerate(centers):
        centers[i,:] = matrix[idx[i], :]
    #print "Initial ref vectors: ", centers

    counter = 0
    stopLoop = False

    # plt.figure(0)
    # plt.scatter(matrix[:,xcol], matrix[:,ycol], c='r',marker='x')
    # plt.suptitle("Initial Setup (before clustering)")
    # plt.show()
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
            pass
            #makePlot(matrix, sets, centers, counter+1, [xcol+1, ycol+1])
            #plot the initial setup

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


def main():
    if len(sys.argv) < 3:
        print "Provide 3 arguments: k, xcol, ycol. Exiting\n"
        return
    k = int(sys.argv[1])
    xcol = int(sys.argv[2])
    ycol = int(sys.argv[3])
    if k < 1 or k > 7:
        print "k out of range. Exiting\n"


    matrix,labels = readFile(dataFile)
    #matrix = extractColumns(matrix,[5,6])
    matrix = standardizeData(np.copy(matrix))
    sets, centers, niter = kmeans(matrix,k, 5, 6)
    print k, "clusters:\n"
    for i,s in enumerate(sets):
        print "Size of cluster " + str(i+1) + ": ", len(s)
    makePlot(matrix, sets, centers, niter, [xcol, ycol])



if __name__ == "__main__":
    main()
