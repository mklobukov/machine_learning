import numpy as np

data = np.array([[4,1,2],
[2,4,0],
[2,3,-8],
[3,6,0],
[4,4,0],
[9,10,1],
[6,8,-2],
[9,5,1],
[8,7,10],
[10,8,-5]])


def main():
    print data
    X = np.array(data)
    X = X.astype(np.float32)

    stdd = standardizeData(X)
    print stdd

    cov = np.cov(stdd, rowvar=False)
    print cov

    eigenval, eigenvec = np.linalg.eig(cov)
    print eigenval
    print eigenvec

    test = np.dot(stdd[0], eigenvec[:,0])
    print test

def standardizeData(matrix):
    # matrix =
    print "Standardizing matrix with shape: ", matrix.shape
    #for each column, find std and mean
    #subtract mean from each entry and
    #divide by std
    for i in range(0, len(matrix[0])):
        std = myStd(matrix[:,i])
        mean = np.mean(matrix[:,i])
        print "std and mean = ", std, mean
        for j in range(0, len(matrix[:,i])):
            print "mat - mean: ", matrix[j][i], "-", mean
            matrix[j][i] = float(matrix[j][i]) - mean
            matrix[j][i] = float(myDivide(matrix[j][i], std))
    #at this point, all data is normalized. Return matrix
    return matrix

def myStd(array):
    array = np.array(array)
    mean = np.mean(array)

    summ = 0
    for el in array:
        summ += pow(el-mean, 2)
    summ /= (len(array) -1)
    summ = pow(summ, 0.5)
    return summ

def myDivide(num, denom):
    if denom == 0:
        return numdddddd
    return float(num)/float(denom)


if __name__ == "__main__":
    main()
