import numpy as np
np.set_printoptions(suppress=True)

#all these theta's don't look good,
#I'll need a prettier way to do it.
def findTheta(X, Y):
    theta = np.dot(np.transpose(X), X)
    theta = np.linalg.inv(theta)
    theta = np.dot(theta, np.transpose(X))
    theta = np.dot(theta, Y)
    return theta
