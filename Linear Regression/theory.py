import numpy as np
np.set_printoptions(suppress=True)
mat = np.array([
[-2, 1],
[-5., -4.],
[-3.,1. ],
[0.,3.],
[-8., 11.],
[-2., 5.],
[1., 0.],
[5., -1.],
[-1., -3.],
[6., 1.]])

mean0 = np.mean(mat[:,0])
std0 = np.std(mat[:,0])
print "STD0: ", std0


mean1= np.mean(mat[:,1])
std1 = np.std(mat[:,1])

col0 = (mat[:,0] - mean0) / std0
print col0

col1 = (mat[:,1] - mean1) / std1
print col1

stdmat = np.array([
[1, -0.2602],
[1, -0.9697],
[1, -0.4967],
[1, 0.2129],
[1, -1.6792],
[1, -0.2602],
[1, 0.4494],
[1, 1.3954],
[1, -0.0237],
[1, 1.6319]
])

y = np.array([
[-0.0936],
[-1.2635],
[-0.0936],
[0.3744],
[2.2462],
[0.8423],
[-0.3276],
[-0.5615],
[-1.0295],
[-0.0936]
])

x = np.zeros([10, 2])
x[:,0] = stdmat[:,1]
x[:,1] = np.ones(len(y)).T

m, c = np.linalg.lstsq(x, mat[:,1])[0]
print "LNINAGL"
print m ,c

xt = np.copy(stdmat)
xt = np.transpose(xt)

xtx = np.dot(xt, stdmat)
print xtx

inv = np.linalg.inv(xtx)

print "inv", inv

stl = np.dot(inv, xt)
print "sec to last"
print stl

theta = np.dot(stl, y)
print theta
