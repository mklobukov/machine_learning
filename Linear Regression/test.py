import numpy as np
import matplotlib.pyplot as plt
from util import *
from CFLinReg import *

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


std = standardizeData(mat)

theta = findTheta(std[:,0], std[:,1])
print theta
