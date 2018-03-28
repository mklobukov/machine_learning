###
import numpy as np

Sw = np.array([[4.9223, -2.1803],
[-2.1803, 8.4526]])

mu1 = np.array([-0.6386, 0.2340])
mu2 = np.array([0.6383, -0.2340])

mu1 = np.reshape(mu1, (1,2))
mu2 = np.reshape(mu2, (1,2))

print mu1, mu2
Sb = np.dot(np.transpose(mu1-mu2), (mu1-mu2))
print Sb
