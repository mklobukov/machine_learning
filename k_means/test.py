import numpy as np
from kmeans import *
matrix = np.array([[61, 120],
[65, 130],
[ 72, 250],
[63, 120],
 [62, 195],
  [62, 120],
   [60, 100],
   [ 70, 140],
   [ 70 ,160],
    [65, 132],
    [48, 75],
     [72, 175],
     [ 67 ,167],
     [69 ,140],
     [96, 285],
      [70, 172],
       [70, 185 ],
[71, 168],
 [70, 180],
  [69 ,170],
   [70 ,150],
   [ 70 ,170 ],
   [71 ,144],
   [ 66 ,140],
   [67, 175],
   [ 67, 165],
   [ 72 ,175] ])

matrix = matrix.astype(np.float64)

for row in matrix:
    print row

std_matrix = standardizeData(matrix)
for row in std_matrix:
    print row
kmeans(matrix, 2)
