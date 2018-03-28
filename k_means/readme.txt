Mark Klobukov
CS383
2/1/2018
HW2: Clustering, k-means

The directory contains:
  --Mark_Klobukov_HW2.pdf

PDF Writeup with results of the assignment.

  --kmeans.py

Code for Part 2 of the assignment.
k-means for k = 2.
Displays three figures: initial setup,
clustering after first iteration,
clustering after final iteration

Prints sizes of clusters and purity value.
Assumes `diabetes.csv` is in the same directory.
How to run: `python kmeans.py`

  --kmeans_p3.py

Code for Part 3 of the assignment.
Flexible k-means for k between 1 and 7

Accepts three arguments:
1) value of k;
2) feature column number to display on x-axis when plotting
3) feature column number to display on y-axis when plotting

Displays a figure with clustering
after final iteration. Prints
number of observations per cluster.
Example commands: `python kmeans_p3.py 5 1 2` will
use k=5 and plot Feature 1 on x-axis and Feature 2 on y-axis

`python kmeans_py.py 2 6 7` will
use k = 2 and plot Feature 6 on x-axis and Feature 7 on y-axis

  --util.py
Contains generic utility functions such as read data,
standardize data, etc.

  --kmeans_util.py
Contains helper functions for k-means algorithm.
They were taken out of the file with the main function
to avoid clutter and thus improve readability.
