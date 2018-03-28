Mark Klobukov
CS 383 HW 4
README

This zip folder contains:
  *** hw4_mark_klobukov.pdf -- writeup with answers to the theory questions
    and statistics for the code portion of the hw

  *** knn.py -- code that performs the KNN classification.
      How to run: python knn.py <path to datafile> <value of k>
      For example, assuming data is in the same directory and k = 5, run this:
        python knn.py spambase.data 5
      Pretty slow -- takes about a minute to finish running for the spambase.data file
      Will print % progress as the computation is happening
      At the end, prints all required statistics,
      as well as FP, TP, FN, TN

  *** util.py -- utility functions developed in previous assignments.
        Reading in the data, standardizing, etc.
        One new addition: classify_knn() function.
        Function operates on one feature vector at a time,
        Hence the for loop over all test rows
        in the main function in knn.py

  *** readme.txt -- this text file

  *** spambase.data -- including the datafile used for the assignment

  *** Makefile -- execute `make` to get the statistics required by the HW instructions
      (equivalent to python knn.py spambase.data 5)
