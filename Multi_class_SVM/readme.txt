Mark Klobukov
CS 383 HW 5 SVMs
2/28/2018

All libraries used in this assignment are installed in Tux.
No additional installations are needed. Code should run as provided.

This ZIP contains:
For the following three files, the execution command follows the pattern:
  `python filename.py path_to_data_file`
  * part1.py -- binary SVM. How to run: `python part1.py spambase.data`
  * part2.py -- multi-class SVM. How to run: `python part2.py CTG.csv`
  * part3.py -- multi-class SVM. Same as part 2 but also prints
              the confusion matrix.
              How to run: `python part3.py CTG.csv`
  * Makefile -- simplification of commands to reproduce results for
    this assignment. Assumes data files are in the same directory.
    Execute:
      `make 1` -- for part 1 of the assignment
      `make 2` -- for part 2 of the assignment
      `make 3` -- for part 3 of the assignment
  * mark_klobukov_hw5.pdf -- report with results and explanations

  *util.py -- utility functions such as reading file, standardizing data, etc.
  *svm.py -- wrapper for calling the SVM library from SKLearn.

  *CTG.csv and spambase.data -- data files used in the assignment

  
