Mark Klobukov
CS 383 HW 3

THE CODE ASSUMES THAT THE CSV DATA FILE IS
IN THE SAME DIRECTORY AS THE CODE FILES

Files in this zip:
  Makefile -- runs q3.py with the required parameters

  report.pdf -- answers to theory questions and discussion of results
              of the programming part.

  q2.py -- closed form linear regression. run with `python q2.py`
        Displays theta, comparison between true and predicted values,
        and prints the RMSE value.

  q3.py -- S-folds cross validation
      Specify S as a command line argument like this:
      `python q3.py 5` for 5 folds,
      `python q3.py 7` for 7 folds, etc. AND:
      `python q3.py N` for leave-one-out validation
      To execute all commands required to produce results for the report,
      just run `make`

  util.py -- utility functions for reading data,
        standardizing data, etc.

  CFLinReg.py -- function for closed form lin regression

  SFoldsLinReg.py -- function for S-folds lin regresson cross validation
