Author: Mark Klobukov
1/26/2018
CS 383 HW 1
Professor Matthew Burlick

All code tested on Tux.
Plots get displayed with X11 tunneling enabled (while running a program like Xming)

Contents of this folder:
  1) Mark Klobukov HW1.pdf --- PDF file with answers to theory questions and result of programming problems

  2) q2.py --- Python code for Dimensionality Reduction via PCA Problem. Assumes the file `diabetes.cvs` is in the same directory
    The code will display the plot of data after the dimensionality is reduced to 2D. 
    How to run: execute `python q2.py`.

  3) q3.py --- Python code for the Eigenfaces problem. Assumes the directory `yalefaces` is present in the same directory.
    How to run: execute `python q3.py`. The value of K (33) will be displayed in the terminal.
    The following pictures will be displayed:
      Visualization of the first Principal Component
      Original Image #1
      Reconstruction of this image from projection onto one PC
      Reconstruction of this image from projection onto k PCs.
