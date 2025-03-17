# PI-PGD
This repository contains the extended technical report of the paper “Proximal Gradient Dynamics and Feedback Control for Equality-Constrained Composite Optimization” by Veronica Centorrino, Francesca Rossi, Prof. Giovanni Russo, and Prof. Francesco Bullo, and the code needed to reproduce the simulations there presented.

The repository contains the following files:

* 2025-extended_version_pi-pgd.pdf with the extended technical report of the paper “Proximal Gradient Dynamics and Feedback Control for Equality-Constrained Composite Optimization”.

* 2025a-Ex_constrained_lasso.py with the necessary code to reproduce the figures featured in the Numerical Experiments paragraph in the Equality-constrained LASSO Section of the manuscript.

* Entropic_reg_ot folder containing:
  * 2025b-entropic_reg_ot.py with the necessary code to reproduce the figures featured in the Numerical Experiments paragraph in the Entropic-Regularized Optimal Transport of the manuscript.
           The part of the code for stippling the points is from the open source [URL](https://github.com/ncassereau/pictures-optimal-transport/tree/master). Please note we are interested in testing our method for solving Entropic-Regularized Optimal Transport, and therefore we compare it with the Sinkhorn algorithm.
  * pi_pgd.gif and sinkhorn.gif show the complete transformations of number 4 into number 1.
  * mnist_data_set.npz containing the stippled version of the MNIST dataset used for the simulation.


**Authors of the final code and figures: Veronica Centorrino and Francesca Rossi**.
