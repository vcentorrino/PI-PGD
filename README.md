# PI-PGD
This repository contains the code needed to reproduce the simulations presented in the paper titled _Proximal Gradient Dynamics and Feedback Control for Equality-Constrained Composite Optimization_ by Veronica Centorrino, Francesca Rossi, Prof. Giovanni Russo, and Prof. Francesco Bullo ([arXiv](https://arxiv.org/abs/2503.15093)).

The repository contains the following files:

* Ex_constrained_lasso.py with the necessary code to reproduce the figures featured in the Numerical Experiments paragraph in the Equality-constrained LASSO Section of the manuscript.

* Ex_nonlinear_constrained_lasso.py with the necessary code to reproduce the figures featured in the Numerical Experiments paragraph in the Nonlinear-equality-constrained LASSO Section of the manuscript.

* Entropic_reg_ot folder containing:
  * 2025b-entropic_reg_ot.py with the necessary code to reproduce the figures featured in Appendix B of the extended technical version of the manuscript.
           The part of the code for stippling the points is from the open source [URL](https://github.com/ncassereau/pictures-optimal-transport/tree/master). Please note we are interested in testing our method for solving Entropic-Regularized Optimal Transport, and therefore we compare it with the Sinkhorn algorithm.
  * pi_pgd.gif and sinkhorn.gif show the complete transformations of number 4 into number 1.
  * mnist_data_set.npz containing the stippled version of the MNIST dataset used for the simulation.


**Authors of the final code and figures: Veronica Centorrino and Francesca Rossi**.


## References
[1] V. Centorrino, F. Rossi, F. Bullo, and G. Russo _Proximal Gradient Dynamics and Feedback Control for Equality-Constrained Composite Optimization_, 2025, [URL](https://arxiv.org/abs/2503.15093).
