# MATLAB-Sparse-Representation-for-Face-Recognition
MATLAB implementation of Sparse Representation Classification (SRC) for Face Recognition. Compares OMP, FISTA, and l1_ls (Truncated Newton Interior Point method) solvers on the ORL face dataset with noise robustness analysis.

## Project Overview
This project explores **Matrix and Tensor techniques** for data science, specifically focusing on **Sparse Representation Classification (SRC)** applied to Face Recognition.
The goal is to classify images from the **ORL Face Database** by solving optimization problems involving $l_0$ and $l_1$ norms.

The project compares three different numerical solvers:
1.  **OMP (Orthogonal Matching Pursuit):** A greedy algorithm solving the $l_0$-minimization problem.
2.  **l1_ls (Truncated Newton Interior-Point Method):** A specialized solver for large-scale $l_1$-regularized least squares.
3.  **FISTA (Fast Iterative Shrinkage-Thresholding Algorithm):** A gradient-based method with momentum for convex optimization.
