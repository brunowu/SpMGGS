# SpMGG (Sparse Matrix Generator with Given Spectrum)

Eigenvalues known sparse matrix generator (Parallel version based on MPI and PETSc library)

Verification:

Verification of generated matrix to keep the given spectrum by Shifted Inverse Power Method.

error = ||Av - lambda v|| / ||Av||
