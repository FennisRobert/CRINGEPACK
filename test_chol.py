from cringepack import SuperLUNatural, CringepackMethod, run_tests, SuperLUOpt, SparseMatrix



from matrices.small import MAT_DAVIS_CHOL


CRINGEPACK = CringepackMethod()

CRINGEPACK.solver.cholesky(MAT_DAVIS_CHOL.A)