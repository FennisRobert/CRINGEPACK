from cringepack import SuperLUNatural, CringepackMethod, run_tests, SuperLUOpt, SparseMatrix



from matrices.small import SET_SPD

mat = SET_SPD[0]
print(mat.A.toarray().real)

CRINGEPACK = CringepackMethod()
SUPERLU = SuperLUOpt()

run_tests(SET_SPD, [CRINGEPACK, SUPERLU])