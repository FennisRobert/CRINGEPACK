from cringepack import SuperLUNatural, CringepackMethod, run_tests, SuperLUOpt, SparseMatrix



from matrices.small import SET_ALL
from matrices.big import SET_EM


SuperLU = SuperLUNatural()
CRINGEPACK = CringepackMethod()
SuperLUOpt = SuperLUOpt()

#run_tests(SET_ALL + SET_EM, [CRINGEPACK, SuperLU, SuperLUOpt])
N = 10
run_tests([SparseMatrix.gen_random(N, 4, 0.3, 'SPD') for _ in range(1)], [CRINGEPACK, SuperLU, SuperLUOpt])
#run_tests([SparseMatrix.gen_random(100, 4, 0.02, 'SPD') for _ in range(10)], [CRINGEPACK, SuperLU, SuperLUOpt])