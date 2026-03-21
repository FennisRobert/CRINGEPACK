from cringepack import compare, SuperLUNatural, CringepackMethod



from matrices.small import SET_CU
from matrices.big import SET_EM


SuperLU = SuperLUNatural()
CRINGEPACK = CringepackMethod()

for A,b in SET_CU:
    compare(A, [b,], SuperLU, CRINGEPACK)

for A,bs in SET_EM:
    compare(A, bs, SuperLU, CRINGEPACK)
        