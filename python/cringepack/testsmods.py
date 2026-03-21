
from time import time
from scipy.sparse import csc_matrix
import numpy as np


class SolveMethod:
    name = 'UNNAMED'
    properties = ''

    
    def solve(self, A: csc_matrix, bs: list[np.ndarray]) -> list[np.ndarray]:
        pass

    def lu(self, A: csc_matrix) -> tuple[csc_matrix, csc_matrix]:
        pass



def compare(A: csc_matrix, bs: np.ndarray, calA: SolveMethod, calB: SolveMethod):
    # Method A
   
    nnz = A.nnz

    t0 = time()
    x0A = calA.solve(A,bs)
    TA = time() - t0


    t0 = time()
    x0B = calB.solve(A,bs)
    TB = time() - t0

    print(f'{calA.name}[{calA.properties}] vs. {calB.name}[{calB.properties}]')
    print('----------------------------------------------')
    print(f'{calA.name} = {0.001*nnz/TA :1f}')
    print(f'{calB.name} = {0.001*nnz/TB :1f}')
    for i, (xa, xb) in enumerate(zip(x0A, x0B)):
        err = np.linalg.norm(xa-xb)
        symb = "✅" if err < 1e8 else "❌"
        print(f'  {symb} b[{i}] = {err:.2e}')