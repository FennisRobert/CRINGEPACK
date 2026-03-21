from cringepack._cringepack import Cringepack as Cringepack_RS
import numpy as np
from scipy.sparse import csc_matrix
from .testsmods import SolveMethod

class Cringepack:

    def __init__(self):
        self.rs: Cringepack_RS = Cringepack_RS()

    @staticmethod
    def _decompose(A: csc_matrix):
        indptr = A.indptr.astype(np.int64)
        rows = A.indices.astype(np.int64)
        data = A.data

        return (rows, indptr, data)
    
    def cholesky(self, A: csc_matrix) -> None:
        self.rs.cholesky(*self._decompose(A))
        
    def factorize(self, A: csc_matrix) -> None:
        self.rs.factorize(*self._decompose(A))

    def solve_b(self, b: np.ndarray) -> np.ndarray:
        return self.rs.solve_b(b)[0]
    
    def solve(self, A: csc_matrix, b: np.ndarray) -> np.ndarray:
        self.factorize(A)
        x, err = self.rs.solve_b(b)
        return x
    

class CringepackMethod(SolveMethod):
    name = 'CRINGEPACK'
    properties = ''

    def __init__(self):
        self.solver = Cringepack()
    
    def _lu(self, A: csc_matrix):
        self.solver.factorize(A)
    
    def _solve_b(self, b: np.ndarray) -> np.ndarray:
        out = self.solver.solve_b(b)
        return out
    
    def _solve(self, A: csc_matrix, bs: list[np.ndarray]) -> np.ndarray:
        self.solver.factorize(A)
        return [self.solver.solve_b(b) for b in bs]

    