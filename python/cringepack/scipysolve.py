from scipy.sparse.linalg import splu, spsolve_triangular
from scipy.sparse import csc_matrix
from enum import Enum
from typing import Literal
import numpy as np

from .testsmods import SolveMethod

def scipy_lu(A: csc_matrix) -> tuple[csc_matrix, csc_matrix]:
    lu = splu(A)
    return (lu.L, lu.U)


def scipy_solve(A: csc_matrix, b: np.ndarray) -> np.ndarray:
    lu = splu(A)
    return lu.solve(b)


class SuperLUNatural(SolveMethod):

    name = 'SuperLU'
    properties = 'NAT-ORD'
    istruth = True
    def __init__(self):
        self.luf = None

    def _solve(self, A: csc_matrix, bs: list[np.ndarray]):
        lu = splu(A, permc_spec='NATURAL')
        return [lu.solve(b) for b in bs]
    
    def _lu(self, A: csc_matrix):
        self.luf = splu(A, permc_spec='NATURAL')
    
    def _solve_b(self, b: np.ndarray):
        return self.luf.solve(b)
    
class SuperLUOpt(SolveMethod):

    name = 'SuperLUOpt'
    properties = 'MMD Aᵀ+A'
    istruth = False
    def __init__(self):
        self.luf = None

    def _solve(self, A: csc_matrix, bs: list[np.ndarray]):
        lu = splu(A, permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.001)
        return [lu.solve(b) for b in bs]
    
    def _lu(self, A: csc_matrix):
        self.luf = splu(A, permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.001)
    
    def _solve_b(self, b: np.ndarray):
        return self.luf.solve(b)
    