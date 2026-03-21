from __future__ import annotations
from scipy.sparse import csc_matrix, load_npz
import numpy as np
from typing import Literal

PRE = './matrices/'

MTYPES = Literal['Symmetric','Unsymmetric','SPD']

class SparseMatrix:
    ctr: int = 0
    def __init__(self, A: csc_matrix, bs: list[np.ndarray], name: str, mtype: MTYPES = 'Unsymmetric'):
        self.A: csc_matrix = A
        self.bs: csc_matrix = bs
        self.name: str = name
        self.mtype: str = mtype

    @staticmethod
    def from_file(name: str, filename: str, n_bvec: int, mtype: MTYPES = 'Unsymmetric') -> SparseMatrix:

        A = load_npz(PRE+filename+'_A.npz')
        bs = [np.load(f'{PRE}{filename}_b{i}.npy') for i in range(n_bvec)]
        return SparseMatrix(A, bs, name, mtype = mtype)

    @staticmethod
    def gen_random(Ndim: int, nb: int, fill_ratio: float = 0.01, mtype: MTYPES = 'Unsymmetric') -> SparseMatrix:
        rng = np.random.default_rng(Ndim * 1000 + nb)

        if mtype == 'SPD':
            # Hermitian positive definite: A = B @ B^H + diag
            nnz_per_row = max(1, int(Ndim * fill_ratio))
            B = np.zeros((Ndim, Ndim), dtype=np.complex128)
            for i in range(Ndim):
                cols = rng.choice(Ndim, size=nnz_per_row, replace=False)
                for j in cols:
                    B[i, j] = rng.uniform(-1, 1) + 1j * rng.uniform(-1, 1)
            A = B @ B.conj().T + Ndim * np.eye(Ndim, dtype=np.complex128)
            A = csc_matrix(A)

        elif mtype == 'Symmetric':
            # Complex symmetric: A = A^T (not hermitian)
            A = np.zeros((Ndim, Ndim), dtype=np.complex128)
            for i in range(Ndim):
                A[i, i] = Ndim + rng.uniform(-2, 2) + 1j * rng.uniform(-2, 2)
            for i in range(Ndim):
                for j in range(i + 1, Ndim):
                    if rng.random() < fill_ratio:
                        val = rng.uniform(-2, 2) + 1j * rng.uniform(-1, 1)
                        A[i, j] = val
                        A[j, i] = val  # symmetric: A^T = A
            A = csc_matrix(A)

        else:
            # Unsymmetric
            A = np.zeros((Ndim, Ndim), dtype=np.complex128)
            for i in range(Ndim):
                A[i, i] = Ndim + rng.uniform(-2, 2) + 1j * rng.uniform(-2, 2)
            for i in range(Ndim):
                for j in range(Ndim):
                    if i != j and rng.random() < fill_ratio:
                        A[i, j] = rng.uniform(-2, 2) + 1j * rng.uniform(-1, 1)
            A = csc_matrix(A)

        bs = []
        for k in range(nb):
            bs.append(rng.uniform(-5, 5, Ndim) + 1j * rng.uniform(-5, 5, Ndim))
        
        name = f'Rand_{mtype[:3]}_{Ndim}x{Ndim}_{nb}_{SparseMatrix.ctr}rhs'
        SparseMatrix.ctr += 1
        return SparseMatrix(A, bs, name, mtype=mtype)