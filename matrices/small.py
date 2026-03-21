from cringepack.matrix import SparseMatrix
import numpy as np
from scipy.sparse import csc_matrix


# ============================================================
#  Small Complex Unsymmetric Matrices
# ============================================================

_MatCU_7_1 = csc_matrix(np.array([
    [ 0.0+1j,  2.0+0j,  0.0+0j,  1.0+0j,  1.0+0j,  0.0+0j,  0.0+0j,  0.0+0j],
    [ 3.0+0j,  0.0+0j,  1.0+0j,  1.0+0j,  0.0+0j,  0.0+0j,  0.0+0j,  0.0+0j],
    [ 0.0+0j,  0.0+0j,  5.0+2j,  5.0+2j,  0.0+0j,  0.0+0j,  1.0+0j,  0.0+0j],
    [ 0.0+0j,  1.0+0j,  2.0+0j,  2.0+0j,  0.0+0j,  4.0-1j,  0.0+0j,  0.0+0j],
    [ 0.0+0j,  0.0+0j,  0.0+0j,  0.0+0j,  6.0+1j,  0.0+0j,  0.0+0j, -1.0+1j],
    [ 0.0+0j,  0.0+0j,  0.0+0j,  0.0+0j,  0.0+0j,  7.0+3j,  0.0+0j,  0.0+0j],
    [ 1.0-2j,  0.0+0j,  0.0+0j,  0.0+0j,  0.0+0j,  0.0+0j,  8.0+2j,  2.0+0j],
    [ 0.0+0j,  0.0+0j,  0.0+0j,  0.0+0j,  3.0+1j,  0.0+0j,  0.0+0j,  5.0-2j],
], dtype=np.complex128))
_MatCU_7_1_b1 = np.array([1+1j, 2-1j, 3+0j, -1+2j, 4+0j, -2+1j, 1-1j, 3+2j], dtype=np.complex128)


# 8x8 complex unsymmetric, forces pivoting (zeros on diagonal)
_MatCU_8_pivot = csc_matrix(np.array([
    [ 0.0+0j,  3.0+1j,  0.0+0j,  0.0+0j,  0.0+0j,  0.0+0j,  1.0+0j,  0.0+0j],
    [ 2.0-1j,  0.0+0j,  0.0+0j,  0.0+0j,  1.0+0j,  0.0+0j,  0.0+0j,  0.0+0j],
    [ 0.0+0j,  0.0+0j,  0.0+0j,  4.0+2j,  0.0+0j,  1.0+0j,  0.0+0j,  0.0+0j],
    [ 0.0+0j,  0.0+0j,  5.0-1j,  0.0+0j,  0.0+0j,  0.0+0j,  0.0+0j,  2.0+0j],
    [ 0.0+0j,  1.0+0j,  0.0+0j,  0.0+0j,  7.0+3j,  0.0+0j,  0.0+0j,  0.0+0j],
    [ 0.0+0j,  0.0+0j,  2.0+0j,  0.0+0j,  0.0+0j,  6.0-2j,  0.0+0j,  1.0+0j],
    [ 3.0+0j,  0.0+0j,  0.0+0j,  0.0+0j,  0.0+0j,  0.0+0j,  9.0+1j,  0.0+0j],
    [ 0.0+0j,  0.0+0j,  0.0+0j,  1.0-1j,  0.0+0j,  2.0+0j,  0.0+0j,  8.0+0j],
], dtype=np.complex128))
_MatCU_8_pivot_b1 = np.array([1+2j, -1+1j, 3-1j, 2+0j, -2+3j, 1+1j, 4-2j, -1+0j], dtype=np.complex128)


# 5x5 complex unsymmetric, nearly singular (numerical stability test)
_MatCU_5_nasty = csc_matrix(np.array([
    [ 1e-8+1e-8j,  1.0+0.5j,    0.0+0j,      0.0+0j,      0.0+0j],
    [ 1.0-0.5j,    2.0+1j,      1.0+0.3j,    0.0+0j,      0.0+0j],
    [ 0.0+0j,      1.0-0.3j,    1e-8-1e-8j,  3.0+1j,      0.0+0j],
    [ 0.0+0j,      0.0+0j,      2.0+1j,      5.0+2j,      1.0-0.5j],
    [ 0.0+0j,      0.0+0j,      0.0+0j,      1.0+0.5j,    4.0-1j],
], dtype=np.complex128))
_MatCU_5_nasty_b1 = np.array([1+0.5j, 0+1j, -1+0.3j, 2+2j, 1-1j], dtype=np.complex128)


# 20x20 random sparse complex unsymmetric
def _make_random_unsym_20():
    rng = np.random.default_rng(42)
    A = np.zeros((20, 20), dtype=np.complex128)
    for i in range(20):
        A[i, i] = 10.0 + rng.uniform(-2, 2) + 1j * rng.uniform(-2, 2)
    for i in range(20):
        for j in range(20):
            if i != j and rng.random() < 0.3:
                A[i, j] = rng.uniform(-2, 2) + 1j * rng.uniform(-1, 1)
    return csc_matrix(A)

_MatCU_20_rand = _make_random_unsym_20()
_MatCU_20_rand_b1 = np.array([(-1)**i * (i + 1) + 1j * (i % 3) for i in range(20)], dtype=np.complex128)
_MatCU_20_rand_b2 = np.ones(20, dtype=np.complex128) * (1 + 1j)


# ============================================================
#  Complex Symmetric Matrices (A = A^T, NOT hermitian)
# ============================================================

# 10x10 complex symmetric
_MatCS_10_1 = csc_matrix(np.array([
    [ 5.0+1j,  1.0+0.5j, 0.0+0j,   0.0+0j,   2.0-1j,  0.0+0j,   0.0+0j,   0.0+0j,   0.0+0j,   1.0+0.2j],
    [ 1.0+0.5j,7.0+2j,   2.0+1j,   0.0+0j,   0.0+0j,  0.0+0j,   0.0+0j,   0.0+0j,   0.0+0j,   0.0+0j],
    [ 0.0+0j,  2.0+1j,   9.0-1j,   1.0+1j,   0.0+0j,  0.0+0j,   3.0+0.5j, 0.0+0j,   0.0+0j,   0.0+0j],
    [ 0.0+0j,  0.0+0j,   1.0+1j,   6.0+0.5j, 0.0+0j,  2.0+1j,   0.0+0j,   1.0+0.3j, 0.0+0j,   0.0+0j],
    [ 2.0-1j,  0.0+0j,   0.0+0j,   0.0+0j,   8.0+3j,  1.0+0.5j, 0.0+0j,   0.0+0j,   0.0+0j,   0.0+0j],
    [ 0.0+0j,  0.0+0j,   0.0+0j,   2.0+1j,   1.0+0.5j,10.0-2j,  0.0+0j,   0.0+0j,   1.0+0.7j, 0.0+0j],
    [ 0.0+0j,  0.0+0j,   3.0+0.5j, 0.0+0j,   0.0+0j,  0.0+0j,   11.0+1j,  2.0+0.5j, 0.0+0j,   0.0+0j],
    [ 0.0+0j,  0.0+0j,   0.0+0j,   1.0+0.3j, 0.0+0j,  0.0+0j,   2.0+0.5j, 7.0+1j,   3.0-1j,   0.0+0j],
    [ 0.0+0j,  0.0+0j,   0.0+0j,   0.0+0j,   0.0+0j,  1.0+0.7j, 0.0+0j,   3.0-1j,   12.0+2j,  1.0+0.1j],
    [ 1.0+0.2j,0.0+0j,   0.0+0j,   0.0+0j,   0.0+0j,  0.0+0j,   0.0+0j,   0.0+0j,   1.0+0.1j, 6.0-1j],
], dtype=np.complex128))
_MatCS_10_1_b1 = np.array([1+1j, -2+0.5j, 3+2j, 0-1j, 2+3j, -1+1j, 4+0.5j, 1-2j, -3+1j, 2+2j], dtype=np.complex128)
_MatCS_10_1_b2 = np.array([0+1j, 1+1j, -1+0.5j, 2-2j, 0+0.3j, 3+1j, -2+3j, 1+0.5j, 0-1j, -1+2j], dtype=np.complex128)


# 6x6 complex symmetric, block-like structure
_MatCS_6_block = csc_matrix(np.array([
    [ 8.0+2j,   1.0+1j,   0.0+0j,   2.0+0.5j, 0.0+0j,   0.0+0j],
    [ 1.0+1j,   6.0-1j,   3.0+0.5j, 0.0+0j,   0.0+0j,   0.0+0j],
    [ 0.0+0j,   3.0+0.5j, 10.0+1j,  0.0+0j,   1.0+0.3j, 0.0+0j],
    [ 2.0+0.5j, 0.0+0j,   0.0+0j,   7.0+3j,   2.0+1j,   0.0+0j],
    [ 0.0+0j,   0.0+0j,   1.0+0.3j, 2.0+1j,   9.0-2j,   1.0+0.5j],
    [ 0.0+0j,   0.0+0j,   0.0+0j,   0.0+0j,   1.0+0.5j, 5.0+1j],
], dtype=np.complex128))
_MatCS_6_block_b1 = np.array([1+1j, 2-1j, -1+2j, 3+0j, 0+1j, -2+1j], dtype=np.complex128)


# 12x12 complex symmetric block tridiagonal (FEM-like)
def _make_sym_block_tridiag_12():
    A = np.zeros((12, 12), dtype=np.complex128)
    for i in range(4):
        r = i * 3
        A[r, r] = 6.0 + 1j * (i + 1)
        A[r, r+1] = 1.0 + 0.5j
        A[r+1, r] = 1.0 + 0.5j  # symmetric: same value
        A[r+1, r+1] = 8.0 - 1j * i
        A[r+1, r+2] = 2.0 + 0.3j
        A[r+2, r+1] = 2.0 + 0.3j
        A[r+2, r+2] = 10.0 + 2j
        if i < 3:
            c = (i + 1) * 3
            A[r+2, c] = -1.0 + 0.3j
            A[c, r+2] = -1.0 + 0.3j
            A[r+1, c+1] = -0.5 + 0.2j
            A[c+1, r+1] = -0.5 + 0.2j
    return csc_matrix(A)

_MatCS_12_btridiag = _make_sym_block_tridiag_12()
_MatCS_12_btridiag_b1 = np.array([1+0j, 0+1j, -1+2j, 2+0j, -1+1j, 3+0j, 0-1j, 1+2j, -2+0j, 1+1j, 0+0j, -1+3j], dtype=np.complex128)
_MatCS_12_btridiag_b2 = np.ones(12, dtype=np.complex128) * (1 + 0.5j)


# ============================================================
#  Complex SPD Matrices (A = A^H, positive definite)
# ============================================================

# 6x6 hermitian positive definite
def _make_hpd_6():
    rng = np.random.default_rng(123)
    B = rng.uniform(-1, 1, (6, 6)) + 1j * rng.uniform(-1, 1, (6, 6))
    A = B @ B.conj().T + 6.0 * np.eye(6, dtype=np.complex128)
    return csc_matrix(A)

_MatSPD_6 = _make_hpd_6()
_MatSPD_6_b1 = np.array([1+1j, 2-1j, -1+2j, 3+0j, 0+1j, -2+1j], dtype=np.complex128)


# 10x10 hermitian positive definite, sparser
def _make_hpd_10():
    rng = np.random.default_rng(456)
    B = np.zeros((10, 10), dtype=np.complex128)
    for i in range(10):
        for j in range(max(0, i-2), min(10, i+3)):
            B[i, j] = rng.uniform(-1, 1) + 1j * rng.uniform(-0.5, 0.5)
    A = B @ B.conj().T + 10.0 * np.eye(10, dtype=np.complex128)
    return csc_matrix(A)

_MatSPD_10 = _make_hpd_10()
_MatSPD_10_b1 = np.array([1+0.5j, -1+1j, 2+0j, 0-1j, 3+2j, -2+1j, 1+1j, 0+0j, -1+3j, 2-1j], dtype=np.complex128)
_MatSPD_10_b2 = np.array([0+1j, 1+0j, -1+1j, 2+2j, -1+0j, 0+3j, 1-1j, -2+1j, 1+0j, 0-2j], dtype=np.complex128)


# 8x8 hermitian positive definite, diagonal dominant
_MatSPD_8_diag = csc_matrix(np.array([
    [20.0+0j,   1.0+1j,   0.0+0j,   0.0+0j,   2.0-1j,  0.0+0j,   0.0+0j,   1.0+0.5j],
    [ 1.0-1j,  18.0+0j,   2.0+0.5j, 0.0+0j,   0.0+0j,  0.0+0j,   0.0+0j,   0.0+0j],
    [ 0.0+0j,   2.0-0.5j, 22.0+0j,  1.0+1j,   0.0+0j,  0.0+0j,   3.0+0.5j, 0.0+0j],
    [ 0.0+0j,   0.0+0j,   1.0-1j,  16.0+0j,   0.0+0j,  2.0+1j,   0.0+0j,   1.0+0.3j],
    [ 2.0+1j,   0.0+0j,   0.0+0j,   0.0+0j,  19.0+0j,  1.0+0.5j, 0.0+0j,   0.0+0j],
    [ 0.0+0j,   0.0+0j,   0.0+0j,   2.0-1j,   1.0-0.5j,21.0+0j,  0.0+0j,   0.0+0j],
    [ 0.0+0j,   0.0+0j,   3.0-0.5j, 0.0+0j,   0.0+0j,  0.0+0j,  17.0+0j,   2.0+0.5j],
    [ 1.0-0.5j, 0.0+0j,   0.0+0j,   1.0-0.3j, 0.0+0j,  0.0+0j,   2.0-0.5j, 15.0+0j],
], dtype=np.complex128))
_MatSPD_8_diag_b1 = np.array([1+1j, -2+0.5j, 3+0j, 0-1j, 2+3j, -1+1j, 4+0.5j, 1-2j], dtype=np.complex128)
_MatSPD_8_diag_b2 = np.array([0+1j, 1+1j, -1+0.5j, 2-2j, 0+0.3j, 3+1j, -2+3j, 1+0.5j], dtype=np.complex128)


_MAT_DAVIS_CHOL = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
])

_Mat_DAVIS_CHOL = csc_matrix(_MAT_DAVIS_CHOL + _MAT_DAVIS_CHOL.T).astype(np.complex128)
_b0 = np.zeros([0,0,0,0,0,0,0,0,0,0,0], dtype=np.complex128)
############################################################
#                           SETS                           #
############################################################

MAT_DAVIS_CHOL = SparseMatrix(_Mat_DAVIS_CHOL, [_b0,], 'SPD')

SET_CU: list[SparseMatrix] = [
    SparseMatrix(_MatCU_7_1, [_MatCU_7_1_b1], name='CU_8x8_basic', mtype='Unsymmetric'),
    SparseMatrix(_MatCU_8_pivot, [_MatCU_8_pivot_b1], name='CU_8x8_pivot', mtype='Unsymmetric'),
    SparseMatrix(_MatCU_5_nasty, [_MatCU_5_nasty_b1], name='CU_5x5_nasty', mtype='Unsymmetric'),
    SparseMatrix(_MatCU_20_rand, [_MatCU_20_rand_b1, _MatCU_20_rand_b2], name='CU_20x20_random', mtype='Unsymmetric'),
]

SET_CS: list[SparseMatrix] = [
    SparseMatrix(_MatCS_10_1, [_MatCS_10_1_b1, _MatCS_10_1_b2], name='CS_10x10_basic', mtype='Symmetric'),
    SparseMatrix(_MatCS_6_block, [_MatCS_6_block_b1], name='CS_6x6_block', mtype='Symmetric'),
    SparseMatrix(_MatCS_12_btridiag, [_MatCS_12_btridiag_b1, _MatCS_12_btridiag_b2], name='CS_12x12_btridiag', mtype='Symmetric'),
]

SET_SPD: list[SparseMatrix] = [
    SparseMatrix(_MatSPD_6, [_MatSPD_6_b1], name='SPD_6x6_random', mtype='SPD'),
    SparseMatrix(_MatSPD_10, [_MatSPD_10_b1, _MatSPD_10_b2], name='SPD_10x10_banded', mtype='SPD'),
    SparseMatrix(_MatSPD_8_diag, [_MatSPD_8_diag_b1, _MatSPD_8_diag_b2], name='SPD_8x8_diag_dom', mtype='SPD'),
]

SET_ALL: list[SparseMatrix] = SET_CU + SET_CS + SET_SPD