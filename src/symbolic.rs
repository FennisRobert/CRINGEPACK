use crate::acc::vec_inv;
use crate::etree::{EliminationTree, col_non_zeros, etree_post};
use crate::perm::Permutation;
use crate::sparse::CCMatrixOwned;
use num_complex::{Complex64, ComplexFloat};
use numpy::ndarray::{Array1, ArrayView1, s};
const CZERO: Complex64 = Complex64::new(0.0, 0.0);
const CONE: Complex64 = Complex64::new(1.0, 0.0);

pub struct CCSymbolic {
    pub etree: EliminationTree,
    pub indptr: Vec<usize>,
    pub leftmost: Vec<usize>,
    pub nnz_l: usize,
    pub nnz_u: usize,
    pub post: Vec<usize>,
}

impl CCSymbolic {
    pub fn new(matrix: &CCMatrixOwned) -> Self {
        let (etree, etree_post, post) = etree_post(&matrix);
        let col_counts = col_non_zeros(&matrix, &etree, &post);
        let n = matrix.n;
        let mut leftmost = vec![usize::MAX; n];
        for col in 0..n {
            let p1 = matrix.indptr[col];
            let p2 = matrix.indptr[col + 1];
            for p in p1..p2 {
                let row = matrix.rows[p];
                leftmost[row] = leftmost[row].min(col);
            }
        }
        let mut colptr = vec![0usize; n + 1usize];
        let mut lnz = 0usize;
        for j in 0..n {
            colptr[j] = lnz;
            lnz += col_counts[j];
        }
        colptr[n] = lnz;

        CCSymbolic {
            etree: EliminationTree::new(etree),
            indptr: colptr,
            leftmost: leftmost,
            nnz_l: lnz,
            nnz_u: 0usize,
            post: post,
        }
    }

    pub fn max_items(&self) -> usize {
        let mut max = 0usize;
        for i in 1..self.indptr.len() {
            max = (self.indptr[i] - self.indptr[i-1]).max(max);
        }
        max
    }
}
