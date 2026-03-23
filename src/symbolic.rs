use num_complex::{Complex64, ComplexFloat};
use numpy::ndarray::{ Array1, ArrayView1, s};
use crate::perm::Permutation;
use crate::sparse::{ CCMatrixOwned };
use crate::acc::{vec_inv};
use crate::etree::{etree_post, col_non_zeros};
const CZERO: Complex64 = Complex64::new(0.0, 0.0);
const CONE: Complex64 = Complex64::new(1.0, 0.0);


pub struct CCSymbolic {
    pub perm_fill: Permutation,
    pub etree: Vec<usize>,
    pub indptr: Vec<usize>,
    pub leftmost: Vec<usize>,
    pub nnz_l: usize,
    pub nnz_u: usize,
}

impl CCSymbolic {
    pub fn new(matrix: &CCMatrixOwned) -> Self {
        let perm_fill = Permutation::metis(&matrix);
        let c = matrix.permute_new(&perm_fill.perm);
        let (etree, post) = etree_post(&c);
        let col_counts = col_non_zeros(&c, &etree, &post);
        println!("Colcounts: {:?}", col_counts);
        let n = matrix.n;
        let mut leftmost = vec![usize::MAX; n];
        for col in 0..n {
            let p1 = c.indptr[col];
            let p2 = c.indptr[col+1];
            for p in p1..p2 {
                let row = c.rows[p];
                leftmost[row] = leftmost[row].min(col);
            }
        }
        let mut colptr = vec![0usize; n+1];
        let mut lnz = 0usize;
        for j in 0..n {
            colptr[j] = lnz;
            lnz += col_counts[j];
        }
        
        CCSymbolic { perm_fill: perm_fill, etree: etree, indptr: colptr, leftmost: leftmost, nnz_l: lnz, nnz_u: 0usize}
    }
}