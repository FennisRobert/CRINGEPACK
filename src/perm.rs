use crate::acc::vec_inv;
use crate::metisperm::nested_dissection;
use crate::sparse::CCMatrixOwned;

fn preorder(matrix: &CCMatrixOwned) -> Vec<usize> {
    let mut p: Vec<usize> = (0..matrix.n).collect();
    let mut score = vec![0usize; matrix.n];
    let mut counter = vec![0usize; matrix.n];
    for i in 0..matrix.n {
        let i1 = matrix.indptr[i] as usize;
        let i2 = matrix.indptr[i + 1] as usize;
        for j in i1..i2 {
            let r = matrix.rows[j] as usize;
            score[i] += r;
            counter[i] += 1;
        }
    }
    let mut rating = vec![0.0f64; matrix.n];
    for i in 0..matrix.n {
        rating[i] = score[i] as f64 / counter[i] as f64;
    }
    p.sort_by(|&a, &b| rating[a].total_cmp(&rating[b]));
    p
}

pub struct Permutation {
    pub perm: Vec<usize>,
    pub iperm: Vec<usize>,
}

impl Permutation {
    pub fn new(perm: Vec<usize>) -> Self {
        let iperm = vec_inv(&perm);
        Permutation { perm, iperm }
    }

    pub fn empty() -> Self {
        Permutation {
            perm: Vec::with_capacity(0usize),
            iperm: Vec::with_capacity(0usize),
        }
    }

    pub fn naive(matrix: &CCMatrixOwned) -> Self {
        Permutation::new(preorder(&matrix))
    }
    pub fn metis(matrix: &CCMatrixOwned) -> Self {
        Permutation::new(nested_dissection(&matrix))
    }
}
