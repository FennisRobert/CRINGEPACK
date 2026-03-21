use crate::sparse::CCMatrixOwned;
use metis_sys::{METIS_NodeND, idx_t};
use std::ptr;

pub fn nested_dissection(matrix: &CCMatrixOwned) -> Vec<usize> {
    let n = matrix.n;

    // Build symmetric adjacency, no diagonal
    let mut neighbors: Vec<Vec<idx_t>> = vec![Vec::new(); n];
    for col in 0..n {
        let p1 = matrix.indptr[col] as usize;
        let p2 = matrix.indptr[col + 1] as usize;
        for p in p1..p2 {
            let row = matrix.rows[p] as usize;
            if row != col {
                neighbors[col].push(row as idx_t);
                neighbors[row].push(col as idx_t);
            }
        }
    }
    for i in 0..n {
        neighbors[i].sort();
        neighbors[i].dedup();
    }

    let mut xadj: Vec<idx_t> = vec![0];
    let mut adjncy: Vec<idx_t> = Vec::new();
    for i in 0..n {
        adjncy.extend_from_slice(&neighbors[i]);
        xadj.push(adjncy.len() as idx_t);
    }

    let mut nvtxs = n as idx_t;
    let mut perm = vec![0 as idx_t; n];
    let mut iperm = vec![0 as idx_t; n];

    unsafe {
        METIS_NodeND(
            &mut nvtxs,
            xadj.as_mut_ptr(),
            adjncy.as_mut_ptr(),
            ptr::null_mut(), // no vertex weights
            ptr::null_mut(), // default options
            perm.as_mut_ptr(),
            iperm.as_mut_ptr(),
        );
    }

    perm.iter().map(|&x| x as usize).collect()
}
