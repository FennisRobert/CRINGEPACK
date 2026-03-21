use log::{debug, error, info, trace, warn};
use num_complex::{Complex64, ComplexFloat};
use numpy::ndarray::{Array1, ArrayView1, s};

use crate::pmat::{PMatColumn, PMatrix};
use crate::sparse::{CCMatrixBase, CCMatrixOwned, CCMatrixView, MatrixType};

const CZERO: Complex64 = Complex64::new(0.0, 0.0);
const CONE: Complex64 = Complex64::new(1.0, 0.0);

// pub fn elimination_tree(matrix: &CCMatrixOwned) -> Vec<usize> {
//     let mut etree = vec![usize::MAX; matrix.n];
//     for icol in 0..(matrix.n-1) {
        
//         let i1 = matrix.indptr[icol];
//         let i2 = matrix.indptr[icol+1];
//         for idx in i1..i2 {
//             let mut child = icol;
//             let mut par = matrix.rows[idx];
//             let mut cur_par = etree[child];
//             if par <= icol {
//                 continue
//             }
//             for iter in 0..matrix.n {
//                 println!("Child [{}] -> {}, {}", child, cur_par, par);
//                 if cur_par == usize::MAX {
//                     etree[child] = par;
//                     break
//                 }else if cur_par < par {
//                     child = cur_par;
//                     cur_par = etree[child];
//                     if child == matrix.n {
//                         break
//                     }
//                 }else if cur_par > par {
//                     etree[child] = par;
//                     (cur_par, par) = (par, cur_par);
//                 }
                
//             }
            

//         }
//         println!("Etree [{}]: {:?}", icol, etree);
//     }
//     println!("Etree: {:?}", etree);
//     etree
// }

pub fn elimination_tree(matrix: &CCMatrixOwned) -> Vec<usize> {
    let n = matrix.n;
    let mut parent = vec![usize::MAX; n];
    let mut ancestor = vec![0usize; n];

    for j in 0..n {
        ancestor[j] = j;
        let i1 = matrix.indptr[j] as usize;
        let i2 = matrix.indptr[j + 1] as usize;
        for idx in i1..i2 {
            let i = matrix.rows[idx] as usize;
            if i >= j {
                continue; // only look at upper triangle
            }
            // Walk from i up to j, compressing the path
            let mut node = i;
            loop {
                let next = ancestor[node];
                if next == j {
                    break; // already linked to j
                }
                ancestor[node] = j; // path compression
                if next == node {
                    // node was a root, make j its parent
                    parent[node] = j;
                    break;
                }
                node = next;
            }
        }
    }
    println!("Etree: {:?}", parent);
    parent
}