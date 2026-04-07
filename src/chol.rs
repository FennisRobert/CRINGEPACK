use std::{collections::HashMap, ops::Index, path::Component};

use crate::sparse::MatrixType;
use crate::sparse::{CCMatrixBase, CCMatrixOwned};
use crate::symbolic::CCSymbolic;
use log::{debug, error, info, trace, warn};
use num_complex::Complex64;
use numpy::ndarray::{Array1, ArrayView1, s};
const CZERO: Complex64 = Complex64::new(0.0, 0.0);
const CONE: Complex64 = Complex64::new(1.0, 0.0);
//use faer::{Accum, Col, ColRef, Mat, Par, Scale, linalg, mat};
//use faer::linalg::matmul::matmul;
//use std::collections::HashSet;

struct ColumnBuffer {
    data: Vec<Complex64>,
    rowi: Vec<usize>,
    counter: usize,
    buffer: Vec<Complex64>,
}

impl ColumnBuffer {
    fn new(size: usize) -> Self {
        let data = vec![CZERO; size];
        let rowi = vec![0usize; size];
        let counter = 0usize;
        let buffer = vec![CZERO; size];
        ColumnBuffer {
            data,
            rowi,
            counter,
            buffer,
        }
    }

    fn add(&mut self, val: Complex64, id: usize) {
        self.data[id] += val;
        self.rowi[self.counter] = id;
        self.counter += 1;
    }

    fn to_column(&mut self, scaler: Complex64, skip_id: usize) -> (Vec<Complex64>, Vec<usize>) {
        let mut ids: Vec<usize> = (0..self.counter).collect();
        ids.sort_by_key(|&i| self.rowi[i]);

        let mut current = usize::MAX;
        let mut out_ids: Vec<usize> = Vec::new();

        for &idx in ids.iter() {
            let row = self.rowi[idx];
            if row == skip_id {
                continue;
            }
            if row != current {
                current = row;
                out_ids.push(current);
                self.buffer[out_ids.len() - 1] = self.data[current] * scaler;
            }
        }

        // Clear data
        for &idx in ids.iter() {
            self.data[self.rowi[idx]] = CZERO;
        }

        self.counter = 0usize;
        let col: Vec<Complex64> = (0..out_ids.len()).map(|i| self.buffer[i]).collect();
        (col, out_ids)
    }
}

struct FrontalMatrix {
    column: Vec<Complex64>,
    matrix: Vec<Vec<Complex64>>,
    colids: Vec<usize>,
    n: usize,
    keep: bool,
}

impl FrontalMatrix {

    fn empty(size: usize) -> Self {
        let column = vec![CZERO; size];
        let colids = vec![0usize; size];
        let mat: Vec<Vec<Complex64>> = Vec::new();
        FrontalMatrix { column: column, matrix: mat, colids: colids, n: 0usize, keep: true}
    }

    fn new(column: Vec<Complex64>, colids: Vec<usize>) -> Self {
        let n = column.len();
        let mat: Vec<Vec<Complex64>> = Vec::new();
        FrontalMatrix {
            column: column,
            matrix: mat,
            colids: colids,
            n: n,
            keep: true,
        }
    }

    fn gen_matrix(&mut self) {
        if self.matrix.len() != 0 {
            return
        }
        let n = self.column.len();
        let mut mat = vec![vec![CZERO; n]; n];
        for i in 0..n {
            for j in 0..n {
                mat[i][j] = self.column[i]*self.column[j]
            }
        }
        self.matrix = mat;
    }
    fn local_index(&self, global: &usize) -> Option<usize> {
        self.colids.binary_search(global).ok()
    }

    fn get_diag(&self, index: &usize) -> Option<Complex64> {
        match self.local_index(index) {
            Some(i) => Some(self.matrix[i][i]),
            None => None,
        }
    }

    fn absorb(&mut self, other: &FrontalMatrix) {
        if self.matrix.len()==0 {
            self.gen_matrix();
        }
        for (i, irow) in other.colids.iter().enumerate().skip(1) {
            let i_row_fm = self.local_index(irow).unwrap();
            for (j, jcol) in other.colids.iter().enumerate().skip(1) {
                let j_col_fm = self.local_index(jcol).unwrap();
                self.matrix[i_row_fm][j_col_fm] += other.matrix[i][j];
            }
        }
    }
    /// Iterates over column i_col and returns the (index, value) tuple
    fn iter_column(&self, i_col: usize) -> Vec<(usize, Complex64)> {
        match self.local_index(&i_col) {
            Some(icol) => (0..self.n)
                .map(|i| (self.colids[i], self.matrix[i][icol]))
                .collect(),
            None => {
                panic!(
                    "Column {} is not contained in FM with {:?}",
                    i_col, self.colids
                );
            }
        }
    }
}

struct FrontalMatrixStack {
    mats: Vec<FrontalMatrix>,
    parents: Vec<usize>,
}

impl FrontalMatrixStack {
    fn new(n_leafs: usize) -> Self {

        FrontalMatrixStack {
            mats: Vec::<FrontalMatrix>::with_capacity(n_leafs),
            parents: Vec::with_capacity(n_leafs),
        }
    }
    
    fn absorb(&mut self, i_base: usize, i_into: usize) {
        assert!(i_base != i_into);
        let ptr = self.mats.as_mut_ptr();
        unsafe {
            let child = &*ptr.add(i_base);
            let parent = &mut *ptr.add(i_into);
            parent.absorb(child);
        };
        self.mats[i_base].keep = false;
    }

    fn add_from_vec(&mut self, column: Vec<Complex64>, colids: Vec<usize>, parent: usize) -> usize {
        let idx = self.mats.len();
        self.mats.push(FrontalMatrix::new(column, colids));
        self.parents.push(parent);
        idx
    }

    fn clean(&mut self) {
        let mut ptr = 0usize;
        while ptr < self.mats.len() {
            if !self.mats[ptr].keep {
                self.mats.swap_remove(ptr);
                self.parents.swap_remove(ptr);
            } else {
                ptr += 1;
            }
        }
    }

    /// Returns the frontal matrices who's parent is index = index.
    fn iter_parent(&mut self, index: usize) -> Vec<usize> {
        (0..self.parents.len()).filter(move |&i| self.parents[i] == index).collect()
    }
}

pub fn cholesky_decomp_mf(mat_c: &CCMatrixOwned, symb: &mut CCSymbolic) -> Option<CCMatrixOwned> {
    //let pmat = PMatrix::from_cc(mat_c).printnz("NZ");
    
    let max_cnz = symb.max_items();
    println!("Max Frontal Matrix Size = {}", max_cnz);
    let nnz_l = symb.nnz_l;
    let mut maxnic = 0usize;
    for i in 0..mat_c.n {
        maxnic = maxnic.max(symb.indptr[i + 1] - symb.indptr[i]);
    }

    // Create Empty Sparse L matrix
    let mut mat_l: CCMatrixOwned =
        CCMatrixOwned::zeros(nnz_l, mat_c.n, maxnic, MatrixType::LowerTriangular);

    // Copy Index pointers to mat_l
    for i in 0..(mat_c.n + 1) {
        mat_l.indptr[i] = symb.indptr[i];
    }

    // Create a stack of frontal matrices
    let n_leafs = symb.etree.n_leafs(&symb.post);
    let mut fmstack = FrontalMatrixStack::new(n_leafs);

    let mut colbuf = ColumnBuffer::new(mat_c.n);
    // To remove contains frontal matrix indices that will be absorbed into the new FM
    let mut to_remove: Vec<usize> = Vec::with_capacity(100);
    // Notation:
    // A11 = top/left item
    // A12 = Vertical column
    // A22 = leftover block matrix
    for i_leaf in 0..mat_c.n {
        let i_col = symb.post[i_leaf];

        // Extract all rows for A12
        let p1 = mat_c.indptr[i_col];
        let p2 = mat_c.indptr[i_col + 1];

        // Start building l11 = A11
        let mut l11 = mat_c.udiag(i_col);

        // Write L12 except L11 onto the column buffer
        for p in p1..p2 {
            let r = mat_c.rows[p];
            // Skip L11
            if r <= i_col {
                continue;
            }
            colbuf.add(mat_c.data[p], r);
        }

        to_remove.clear();

        // For each frontal matrix of which the current column is the parent.
        for &ip in fmstack.iter_parent(i_col).iter() {

            // subtract the frontal matrix entry from L11
            fmstack.mats[ip].gen_matrix();
            match fmstack.mats[ip].get_diag(&i_col) {
                Some(value) => l11 = l11 - value,
                None => {}
            }
            // For each other entry in the frontal matrix
            //println!("{}, Mergin frontal matrix with nodes: {:?}", i_col, fm.colids);
            for (irow, d) in fmstack.mats[ip].iter_column(i_col) {
                // Skip the diagonal L11
                if irow <= i_col {
                    continue;
                }
                // Subtract the frontal matrix entry in row[irow] from L12
                colbuf.add(-d, irow);
            }
            to_remove.push(ip)
        }

        // Compute L11 and store
        l11 = l11.sqrt();
        mat_l.data[mat_l.indptr[i_col]] = l11;
        mat_l.rows[mat_l.indptr[i_col]] = i_col;

        // If there are entries in the column (almost always)
        if colbuf.counter > 0 {
            // Divide L12 by L11
            let (l12, l12_ids) = colbuf.to_column(CONE / l11, i_col);

            let mut ptr = 0usize;
            for (i, &row) in l12_ids.iter().enumerate() {
                mat_l.data[mat_l.indptr[i_col] + ptr + 1] += l12[i];
                mat_l.rows[mat_l.indptr[i_col] + ptr + 1] = row;
                ptr += 1
            }

            //let mut fmat = FrontalMatrix::from_vec(l12, l12_ids);
            let i_fmat = fmstack.add_from_vec(l12, l12_ids, symb.etree.etree[i_col]);
            for &i in to_remove.iter() {
                fmstack.absorb(i, i_fmat);
            }

            fmstack.clean();
            //fmstack.add(fmat, symb.etree.etree[i_col]);
        }
    }

    // mat_l.printsmall();
    // println!("True Lower Triangular Matrix");
    // match cholesky_decomp_ul(mat_c, symb) {
    //     Ok(mat) => {
    //         for i in 0..mat_l.nnz {
    //             println!("UL = {:?}[{}] == [{}]{:?} MF", mat_l.data[i], mat_l.rows[i], mat.rows[i], mat.data[i]);
    //         }
    //         for i in 0..mat_l.n+1 {
    //             println!(" UL = {} == {} = MF", mat_l.indptr[i], mat.indptr[i]);
    //         }
    //         mat.printsmall()},
    //     Err(err) => {println!("Aborted with: {}", err);},
    // }

    Some(mat_l)
}

pub fn cholesky_decomp_ul(
    mat_c: &CCMatrixOwned,
    symb: &mut CCSymbolic,
) -> Result<CCMatrixOwned, String> {
    let nnz_l = symb.nnz_l;
    let mut maxnic = 0usize;
    for i in 0..mat_c.n {
        maxnic = maxnic.max(symb.indptr[i + 1] - symb.indptr[i]);
    }

    let mut mat_l: CCMatrixOwned =
        CCMatrixOwned::zeros(nnz_l, mat_c.n, maxnic, MatrixType::LowerTriangular);
    let mut col_buffer = vec![CZERO; mat_c.n];
    let mut col_pointer = vec![0usize; mat_c.n + 1usize];

    for i in 0..(mat_c.n + 1) {
        mat_l.indptr[i] = symb.indptr[i];
        col_pointer[i] = symb.indptr[i];
    }

    for i_row in 0..mat_c.n {
        // For each Row in the permuted matric C.
        symb.etree.reach(&mat_c, i_row);

        let pa1 = mat_c.indptr[i_row];
        let pa2 = mat_c.indptr[i_row + 1];

        // Put the data of col/row i_row into the rowbuffer
        for p in pa1..pa2 {
            if mat_c.rows[p] <= i_row {
                col_buffer[mat_c.rows[p]] = mat_c.data[p];
            }
        }

        // Isolate the diagonal
        let mut d = col_buffer[i_row];
        col_buffer[i_row] = CZERO;

        // Pattern is a slice to the etree non-zero array.
        let pattern = symb.etree.non_zeros();

        for &row_l in pattern {
            // C[k,i] = row[i] / L[k,k]
            let lki = col_buffer[row_l] / mat_l.data[mat_l.indptr[row_l]];
            col_buffer[row_l] = CZERO;

            // pi_start = row/data index of the start of that column
            // pi_end = column_pointer[k] which is the same at the start of the loop?
            let pi_start = mat_l.indptr[row_l] + 1;
            let pi_end = col_pointer[row_l];

            //
            for p in pi_start..pi_end {
                col_buffer[mat_l.rows[p]] -= mat_l.data[p] * lki;
            }

            d -= lki * lki;

            let p = col_pointer[row_l];
            col_pointer[row_l] += 1;
            mat_l.rows[p] = i_row;
            mat_l.data[p] = lki;
        }

        if d.re <= 0.0 && d.im == 0.0 {
            return Err(format!("Not positive definite at column {}", i_row));
        }

        let p = col_pointer[i_row];
        col_pointer[i_row] += 1;
        mat_l.rows[p] = i_row;
        mat_l.data[p] = d.sqrt();
    }

    Ok(mat_l)
}
