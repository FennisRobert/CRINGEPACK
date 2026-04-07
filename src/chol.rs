use std::{collections::HashMap, ops::Index, path::Component};

use crate::sparse::MatrixType;
use crate::sparse::{CCMatrixBase, CCMatrixOwned};
use crate::symbolic::CCSymbolic;
use log::{debug, error, info, trace, warn};
use num_complex::Complex64;
use numpy::ndarray::{Array1, ArrayView1, s};
const CZERO: Complex64 = Complex64::new(0.0, 0.0);
const CONE: Complex64 = Complex64::new(1.0, 0.0);
// use faer::complex::c64;
// use faer::linalg::matmul::matmul;
// use faer::{Par, MatMut, ColRef};

struct ColumnBuffer {
    data: Vec<Complex64>,
    rowi: Vec<usize>,
    counter: usize,
    buffer: Vec<Complex64>,
    outids: Vec<usize>,
    sortbuf: Vec<usize>,
}

impl ColumnBuffer {
    
    fn new(size: usize) -> Self {
        let data = vec![CZERO; size];
        let rowi = vec![0usize; size];
        let counter = 0usize;
        let buffer = vec![CZERO; size];
        let outids = vec![0usize; size];
        let sortbuf = vec![0usize; size];
        ColumnBuffer {
            data,
            rowi,
            counter,
            buffer,
            outids,
            sortbuf,
        }
    }

    fn add(&mut self, val: Complex64, id: usize) {
        self.data[id] += val;
        self.rowi[self.counter] = id;
        self.counter += 1;
    }

    fn to_column(&mut self, scaler: Complex64, skip_id: usize) -> (&[Complex64], &[usize]) {
        self.sortbuf[..self.counter].copy_from_slice(&self.rowi[..self.counter]);
        self.sortbuf[..self.counter].sort();

        let mut current = usize::MAX;
        let mut nout = 0usize;

        for i in 0..self.counter {
            let row = self.sortbuf[i];
            if row == skip_id {
                continue;
            }
            if row != current {
                current = row;
                self.outids[nout] = current;
                self.buffer[nout] = self.data[current] * scaler;
                self.data[current] = CZERO;
                nout += 1;
            }
        }

        self.counter = 0;
        (&self.buffer[..nout], &self.outids[..nout])
    }
}

struct FrontalMatrixStack {
    fm_data: Vec<Complex64>,
    fm_data_idstart: Vec<usize>,
    fm_ids: Vec<usize>,
    fm_ids_idstart: Vec<usize>,
    parents: Vec<usize>,
    dims: Vec<usize>,
    newfm: Vec<Complex64>,
    search_buffer: Vec<usize>,
    n_alive: usize
}

impl FrontalMatrixStack {
    fn new(max_alive: usize, ndimmax: usize, max_size: usize, ndim: usize) -> Self {

        FrontalMatrixStack {
            fm_data: vec![CZERO; max_size],
            fm_data_idstart: vec![0usize; max_alive+1],
            fm_ids: vec![0usize; max_size],
            fm_ids_idstart: vec![0usize; max_alive+1],
            parents: vec![usize::MAX; max_alive],
            dims: vec![0usize; max_alive],
            newfm: vec![CZERO; ndimmax*ndimmax],
            n_alive: 0usize,
            search_buffer: vec![0usize; ndim],
        }
    }

    fn remove_fm(&mut self, index: usize) {
        if index != self.n_alive -1 {
            panic!("Trying to remove a frontal matrix from the middle :(");
        }
        self.parents[index] = usize::MAX;
        self.dims[index] = 0usize;
        self.n_alive -= 1;
    }

    /// Returns the local frontal matrix index and data entry for the diagonal for global matrix column icol
    fn get_diag(&self, ifm: usize, g_icol: usize) -> Option<(usize, Complex64)> {
        for i in 0..self.dims[ifm] {
            if self.fm_ids[self.fm_ids_idstart[ifm] + i] == g_icol {
                return Some((i, self.getmat(ifm, i, i)))
            }
        }
        None
    }

    fn getmat(&self, ifm: usize, i: usize, j: usize) -> Complex64 {
        self.fm_data[self.fm_data_idstart[ifm] + i*self.dims[ifm] + j]
    }

    fn add_from_vec(&mut self, colindex: usize, column: &[Complex64], colids: &[usize], parent: usize) -> usize {
        let n = column.len();
        // Cache the new frontal matrix
        for i in 0..n {
            for j in i..n {
                self.newfm[i*n + j] = (column[i]*column[j]);
            }
        }
        
        for i in 0..n {
            self.search_buffer[colids[i]] = i;
        }
        for ifm in (0..self.n_alive).rev() {
            // If this frontal matrix has this column as its parent
            if self.parents[ifm] == colindex {
                let nd: usize = self.dims[ifm];
                
                for ir_loc in 1..nd {
                    let ir_glob = self.fm_ids[self.fm_ids_idstart[ifm] + ir_loc];
                    let ir = self.search_buffer[ir_glob];
                    for ic_loc in ir_loc..nd {
                        let ic_glob = self.fm_ids[self.fm_ids_idstart[ifm] + ic_loc];
                        let ic = self.search_buffer[ic_glob];
                        let to_add = self.fm_data[self.fm_data_idstart[ifm] + ir_loc*nd + ic_loc];
                        self.newfm[ir*n + ic] += to_add;
                    }
                    
                }
                self.remove_fm(ifm);
            }else {
                break
            }
        }

        // Write new frontal matrix
        let ifm = self.n_alive;
        let istart = self.fm_data_idstart[ifm];
        self.fm_data_idstart[ifm+1] = istart + n*n;
        self.fm_ids_idstart[ifm+1] = self.fm_ids_idstart[ifm] + n;
        self.parents[ifm] = parent;
        for i in 0..n {
            self.fm_ids[self.fm_ids_idstart[ifm] + i] = colids[i];
            for j in i..n {
                let val = self.newfm[i*n + j];
                self.fm_data[istart + i*n + j] = val;
                self.fm_data[istart + j*n + i] = val;
            }
        }
        self.dims[ifm] = n;
        self.n_alive += 1;
        //println!("maxptr: {}", self.fm_data_idstart[self.n_alive]);
        0usize
    }

}

pub fn cholesky_decomp_mf(mat_c: &CCMatrixOwned, symb: &mut CCSymbolic) -> Option<CCMatrixOwned> {
    //let pmat = PMatrix::from_cc(mat_c).printnz("NZ");
    
    let (max_fm_alive, max_fm_dim, max_totsize) = symb.max_size();
    println!("Max Alive: {}, Max DIM: {}, max size: {}, nnz: {}", max_fm_alive, max_fm_dim, max_totsize, symb.nnz_l);
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
    let mut fmstack = FrontalMatrixStack::new(max_fm_alive, max_fm_dim, max_totsize, mat_c.n);

   
    let mut colbuf = ColumnBuffer::new(mat_c.n);
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

        // For each frontal matrix of which the current column is the parent.
        for ip in (0..fmstack.n_alive).rev() {
            if fmstack.parents[ip] != i_col {
                break
            }
            // subtract the frontal matrix entry from L11
            let mut index = usize::MAX;
            match fmstack.get_diag(ip, i_col) {
                Some((locind, value)) => {
                    index = locind;
                    l11 = l11 - value;
                }
                None => {}
            }
            // For each other entry in the frontal matrix
            //println!("{}, Mergin frontal matrix with nodes: {:?}", i_col, fm.colids);
            if index==usize::MAX {
                continue
            }
            for i in 0..fmstack.dims[ip] {
                let irow = fmstack.fm_ids[fmstack.fm_ids_idstart[ip] + i];
                let d = fmstack.getmat(ip, i, index);
                if irow <= i_col {
                    continue;
                }
                // Subtract the frontal matrix entry in row[irow] from L12
                colbuf.add(-d, irow);
            }
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
            let i_fmat = fmstack.add_from_vec(i_col, l12, l12_ids, symb.etree.etree[i_col]);
        }
    }


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
