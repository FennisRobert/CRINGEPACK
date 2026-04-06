use std::{collections::HashMap, ops::Index, path::Component};

use log::{debug, error, info, trace, warn};
use num_complex::{Complex64};
use crate::sparse::{CCMatrixBase, CCMatrixOwned};
use numpy::ndarray::{ArrayView1, Array1, s};
use crate::symbolic::CCSymbolic;
use crate::sparse::MatrixType;
const CZERO: Complex64 = Complex64::new(0.0, 0.0);
const CONE: Complex64 = Complex64::new(1.0, 0.0);
use faer::{Par, Accum, Mat, Scale, mat, Col,ColRef, linalg};
use std::collections::{HashSet};
use crate::pmat::PMatrix;

struct ColumnBuffer {
    data: Vec<Complex64>,
    rowi: Vec<usize>,
    counter: usize,
    buffer: Vec<Complex64>,
}

impl ColumnBuffer {

    fn new(size: usize) -> Self{
        let data = vec![CZERO; size];
        let rowi = vec![0usize; size];
        let counter = 0usize;
        let buffer = vec![CZERO; size];
        ColumnBuffer { data, rowi, counter, buffer}
    }

    fn add(&mut self, val: Complex64, id: usize) {
        self.data[id] += val;
        self.rowi[self.counter] = id;
        self.counter += 1;
    }

    fn to_column(&mut self, scaler: Complex64, skip_id: usize) -> (Mat<Complex64>, Vec<usize>) {
        let mut ids: Vec<usize> = (0..self.counter).collect();
        ids.sort_by_key(|&i| self.rowi[i]);
        
        let mut ptr = 0usize;
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
        self.counter = 0;

        let col = Mat::from_fn(out_ids.len(), 1, |i, _| self.buffer[i]);
        (col, out_ids)
    }
    // fn to_column(&mut self, scaler: Complex64, skip_id: usize) -> (Mat<Complex64>, Vec<usize>) {
    //     let mut ids: Vec<usize> = (0..self.counter).collect();
    //     ids.sort_by_key(|&i| self.rowi[i]);
    //     let mut ptr = 0usize;
    //     let mut current = self.rowi[ids[0]];

    //     for i in 0..self.counter{
    //         if self.rowi[i] == skip_id{
    //             continue
    //         }
    //         if current != self.rowi[i] {
    //             ptr += 1;
    //             current = self.rowi[i];
    //             self.rowi[ptr] = current;
    //         }
    //         self.buffer[ptr] = self.data[current] * scaler;
    //         self.data[current] = CZERO;
    //     }
    //     self.counter = 0usize;
    //     let col = Mat::from_fn(ptr+1, 1,|i,j| self.buffer[i]);
    //     let ids = self.rowi[..ptr+1].to_vec();
    //     (col, ids)
    // }


}
struct FrontalMatrix {
    matrix: Mat<Complex64>,
    colids: Vec<usize>,
    icolids: HashMap<usize, usize>
}

impl FrontalMatrix {
    fn new(matrix: Mat<Complex64>, colids: Vec<usize>) -> Self {
        let map: HashMap<usize, usize> = colids.iter().enumerate().map(|(i, &v)| (v, i)).collect();
        FrontalMatrix {
            matrix: matrix,
            colids: colids,
            icolids: map,
        }
    }

    fn from_vec(vec: Mat<Complex64>, colids: Vec<usize>) -> Self {
        let mat = &vec * &vec.transpose();
        FrontalMatrix::new(mat, colids)
    }

    fn printself(&self) {
        println!("Mat [{:?}]", self.colids);
        println!("{:?}", self.matrix);
    }

    fn idf(&self) -> String {
        format!("Matrix[{:?}]", self.colids).to_string()
    }
    fn is_superset(&self, other: &FrontalMatrix) -> bool {
        for colid in other.colids.iter() {
            if !self.icolids.contains_key(colid) {
                return false
            }
        }
        true
    }

    fn get_diag(&self, index: &usize) -> Option<Complex64> {
        match self.icolids.get(index) {
            Some(i) => {
                Some(self.matrix[(*i, *i)])
            },
            None => None,
        }
    }
    
    fn absorb(&mut self, other: &FrontalMatrix, exclude_id: usize)  {
        for (i, icol) in other.colids.iter().enumerate() {
            if *icol == exclude_id {
                continue
            }
            if !self.icolids.contains_key(icol) {
                panic!("absorb: missing colid {} in self.colids={:?}, other.colids={:?}, exclude={}",
                    icol, self.colids, other.colids, exclude_id);
            }

            let &is = self.icolids.get(icol).unwrap();
            for (j, jcol) in other.colids.iter().enumerate() {
                if *jcol == exclude_id {
                    continue
                }
                let &js = self.icolids.get(jcol).unwrap();
                self.matrix[(is,js)] -= other.matrix[(i,j)];
            }
        }
    }

    fn iter_off_diag_column(&self, i_col: usize) -> Vec<(usize, Complex64)>{
        
        match self.icolids.get(&i_col) {
            Some(&icol) => (0..self.colids.len()).map(|i| (self.colids[i], self.matrix[(i,icol)])).collect(),
            None => (0..self.colids.len()).map(|i| (self.colids[i], CZERO)).collect(),
        }
        
    }
}

struct FrontalMatrixStack {
    mats: Vec<FrontalMatrix>,
    parents: Vec<usize>,
    keep: Vec<bool>
}

impl FrontalMatrixStack {
    fn new() -> Self {
        FrontalMatrixStack { mats: Vec::<FrontalMatrix>::with_capacity(100), 
            parents: Vec::<usize>::with_capacity(100), 
            keep: Vec::<bool>::with_capacity(100)}
    }

    fn add(&mut self, fm: FrontalMatrix, parent: usize) {
        self.mats.push(fm);
        self.parents.push(parent);
        self.keep.push(true);
    }

    fn print_overview(&self) {
        println!("Frontal Matrices:");
        for i in 0..self.mats.len(){
            println!(" - {}", self.mats[i].idf());
        }
    }
    fn clean(&mut self) {
        let mut i = 0;
        while i < self.keep.len() {
            if !self.keep[i] {
                self.mats.swap_remove(i);
                self.parents.swap_remove(i);
                self.keep.swap_remove(i);
            } else {
                i += 1;
            }
        }

    }

    fn iter_parent(&self, index: usize) -> impl Iterator<Item = (usize, &FrontalMatrix)> {
        (0..self.parents.len()).filter(move |&i| self.parents[i] == index).map(move |i| (i, &self.mats[i]))
    }


}

fn gen_vec(arr: ArrayView1<Complex64>, i1: usize, i2: usize) -> Mat<Complex64> {
    Mat::from_fn(i2-i1, 1usize, |i, j| arr[i1+i])
}


pub fn cholesky_decomp_mf(mat_c: &CCMatrixOwned, symb: &mut CCSymbolic) -> Option<CCMatrixOwned> {

    let pmat = PMatrix::from_cc(mat_c).printnz("NZ");
    let nnz_l = symb.nnz_l;
    let mut maxnic = 0usize;
    for i in 0..mat_c.n {
        maxnic = maxnic.max(symb.indptr[i+1]-symb.indptr[i]);
    }

    // Create Empty Sparse L matrix 
    let mut mat_l: CCMatrixOwned = CCMatrixOwned::zeros(nnz_l, mat_c.n, maxnic, MatrixType::LowerTriangular);

    // Copy Index pointers to mat_l
    for i in 0..(mat_c.n + 1) {
        mat_l.indptr[i] = symb.indptr[i];
    }

    // Create a stack of frontal matrices
    let mut fmstack = FrontalMatrixStack::new();


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
                continue
            }
            colbuf.add(mat_c.data[p], r);
        }

        // To remove contains frontal matrix indices that will be absorbed into the new FM
        let mut to_remove: Vec<usize> = Vec::with_capacity(100);

        // For each frontal matrix that is a parent of the current column
        for (ip, fm) in fmstack.iter_parent(i_col){
            // subtract the frontal matrix entry from L11
            match fm.get_diag(&i_col) {
                Some(value) => l11 = l11 - value,
                None => {}
            }
            // For each other entry in the frontal matrix
            //println!("{}, Mergin frontal matrix with nodes: {:?}", i_col, fm.colids);
            for (irow,d) in fm.iter_off_diag_column(i_col) {
                // Skip the diagonal L11
                //println!("---{},{}",irow, d);
                if irow == i_col {
                    continue
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

        
        // if its the second last column:
        if i_col == mat_l.n-1 {
            break
        }

        if colbuf.counter > 0 {
            // Divide L12 by L11
            let (l12, l12_ids) = colbuf.to_column(CONE/l11, i_col);
            
            let mut ptr = 0usize;
            for (i, &row) in l12_ids.iter().enumerate() {
                mat_l.data[mat_l.indptr[i_col] + ptr + 1] += l12[(i, 0usize)];
                mat_l.rows[mat_l.indptr[i_col] + ptr + 1] = row;
                ptr += 1
            }

            let mut fmat = FrontalMatrix::from_vec(l12, l12_ids);

            //println!("Icol: {}, parent: {} FM IDS: {:?}", i_col, symb.etree.etree[i_col], fmat.colids, );
            for &i in to_remove.iter(){
                //println!("   FM({}) - {:?}", fmstack.parents[i], fmstack.mats[i].colids);
                fmat.absorb(&fmstack.mats[i], i_col);
                fmstack.keep[i] = false;
            }

            fmstack.clean();
        
            fmstack.add(fmat, symb.etree.etree[i_col]);

        }

        
    }

    mat_l.printsmall();
    println!("True Lower Triangular Matrix");
    match cholesky_decomp_ul(mat_c, symb) {
        Ok(mat) => {
            for i in 0..mat_l.nnz {
                println!("UL = {:?}[{}] == [{}]{:?} MF", mat_l.data[i], mat_l.rows[i], mat.rows[i], mat.data[i]);
            }
            for i in 0..mat_l.n+1 {
                println!(" UL = {} == {} = MF", mat_l.indptr[i], mat.indptr[i]);
            }
            mat.printsmall()},
        Err(err) => {println!("Aborted with: {}", err);},
    }

    Some(mat_l)
}




pub fn cholesky_decomp_ul(mat_c: &CCMatrixOwned, symb: &mut CCSymbolic) -> Result<CCMatrixOwned, String> {
    
    let nnz_l = symb.nnz_l;
    let mut maxnic = 0usize;
    for i in 0..mat_c.n {
        maxnic = maxnic.max(symb.indptr[i+1]-symb.indptr[i]);
    }

    let mut mat_l: CCMatrixOwned = CCMatrixOwned::zeros(nnz_l, mat_c.n, maxnic, MatrixType::LowerTriangular);
    let mut col_buffer = vec![CZERO; mat_c.n];
    let mut col_pointer = vec![0usize; mat_c.n + 1usize];

    for i in 0..(mat_c.n+1) {
        mat_l.indptr[i] = symb.indptr[i];
        col_pointer[i] = symb.indptr[i];
    }

    for i_row in 0..mat_c.n {
        // For each Row in the permuted matric C.
        symb.etree.reach(&mat_c, i_row);
        
        let pa1 = mat_c.indptr[i_row];
        let pa2 = mat_c.indptr[i_row+1];

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

            d -= lki*lki;

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