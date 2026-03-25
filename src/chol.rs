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
use crate::pmat::PMatrix;

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
                println!("Diag ({},{})", i,i);
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

    // fn split(&self, colid: usize) -> Option<(Complex64, Col<Complex64>, Mat<Complex64>)> {
    //     match self.icolids.get(&colid) {
    //         Some(&icol) => {
    //             (self.matrix[(icol, icol)], self.matrix.col(icol)[1..], self.matrix[(1.., 1..)])
    //         }
    //         None => None
    //     }
    // }
    fn iter_off_diag_column(&self, i_col: usize) -> Vec<(usize, Complex64)>{
        match self.icolids.get(&i_col) {
            Some(&icol) => (1..self.colids.len()).map(|i| (self.colids[i], self.matrix[(i,icol)])).collect(),
            None => Vec::new(),
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


    let nnz_l = symb.nnz_l;
    let mut maxnic = 0usize;
    for i in 0..mat_c.n {
        maxnic = maxnic.max(symb.indptr[i+1]-symb.indptr[i]);
    }

    let mut mat_l: CCMatrixOwned = CCMatrixOwned::zeros(nnz_l, mat_c.n, maxnic, MatrixType::LowerTriangular);

    for i in 0..(mat_c.n + 1) {
        mat_l.indptr[i] = symb.indptr[i];
    }

    let mut fmstack = FrontalMatrixStack::new();

    let mut col_buffer = Col::<Complex64>::zeros(mat_c.n);
    let mut buffer_ids: Vec<usize> = vec![usize::MAX; mat_c.n*5];
    let mut buffer_ctr = 0usize;

    PMatrix::from_cc(mat_c).printnz("MATRIX");
    println!("Rows: {:?}", mat_c.rows);
    println!("Inds: {:?}", mat_c.indptr);
    // Signel column multifrontal
    println!("Postorder = {:?}", symb.post);
    for i_leaf in 0..mat_c.n-1 {
        println!("");
        let i_col = symb.post[i_leaf];
        println!("Solving Entry/Column: {} -> {} [parent = {}]", i_leaf, i_col, symb.etree.etree[i_col]);

        let p1 = mat_c.indptr[i_col];
        let p2 = mat_c.indptr[i_col + 1];

        let mut l11 = mat_c.data[p1];

        mat_l.rows[mat_l.indptr[i_col]] = i_col;
        
        // Formation of the dense vector 
        
        // Write L12 onto the column buffer
        for p in (p1)..p2 {
            
            let r = mat_c.rows[p];
            if r <= i_col {
                continue
            }
            println!("Assigning {:?} to {:?}", mat_c.data[p], r);
            col_buffer[r] = mat_c.data[p];
            buffer_ids[buffer_ctr] = r;
            buffer_ctr += 1
        }

        let mut to_remove: Vec<usize> = Vec::with_capacity(100);

        for (ip, fm) in fmstack.iter_parent(i_col){
            match fm.get_diag(&i_col) {
                Some(value) => l11 = l11 - value,
                None => {}
            }

            // Add the column of the frontal matrix to the column buffer
            for (irow,d) in fm.iter_off_diag_column(i_col) {
                println!("Assigning {:?} to {:?}", d, irow);
                col_buffer[irow] -= d;
                buffer_ids[buffer_ctr] = irow;
                buffer_ctr += 1;
            }
            to_remove.push(ip)
        }

        

        l11 = l11.sqrt();
        println!("L11 = {}", l11);
        
        // Store L11 = A11.sqrt() onto the diagonal
        mat_l.data[mat_l.indptr[i_col]] = l11;
        
        col_buffer = Scale(CONE/ l11) *  col_buffer; // a21 / l11
        
        // Column buffer to new Frontal Matrix
        if i_col == mat_l.n-2 {
            println!("Last entry!");
            let mut last = mat_c.data[mat_c.data.len()-1];
            for fm in fmstack.mats {
                last -= fm.matrix[(1,1)]*fm.matrix[(1,1)];
                fm.printself();
            }
            mat_l.data[mat_l.indptr[i_col + 1]] = last.sqrt();
            mat_l.rows[mat_l.indptr[i_col + 1]] = mat_l.n-1;
            mat_l.printsmall();
            break
        }

        let mut ids: Vec<usize> = (0..buffer_ctr).collect();
        ids.sort_by_key(|&i| buffer_ids[i]);
        let mut cur = buffer_ids[ids[0]];
        let mut ptr = 0usize;
        let mut data = vec![CZERO; buffer_ctr];
        
        let mut indices = vec![0usize; buffer_ctr];
        
        data[0] = col_buffer[cur];
        indices[ptr] = cur;
        mat_l.data[mat_l.indptr[i_col] + 1usize + ptr] = col_buffer[cur];
        mat_l.rows[mat_l.indptr[i_col] + 1usize + ptr] = cur;
        println!("Buffer:");
        println!("COL Buffer: {:?}", col_buffer);
        println!("Buffer_IDS: {:?}", buffer_ids);
        println!("{}", buffer_ctr);

        println!("Expected Nonzeros: {}", mat_l.indptr[i_col+1]-mat_l.indptr[i_col]);
        for i in 1..buffer_ctr {
            let new_row = buffer_ids[ids[i]];
            if new_row != cur {
                ptr += 1;
                
                data[ptr] = col_buffer[new_row];
                indices[ptr] = new_row;

                
                col_buffer[cur] = CZERO;
                cur = new_row;
            }
            println!("{} -> {} -> {} -> {}", i, ids[i], ptr, cur);
            buffer_ids[ids[i]] = 0usize;

        }
        buffer_ctr = 0;
        println!("Ptr= {}", ptr);
        let col = Mat::from_fn(ptr+1, 1usize, |i,j| data[i]);

        let mut fmat = FrontalMatrix::from_vec(col, indices[..ptr+1].to_vec());

        for &i in to_remove.iter(){
            println!("Absorbing fm{} into the new matrix with ids: {:?}", i, indices[..ptr+1].to_vec());
            fmat.absorb(&fmstack.mats[i], i_col);
            fmstack.keep[i] = false;
        }
        fmat.printself();
        fmstack.clean();
        
        fmstack.add(fmat, symb.etree.etree[i_col]);
        
        mat_l.printsmall();

        
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