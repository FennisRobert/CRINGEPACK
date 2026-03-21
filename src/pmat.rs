use crate::sparse::{CCMatrixBase, CCMatrixOwned, MatrixType};
use log::{debug, error, info, trace, warn};
use num_complex::{Complex, Complex64, ComplexFloat};
use numpy::array;
use numpy::ndarray::{Array1, ArrayView1, s};
use std::fmt;

const CZERO: Complex64 = Complex64::new(0.0, 0.0);
const CONE: Complex64 = Complex64::new(1.0, 0.0);
pub struct PMatColumn {
    data: Vec<Complex64>,
    rows: Vec<usize>,
    icap: usize,
    n: usize,
    sorted: bool,
}

pub struct PMatrix {
    pub n: usize,
    cols: Vec<PMatColumn>,
    acu_dense_col: Vec<Complex64>,
    acuids: Vec<usize>,
    acuctr: usize,
}

impl fmt::Display for PMatColumn {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for i in 0..self.n {
            write!(f, " - row {}: {:.3}\n", self.rows[i], self.data[i])?;
        }
        Ok(())
    }
}

impl PMatColumn {
    pub fn new(n: usize) -> Self {
        let data: Vec<Complex64> = Vec::with_capacity(n);
        let rows: Vec<usize> = Vec::with_capacity(n);

        PMatColumn {
            data,
            rows,
            icap: n,
            n: 0usize,
            sorted: true,
        }
    }

    fn sort(&mut self) {
        if self.n < 2 {
            return;
        }
        let mut ids: Vec<usize> = (0..self.n).collect();
        ids.sort_by_key(|&i| self.rows[i]);
        let new_data: Vec<Complex64> = ids.iter().map(|&i| self.data[i]).collect();
        let new_rows: Vec<usize> = ids.iter().map(|&i| self.rows[i]).collect();
        self.data = new_data;
        self.rows = new_rows;
        self.n = self.data.len();
        self.sorted = true;
    }

    pub fn add_row(&mut self, row: usize, value: Complex64) {
        self.rows.push(row);
        self.data.push(value);
        self.n = self.n + 1;
        if self.n >= 2 && self.sorted && (self.rows[self.n - 1] < self.rows[self.n - 2]) {
            self.sorted = false;
        }
    }

    pub fn get(&mut self, row: usize) -> Complex64 {
        for i in 0..self.n {
            if self.rows[i] == row {
                return self.data[i];
            }
        }
        CZERO
    }

    pub fn max(&self, startrow: usize) -> (usize, usize, Complex64) {
        let (idx, _) = self.data[..self.n]
            .iter()
            .enumerate()
            .filter(|(i, _)| self.rows[*i] >= startrow)
            .max_by(|(_, a), (_, b)| a.norm_sqr().total_cmp(&b.norm_sqr()))
            .unwrap();
        (idx, self.rows[idx], self.data[idx])
    }

    pub fn find_row(&mut self, rownr: usize) -> Option<usize> {
        self.sort();
        match self.rows.binary_search(&rownr) {
            Ok(index) => Some(index),
            Err(_) => None,
        }
    }
    pub fn find_row_scan(&mut self, rownr: usize) -> Option<usize> {
        for i in 0..self.n {
            if self.rows[i] == rownr {
                return Some(i);
            }
        }
        None
    }

    pub fn ppivot(&mut self, rowpair: (usize, usize)) {
        for i in 0..self.n {
            if self.rows[i] == rowpair.0 {
                self.rows[i] = rowpair.1
            } else if self.rows[i] == rowpair.1 {
                self.rows[i] = rowpair.0
            }
        }
        self.sorted = false;
    }
    pub fn printcol(&self) {
        println!("Col:");
        println!("rows: {:?}", self.rows);
        println!("rows: {:?}", self.data);
    }
}

impl PMatrix {
    pub fn new(dim: usize, nmic: usize) -> Self {
        let mut cols: Vec<PMatColumn> = Vec::with_capacity(dim);
        for i in 0..dim {
            cols.push(PMatColumn::new(nmic));
        }
        let colacu: Vec<Complex64> = vec![CZERO; dim];
        let colids: Vec<usize> = vec![0usize; dim];
        PMatrix {
            n: dim,
            cols: cols,
            acu_dense_col: colacu,
            acuids: colids,
            acuctr: 0usize,
        }
    }

    pub fn eye(dim: usize, nmic: usize) -> Self {
        debug!("New identity matrix with dim {}", dim);
        let mut cols: Vec<PMatColumn> = Vec::with_capacity(dim);
        for i in 0..dim {
            let mut col = PMatColumn::new(nmic);
            col.add_row(i, CONE);
            cols.push(col);
        }
        let colacu: Vec<Complex64> = vec![CZERO; dim];
        let colids: Vec<usize> = vec![0usize; dim];
        PMatrix {
            n: dim,
            cols: cols,
            acu_dense_col: colacu,
            acuids: colids,
            acuctr: 0usize,
        }
    }

    pub fn from_cc(matrix: &impl CCMatrixBase) -> Self {
        debug!("New PMatrix with dim = {}", matrix.get_n());
        let mut cols: Vec<PMatColumn> = Vec::with_capacity(matrix.get_n());
        let nmic = matrix.get_maxnic() * 10;
        for i in 0..matrix.get_n() {
            let mut newcol = PMatColumn::new(nmic);
            for (r, d) in matrix.iter_col(i) {
                newcol.add_row(r, d);
            }
            cols.push(newcol);
        }
        let colacu: Vec<Complex64> = vec![CZERO; matrix.get_n()];
        let colids: Vec<usize> = vec![0usize; matrix.get_n()];
        PMatrix {
            n: matrix.get_n(),
            cols: cols,
            acu_dense_col: colacu,
            acuids: colids,
            acuctr: 0usize,
        }
    }

    pub fn get(&mut self, row: usize, col: usize) -> Complex64 {
        self.cols[col].get(row)
    }

    pub fn getcol(&mut self, colid: usize) -> &PMatColumn {
        &self.cols[colid]
    }

    pub fn iter_col(&mut self, col: usize, start: usize) -> (Vec<usize>, Vec<Complex64>) {
        self.cols[col].sort();
        let col = &self.cols[col];
        let mut rows = Vec::with_capacity(col.data.capacity());
        let mut data = Vec::with_capacity(col.data.capacity());
        for i in 0..col.n {
            if col.rows[i] < start {
                continue;
            }
            rows.push(col.rows[i]);
            data.push(col.data[i]);
        }
        (rows, data)
    }

    pub fn set(&mut self, row: usize, col: usize, value: Complex64) {
        self.cols[col].add_row(row, value);
    }

    pub fn ppivot(&mut self, rowpair: (usize, usize), endid: usize) {
        for i in 0..endid {
            self.cols[i].ppivot(rowpair);
        }
    }

    pub fn read_column(&mut self, colid: usize) {
        let col = &self.cols[colid];
        let mut ctr: usize = 0;
        for i in 0..col.n {
            if col.data[i] == CZERO {
                continue;
            }
            self.acu_dense_col[col.rows[i]] = col.data[i];
            self.acuids[ctr] = col.rows[i];
            ctr += 1
        }
        self.acuctr = ctr;
    }

    pub fn clear_acu(&mut self, colid: usize) {
        let mut data: Vec<Complex64> = vec![CZERO; self.acuctr];
        let mut rows: Vec<usize> = vec![0usize; self.acuctr];
        let mut ctr: usize = 0usize;

        for i in 0..self.acuctr {
            let ir = self.acuids[i];
            if self.acu_dense_col[ir] != CZERO {
                data[ctr] = self.acu_dense_col[ir];
                rows[ctr] = ir;
                ctr += 1;
            }
            self.acu_dense_col[self.acuids[i]] = CZERO;
            self.acuids[i] = 0usize;
        }
        self.cols[colid].n = ctr;
        self.cols[colid].sorted = true;
        self.acuctr = 0usize;
        self.cols[colid].data = data;
        self.cols[colid].rows = rows;
    }

    pub fn eliminate_rows(
        &mut self,
        source_row: usize,
        target_rows: &Vec<usize>,
        values: &Vec<Complex64>,
    ) {
        self.read_column(source_row);
        // Elimination of the Pivot column
        for ie in 0..target_rows.len() {
            let irow = target_rows[ie];
            self.acu_dense_col[irow] = CZERO;
        }
        self.clear_acu(source_row);

        // Elimination of all subsequent columns
        for icol in (source_row + 1)..self.n {
            self.read_column(icol);
            let mat_elem = self.acu_dense_col[source_row];
            // Only do elimination if there is something to subtract
            if mat_elem != CZERO {
                for (&irow, &value) in target_rows.iter().zip(values.iter()) {
                    // Iff the column has a zero entry, add a non-zero
                    if self.acu_dense_col[irow] == CZERO {
                        self.acuids[self.acuctr] = irow;
                        self.acuctr += 1;
                    }
                    // Subtract the value
                    self.acu_dense_col[irow] -= value * mat_elem;
                }
            }
            self.clear_acu(icol);
        }
    }

    pub fn to_ccmatrix(&mut self, mtype: MatrixType) -> CCMatrixOwned {
        let mut nnz: usize = 0;
        for icol in 0..self.n {
            nnz = nnz + self.cols[icol].n;
        }
        let mut data = Array1::<Complex64>::zeros(nnz);
        let mut rows = Array1::<usize>::zeros(nnz);
        let mut indptr = Array1::<usize>::zeros(self.n + 1usize);

        let mut ptr: usize = 0;

        for icol in 0..self.n {
            self.read_column(icol);
            for indx in 0..self.acuctr {
                let r = self.acuids[indx];
                let d = self.acu_dense_col[r];
                if d != CZERO {
                    data[ptr] = d;
                    rows[ptr] = r;
                    ptr += 1;
                }
            }

            self.clear_acu(icol);
            indptr[icol + 1] = ptr;
        }
        let data = data.slice(s![0..ptr]).to_owned();
        let rows = rows.slice(s![0..ptr]).to_owned();
        CCMatrixOwned::new(rows, indptr, data, mtype)
    }
    pub fn printcols(&self) {
        println!("Matrix Columns:");
        for icol in 0..self.n {
            println!("Column: {}", icol);
            println!("{}", self.cols[icol]);
        }
    }
    pub fn printsmall(&mut self, prefix: &str) {
        println!("{}", prefix);
        for row in 0..self.n {
            let mut line = String::new();
            for col in 0..self.n {
                let val = self.get(row, col).norm();
                line.push_str(&format!("{:>8.8} ", val));
            }
            println!("{}", line);
        }
    }
    pub fn printnz(&mut self, prefix: &str) {
        println!("{}", prefix);
        for row in 0..self.n {
            let mut line = String::new();
            line.push('|');
            for col in 0..self.n {
                let mut l = ' ';
                let mut r = ' ';
                if col == row {
                    l = '[';
                    r = ']';
                }
                let val = self.get(row, col);
                if val == CZERO {
                    line.push_str(&format!("{} {}", l, r));
                } else if val.norm() < 1e-10 {
                    line.push_str(&format!("{}0{}", l, r));
                } else {
                    line.push_str(&format!("{}⌧{}", l, r));
                }

                line.push('|');
            }
            println!("{}", line);
        }
    }
}
