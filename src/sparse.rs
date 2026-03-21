use num_complex::{Complex64, ComplexFloat};
use numpy::ndarray::{Array, Array1, ArrayView1, s};
use pyo3::basic::CompareOp;

const CZERO: Complex64 = Complex64::new(0.0, 0.0);
const CONE: Complex64 = Complex64::new(1.0, 0.0);

#[derive(PartialEq, Copy, Clone, Debug)]
pub enum MatrixType {
    Generic,
    Symmetric,
    UnSymmetric,
    LowerTriangular,
    UpperTriangular,
}

pub struct CCMatrixView<'a> {
    pub nnz: usize,
    pub n: usize,
    pub rows: Array1<usize>,
    pub indptr: Array1<usize>,
    pub data: ArrayView1<'a, Complex64>,
    pub maxnic: usize, // max numbers in any column
    pub mtype: MatrixType,
}

pub struct CCMatrixOwned {
    pub nnz: usize,
    pub n: usize,
    pub rows: Array1<usize>,
    pub indptr: Array1<usize>,
    pub data: Array1<Complex64>,
    pub maxnic: usize, // max numbers in any column
    pub mtype: MatrixType,
}

#[derive(Clone)]
pub struct ColBuffer {
    pub nic: usize,
    pub data: Vec<Vec<Complex64>>,
    pub rows: Vec<Vec<usize>>,
    pub colids: Vec<usize>,
    pub coln: Vec<usize>,
    pub inuse: Vec<bool>,
}

impl ColBuffer {
    fn new(nic: usize) -> Self {
        println!("New Column Buffer with size {}", nic);
        let data = vec![vec!(CZERO; nic); nic];
        let rows = vec![vec![0usize; nic]; nic];
        let colids = vec![0usize; nic];
        let coln = vec![0usize; nic];
        let inuse = vec![false; nic];
        ColBuffer {
            nic: nic,
            data: data,
            rows: rows,
            colids: colids,
            coln: coln,
            inuse: inuse,
        }
    }

    fn clear(&mut self, index: usize) {
        self.colids[index] = 0usize;
        self.coln[index] = 0usize;
        self.data[index].fill(CZERO);
        self.rows[index].fill(0usize);
        self.inuse[index] = false;
    }

    fn find_slot(&self) -> Result<usize, String> {
        for i in 0..self.nic {
            if self.inuse[i] == false {
                return Ok(i);
            }
        }
        Err("No free slot available".to_string())
    }
    fn store(
        &mut self,
        icol: usize,
        data: ArrayView1<Complex64>,
        rows: ArrayView1<usize>,
    ) -> Result<(), String> {
        // find first available slot
        println!("Storing dataset {:?}, {:?}", data, rows);
        match self.find_slot() {
            Ok(index) => {
                self.inuse[index] = true;
                self.colids[index] = icol;
                self.coln[index] = data.len();
                for (i, (&c, &r)) in data.iter().zip(rows.iter()).enumerate() {
                    self.data[index][i] = c;
                    self.rows[index][i] = r;
                }
            }
            Err(message) => return Err(message),
        }

        Ok(())
    }
    fn is_buffered(&self, index: usize) -> Result<usize, ()> {
        for i in 0..self.nic {
            if self.colids[i] == index {
                return Ok(i);
            }
        }
        return Err(());
    }
}
pub trait CCMatrixBase {
    fn get_rows(&self) -> ArrayView1<usize>;
    fn get_indptr(&self) -> ArrayView1<usize>;
    fn get_data(&self) -> ArrayView1<Complex64>;
    fn get_n(&self) -> usize;
    fn get_nnz(&self) -> usize;
    fn get_maxnic(&self) -> usize;
    fn get_mtype(&self) -> MatrixType;

    fn transpose_gen(&self, conjugate: bool) -> CCMatrixOwned {
        let mut indptr = Array1::<usize>::zeros(self.get_indptr().len());
        let mut rows = Array1::<usize>::zeros(self.get_rows().len());
        let mut data = Array1::<Complex64>::zeros(self.get_data().len());

        let mut colid = Array1::<usize>::zeros(self.get_rows().len());
        let mut dataid: Vec<usize> = (0..self.get_nnz()).collect();

        for icol in 0..self.get_n() {
            let id1 = self.get_indptr()[icol] as usize;
            let id2 = self.get_indptr()[icol + 1] as usize;
            for i in id1..id2 {
                colid[i] = icol;
            }
        }

        dataid.sort_by(|&a, &b| {
            self.get_rows()[a]
                .cmp(&self.get_rows()[b])
                .then(colid[a].cmp(&colid[b]))
        });
        let mut current: usize = 0;
        let mut indptrptr: usize = 0;

        for (io, id) in dataid.iter().enumerate() {
            data[io] = self.get_data()[*id];
            rows[io] = colid[*id];
            if self.get_rows()[*id] != current {
                current = self.get_rows()[*id];
                indptrptr = indptrptr + 1;
                indptr[indptrptr] = io;
            }
        }
        indptr[self.get_n()] = self.get_indptr()[self.get_n()];
        if conjugate == true {
            data = data.map(|&x| x.conj());
        }
        let mut mtype: MatrixType = self.get_mtype();
        match mtype {
            MatrixType::LowerTriangular => {
                mtype = MatrixType::UpperTriangular;
            }
            MatrixType::UpperTriangular => {
                mtype = MatrixType::LowerTriangular;
            }
            _ => {}
        };
        CCMatrixOwned::new(rows, indptr, data, mtype)
    }

    fn mult_vec(&self, vec: &ArrayView1<Complex64>) -> Array1<Complex64> {
        let mut result = Array1::zeros(self.get_n());
        for icol in 0..self.get_n() {
            let val = vec[icol];

            if val == CZERO {
                continue;
            }
            let id1 = self.get_indptr()[icol];
            let id2 = self.get_indptr()[icol + 1];
            for idat in id1..id2 {
                result[self.get_rows()[idat]] += vec[icol] * self.get_data()[idat];
            }
        }

        result
    }

    fn get(&self, row: usize, col: usize) -> Complex64 {
        let idstart = self.get_indptr()[col] as usize;
        let idend = self.get_indptr()[col + 1] as usize;
        let mut out = Complex64::new(0.0, 0.0);
        let pos = self
            .get_rows()
            .slice(s![idstart..idend])
            .iter()
            .position(|&r| r == row);
        match pos {
            Some(index) => {
                out = self.get_data()[index + idstart];
            }
            None => {}
        }
        out
    }

    fn transpose(&self) -> CCMatrixOwned {
        self.transpose_gen(false)
    }

    fn conjtranspose(&self) -> CCMatrixOwned {
        self.transpose_gen(true)
    }

    fn printsmall(&self) {
        println!("Matrix = ");
        for irow in 0..self.get_n() {
            let mut line = String::new();
            for icol in 0..self.get_n() {
                let val = self.get(irow, icol);
                line.push_str(&format!("{:.3} ", val.re));
            }
            println!("{}", line);
        }
    }

    fn iter_col(&self, col: usize) -> impl Iterator<Item = (usize, Complex64)> + '_ {
        let start = self.get_indptr()[col];
        let end = self.get_indptr()[col + 1];
        (start..end).map(|i| (self.get_rows()[i], self.get_data()[i]))
    }
}

impl CCMatrixBase for CCMatrixOwned {
    fn get_rows(&self) -> ArrayView1<'_, usize> {
        self.rows.view()
    }
    fn get_indptr(&self) -> ArrayView1<'_, usize> {
        self.indptr.view()
    }
    fn get_data(&self) -> ArrayView1<'_, Complex64> {
        self.data.view()
    }
    fn get_n(&self) -> usize {
        self.n
    }
    fn get_nnz(&self) -> usize {
        self.nnz
    }
    fn get_maxnic(&self) -> usize {
        self.maxnic
    }
    fn get_mtype(&self) -> MatrixType {
        self.mtype
    }
}

impl<'a> CCMatrixBase for CCMatrixView<'a> {
    fn get_rows(&self) -> ArrayView1<'_, usize> {
        self.rows.view()
    }
    fn get_indptr(&self) -> ArrayView1<'_, usize> {
        self.indptr.view()
    }
    fn get_data(&self) -> ArrayView1<'_, Complex64> {
        self.data
    }
    fn get_n(&self) -> usize {
        self.n
    }
    fn get_nnz(&self) -> usize {
        self.nnz
    }
    fn get_maxnic(&self) -> usize {
        self.maxnic
    }
    fn get_mtype(&self) -> MatrixType {
        self.mtype
    }
}

impl CCMatrixOwned {
    pub fn new(
        rows: Array1<usize>,
        indptr: Array1<usize>,
        data: Array1<Complex64>,
        mtype: MatrixType,
    ) -> Self {
        let nnz = data.len();
        let n = indptr.len() - 1;
        let maxnic: usize = (0..indptr.len() - 1)
            .map(|i| (indptr[i + 1] - indptr[i]) as usize)
            .max()
            .unwrap_or(0);
        let mat = CCMatrixOwned {
            nnz,
            n,
            rows,
            indptr,
            data,
            maxnic: maxnic,
            mtype: mtype,
        };
        mat
    }

    pub fn empty() -> Self {
        CCMatrixOwned {
            nnz: 0,
            n: 0,
            rows: Array1::<usize>::zeros(0),
            indptr: Array1::<usize>::zeros(1),
            data: Array1::<Complex64>::zeros(0),
            maxnic: 0,
            mtype: MatrixType::Generic,
        }
    }

    pub fn iter_col(&self, col: usize) -> impl Iterator<Item = (usize, Complex64)> + '_ {
        let start = self.indptr[col];
        let end = self.indptr[col + 1];
        (start..end).map(|i| (self.rows[i], self.data[i]))
    }

    fn get(&self, row: usize, col: usize) -> Complex64 {
        let idstart = self.indptr[col] as usize;
        let idend = self.indptr[col + 1] as usize;
        let mut out = Complex64::new(0.0, 0.0);
        let pos = self
            .get_rows()
            .slice(s![idstart..idend])
            .iter()
            .position(|&r| r == row);
        match pos {
            Some(index) => {
                out = self.get_data()[index + idstart];
            }
            None => {}
        }
        out
    }

    pub fn udiag(&self, ij: usize) -> Complex64 {
        let itest = self.indptr[ij + 1] as usize - 1usize;
        if self.get_rows()[itest] == ij {
            return self.get_data()[itest];
        }

        let idstart = self.get_indptr()[ij] as usize;
        let idend = self.get_indptr()[ij + 1] as usize - 1usize;
        for i in (idstart..idend).rev() {
            if self.get_rows()[i] == ij {
                return self.get_data()[i];
            }
        }
        Complex64::new(0.0, 0.0)
    }

    fn mult_vec(&self, vec: &ArrayView1<Complex64>) -> Array1<Complex64> {
        let mut result = Array1::zeros(self.n);
        for icol in 0..self.n {
            let val = vec[icol];

            if val == CZERO {
                continue;
            }
            let id1 = self.indptr[icol];
            let id2 = self.indptr[icol + 1];
            for idat in id1..id2 {
                result[self.rows[idat]] += vec[icol] * self.data[idat];
            }
        }

        result
    }

    fn n_in_col(&self, col: usize) -> usize {
        self.get_indptr()[col + 1] - self.get_indptr()[col]
    }

    fn mult_mat(&self, matb: &CCMatrixOwned) -> CCMatrixOwned {
        let mut indptr = Array1::<usize>::zeros(self.get_indptr().len());

        let mut nnz: usize = 0;
        let mut cola_nz = Array1::from_elem(self.get_n(), false);
        let mut maxn: usize = 0;

        for col_of_b in 0..matb.get_n() {
            let idb1 = matb.get_indptr()[col_of_b] as usize;
            let idb2 = matb.get_indptr()[col_of_b + 1] as usize;

            // for each row in the column of matrix B...
            // Thus for each column in matrix A
            let mut colctr: usize = 0;
            for irb_ptr in idb1..idb2 {
                // extract the column index in A (row of B)
                let col_of_a = matb.get_rows()[irb_ptr] as usize;

                // for each row id ptr in the column of A
                for ira_ptr in self.get_indptr()[col_of_a]..self.get_indptr()[col_of_a + 1] {
                    // non-zero in the column of A at row = True
                    cola_nz[self.get_rows()[ira_ptr as usize] as usize] = true;
                    colctr += 1;
                }
            }
            // The maximum number of total multiplications in all columns
            maxn = maxn.max(colctr);
            let n = cola_nz.iter().filter(|x| **x).count();
            nnz = nnz + n;
        }

        maxn = maxn as usize;

        let mut rows = Array1::<usize>::zeros(nnz);
        let mut data = Array1::<Complex64>::zeros(nnz);

        let mut mult_data = vec![Complex64::new(0.0, 0.0); maxn];
        let mut mult_row = vec![maxn + 1; maxn];

        let mut pointer: usize = 0;

        for col_of_b in 0..matb.get_n() {
            mult_data.fill(Complex64::new(0.0, 0.0));
            mult_row.fill(maxn + 1);
            let mut mult_id: Vec<usize> = (0..maxn).collect();
            let idb1 = matb.get_indptr()[col_of_b] as usize;
            let idb2 = matb.get_indptr()[col_of_b + 1] as usize;
            let mut mctr: usize = 0; // Counts the total multiplications performed
            for idata_b in idb1..idb2 {
                let cola_rowb = matb.get_rows()[idata_b] as usize;
                for idata_a in self.get_indptr()[cola_rowb]..self.get_indptr()[cola_rowb + 1] {
                    let row_of_a = self.get_rows()[idata_a as usize] as usize;
                    mult_data[mctr] = matb.get_data()[idata_b] * self.get_data()[idata_a as usize];
                    mult_row[mctr] = row_of_a;
                    mctr += 1;
                }
            }
            mult_id.sort_by_key(|&i| mult_row[i]);
            let mut current_row = mult_row[mult_id[0]];

            for pt in 0..mctr {
                let indx = mult_id[pt];
                let next_row = mult_row[indx];
                if next_row > current_row {
                    current_row = next_row;
                    pointer += 1;
                    rows[pointer] = current_row;
                }
                data[pointer] += mult_data[indx];
                rows[pointer] = current_row;
            }
            pointer += 1;
            indptr[col_of_b + 1] = pointer;
        }
        indptr[self.get_n()] = nnz;
        CCMatrixOwned::new(rows, indptr, data, MatrixType::Generic)
    }

    pub fn permute(&mut self, permvec: &Vec<usize>) {
        let mut inv_perm = vec![0usize; permvec.len()];
        for i in 0..permvec.len() {
            inv_perm[permvec[i]] = i;
        }
        let mut data = Array1::<Complex64>::zeros(self.nnz);
        let mut rows = Array1::<usize>::zeros(self.nnz);
        let mut indptr = Array1::<usize>::zeros(self.indptr.len());

        let mut ptr = 0usize;

        for i in 0..self.n {
            let colnr = permvec[i];
            let i1 = self.indptr[colnr];
            let i2 = self.indptr[colnr + 1];
            let n = i2 - i1;
            for (j, index) in (i1..i2).enumerate() {
                data[ptr + j] = self.data[index];
                rows[ptr + j] = inv_perm[self.rows[index]];
            }
            ptr += n;
            indptr[i + 1] = ptr;
        }
        self.rows = rows;
        self.data = data;
        self.indptr = indptr;
    }
    // pub fn permute(&mut self, permvec: Vec<usize>) {
    //     // if self.n < 500_000 {
    //     //     return self.quick_permute(permvec)
    //     // }
    //     return self.quick_permute(permvec);

    //     let mut buffer = ColBuffer::new(self.maxnic);

    //     for i in 0..self.maxnic {
    //         let i1 = self.indptr[i];
    //         let i2 = self.indptr[i+1];
    //         buffer.store(i, self.data.slice(s![i1..i2]),self.rows.slice(s![i1..i2])).unwrap();
    //     }

    //     // first Maxnic columns are free to ovewrite
    //     let mut data: Vec<Complex64> = vec![CZERO; self.maxnic];
    //     let mut rows: Vec<usize> = vec![0usize; self.maxnic];
    //     let mut n: usize;

    //     let mut buffer_col: usize = self.maxnic;
    //     let mut ptr = 0usize;

    //     for ii in 0..self.n-self.maxnic {
    //         let icol = permvec[ii];

    //         match buffer.is_buffered(icol) {
    //             Ok(index) => {
    //                 data = buffer.data[index].clone();
    //                 rows = buffer.rows[index].clone();
    //                 n = buffer.coln[index];
    //                 buffer.clear(index);
    //             },
    //             Err(_) => {
    //                 let i1 = self.indptr[icol];
    //                 let i2 = self.indptr[icol+1];
    //                 for (i, (r,c)) in self.iter_col(icol).enumerate() {
    //                     data[i] = c;
    //                     rows[i] = r;
    //                 }
    //                 n = i2-i1
    //             }
    //         }
    //         // from here the data is ready
    //         self.indptr[ii+1] = n;
    //         for j in 0..n {
    //             self.data[ptr + j] = data[j];
    //             self.rows[ptr + j] = permvec[rows[j]];
    //         }
    //         ptr = ptr + n;
    //         if buffer_col < self.n {
    //             let i1 = self.indptr[buffer_col];
    //             let i2 = self.indptr[buffer_col];
    //             buffer.store(buffer_col, self.data.slice(s![i1..i2]),self.rows.slice(s![i1..i2])).unwrap();
    //         }
    //         buffer_col = buffer_col + 1;

    //         data.fill(CZERO);
    //         rows.fill(0usize);
    //     }

    // }

    fn add_mat_gen(&self, matb: &CCMatrixOwned, coeff: Complex64) -> CCMatrixOwned {
        if self.get_n() != matb.get_n() {
            panic!(
                "Trying to add two matrices with dissimilar shape {}, {}",
                self.get_n(),
                matb.get_n()
            );
        }
        let mut nnz: usize = 0;
        let nmax = self.get_maxnic() + matb.get_maxnic();
        let mut rowids = Vec::with_capacity(nmax);
        for icol in 0..self.get_n() {
            let mut i: usize = 0;
            rowids.clear();
            for (row, data) in self.iter_col(icol) {
                rowids.push(row);
                i += 1;
            }
            for (row, data) in matb.iter_col(icol) {
                rowids.push(row);
                i += 1;
            }
            rowids.sort();
            rowids.dedup();
            nnz += rowids.len();
        }
        let mut indptr = Array1::<usize>::zeros(self.get_indptr().len());
        let mut rows = Array1::<usize>::zeros(nnz);
        let mut data = Array1::<Complex64>::zeros(nnz);

        let mut rowdata = Vec::with_capacity(nmax);

        let mut pointer: usize = 0;

        for icol in 0..self.get_n() {
            let mut i: usize = 0;
            rowids.clear();
            rowdata.clear();
            for (row, data) in self.iter_col(icol) {
                rowids.push(row);
                rowdata.push(data);
                i += 1;
            }
            for (row, data) in matb.iter_col(icol) {
                rowids.push(row);
                rowdata.push(coeff * data);
                i += 1;
            }
            let mut ids: Vec<usize> = (0..i).collect();
            ids.sort_by_key(|&i| rowids[i]);
            let mut current_row: usize = rowids[ids[0]];

            for pt in 0..i {
                let indx = ids[pt];
                let next_row = rowids[indx];
                if next_row > current_row {
                    current_row = next_row;
                    pointer += 1;
                    rows[pointer] = current_row;
                }
                data[pointer] += rowdata[indx];
                rows[pointer] = current_row;
            }
            pointer += 1;
            indptr[icol + 1] = pointer;
        }
        indptr[self.get_n()] = nnz;
        CCMatrixOwned::new(rows, indptr, data, MatrixType::Generic)
    }

    fn add_mat(&self, matb: &CCMatrixOwned) -> CCMatrixOwned {
        self.add_mat_gen(matb, Complex64::new(1.0, 0.0))
    }

    fn sub_mat(&self, matb: &CCMatrixOwned) -> CCMatrixOwned {
        self.add_mat_gen(matb, Complex64::new(-1.0, 0.0))
    }
}

impl<'a> CCMatrixView<'a> {
    pub fn new(
        rows: ArrayView1<'a, i64>,
        indptr: ArrayView1<'a, i64>,
        data: ArrayView1<'a, Complex64>,
        mtype: MatrixType,
    ) -> Self {
        let nnz = data.len();
        let n = indptr.len() - 1;
        let maxnic: usize = (0..indptr.len() - 1)
            .map(|i| (indptr[i + 1] - indptr[i]) as usize)
            .max()
            .unwrap_or(0);
        let rows_usize: Array1<usize> = rows.to_owned().mapv(|x| x as usize);
        let indptr_usize: Array1<usize> = indptr.to_owned().mapv(|x| x as usize);
        let mat = CCMatrixView {
            nnz,
            n,
            rows: rows_usize,
            indptr: indptr_usize,
            data,
            maxnic: maxnic,
            mtype: mtype,
        };

        mat
    }

    pub fn iter_col(&self, col: usize) -> impl Iterator<Item = (usize, Complex64)> + '_ {
        let start = self.indptr[col];
        let end = self.indptr[col + 1];
        (start..end).map(|i| (self.rows[i], self.data[i]))
    }

    pub fn to_owned(&self) -> CCMatrixOwned {
        CCMatrixOwned::new(
            self.rows.to_owned(),
            self.indptr.to_owned(),
            self.data.to_owned(),
            self.mtype,
        )
    }
}
