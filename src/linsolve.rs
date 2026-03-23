use log::{debug, error, info, trace, warn};
use num_complex::{Complex64, ComplexFloat};
use numpy::ndarray::{Array1, ArrayView1, s};

use crate::pmat::{PMatrix};
use crate::sparse::{CCMatrixBase, CCMatrixOwned, MatrixType};
use crate::perm::Permutation;

const CZERO: Complex64 = Complex64::new(0.0, 0.0);
const CONE: Complex64 = Complex64::new(1.0, 0.0);
const PIVOT_TH: f64 = 0.0001;
const PIVOT_SQ: f64 = PIVOT_TH * PIVOT_TH;

fn zero_vec(arr: &ArrayView1<Complex64>) -> Array1<Complex64> {
    Array1::<Complex64>::zeros(arr.len())
}


fn solve_drhs_isl(matrix: &CCMatrixOwned, vec: &ArrayView1<Complex64>) -> Array1<Complex64> {
    debug!("Calling Lower Triangular solve routine...");
    let mut sol = vec.to_owned();
    for j in 0..matrix.n - 1 {
        for (i, d) in matrix.iter_col(j).filter(|&(i, _)| i > j) {
            sol[i] = sol[i] - d * sol[j];
        }
    }
    sol
}

fn solve_drhs_isl_sparse(
    matrix: &CCMatrixOwned,
    vec: &ArrayView1<Complex64>,
    xset: &Vec<usize>,
) -> Array1<Complex64> {
    debug!("Calling Sparse Lower Triangular solve routine...");
    let mut sol = vec.to_owned();
    for &j in xset.iter().rev() {
        sol[j] = sol[j]; // / matrix.get(j,j);

        for (i, d) in matrix.iter_col(j).filter(|&(i, _)| i > j) {
            sol[i] = sol[i] - d * sol[j];
        }
    }
    sol
}

fn solve_drhs_isu(matrix: &CCMatrixOwned, vec: &ArrayView1<Complex64>) -> Array1<Complex64> {
    debug!("Calling Upper Triangular solve routine...");
    //println!("{:?}", matrix.indptr());
    let mut sol = vec.to_owned();
    for j in (0..matrix.n).rev() {
        //sol[j] = sol[j] / matrix.get(j,j);
        sol[j] = sol[j] / matrix.udiag(j);
        //sol[j] = sol[j] / matrix.data()[matrix.indptr()[j+1] as usize -1usize];
        let p1 = matrix.indptr[j] as usize;
        let p2 = matrix.indptr[j + 1] as usize;
        for p in p1..p2 {
            let i = matrix.rows[p] as usize;
            if i < j {
                sol[i] = sol[i] - matrix.data[p] * sol[j];
            }
        }
    }
    sol
}
fn solve_drhs_isu_sparse(
    matrix: &CCMatrixOwned,
    vec: &ArrayView1<Complex64>,
    xset: &Vec<usize>,
) -> Array1<Complex64> {
    debug!("Calling Sparse Upper Triangular solve routine...");
    let mut sol = vec.to_owned();
    for &j in xset.iter().rev() {
        sol[j] = sol[j] / matrix.get(j, j);
        for (i, d) in matrix.iter_col(j).filter(|&(i, _)| i < j) {
            sol[i] = sol[i] - d * sol[j];
        }
    }
    sol
}

fn generic_lu(matrix: &CCMatrixOwned) -> (CCMatrixOwned, CCMatrixOwned, Vec<usize>) {
    debug!("Generating lower and upper triangular matrices...");
    let mut lower = PMatrix::eye(matrix.get_n(), matrix.get_maxnic() * matrix.get_maxnic());
    let mut upper = PMatrix::from_cc(matrix);
    let mut perm: Vec<usize> = (0..matrix.get_n()).collect();

    let n = matrix.get_n();

    //lower.printsmall("LOWER:");
    //upper.printsmall("UPPER:");
    // Start standard gaussian elimination
    for icol in 0..(n - 1) {
        //debug!("Pivoting column {}", icol);
        let mut pivot = upper.get(icol, icol);
        // Partial Pivoting
        //upper.printnz("UPPER:");
        if pivot.norm_sqr() < PIVOT_SQ {
            let (i, row, val) = upper.getcol(icol).max(icol);
            let swap_pair = (icol, row);
            //println!("Pivoting rows {}←→{}", icol, row);
            upper.ppivot(swap_pair, upper.n);
            lower.ppivot(swap_pair, icol);
            perm.swap(icol, row);
            pivot = val;
            //upper.printnz("UPPER:");
        }

        let (rows, data) = upper.iter_col(icol, icol + 1);

        //println!("Eliminating col {}, with:", icol);
        //println!("row = {:?}", rows);
        //println!("data = {:?}", data);
        //upper.printsmall("BEFORE ELIM:");
        upper.eliminate_rows(icol, &rows, &data.iter().map(|v| v / pivot).collect());

        for (irow, val) in rows.iter().zip(data) {
            let val_new = -val / pivot;
            lower.set(*irow, icol, -val_new);
        }
    }
    //lower.printsmall("LOWER:");
    //upper.printsmall("UPPER:");
    //upper.printnz("UPPER:");
    (
        lower.to_ccmatrix(MatrixType::LowerTriangular),
        upper.to_ccmatrix(MatrixType::UpperTriangular),
        perm,
    )
}

pub struct SparseSolver {
    lower: CCMatrixOwned,
    upper: CCMatrixOwned,
    initialized: bool,
    perm_ppiv: Vec<usize>,
    perm_fill: Permutation,
    xset: Vec<usize>,
    xmark: Vec<bool>,
}

impl SparseSolver {
    pub fn new() -> Self {
        SparseSolver {
            lower: CCMatrixOwned::empty(),
            upper: CCMatrixOwned::empty(),
            initialized: false,
            perm_ppiv: Vec::new(),
            perm_fill: Permutation::empty(),
            xset: Vec::new(),
            xmark: Vec::new(),
        }
    }

    pub fn lu_decomp(&mut self, matrix: &mut CCMatrixOwned) {
        self.perm_fill = Permutation::metis(&matrix);
        matrix.permute(&self.perm_fill.perm);
        let (lower, upper, perm) = generic_lu(matrix);
        self.lower = lower;
        self.upper = upper;
        self.perm_ppiv = perm;
        self.initialized = true;
        debug!("LU Decomposition complete!");
    }

    pub fn xset_dfs_lower(&mut self, j: usize, depth: usize) {
        self.xmark[j] = true;

        let neighbors: Vec<(usize, Complex64)> = self.lower.iter_col(j).collect();
        for (r, _) in neighbors {
            if r != j && self.xmark[r] != true && depth + 1 < self.lower.get_n() {
                self.xset_dfs_lower(r, depth + 1usize);
            }
        }
        self.xset.push(j);
    }
    pub fn xset_dfs_upper(&mut self, j: usize, depth: usize) {
        self.xmark[j] = true;

        let neighbors: Vec<(usize, Complex64)> = self.upper.iter_col(j).collect();
        for (r, _) in neighbors {
            if r != j && self.xmark[r] != true && depth + 1 < self.upper.get_n() {
                self.xset_dfs_upper(r, depth + 1usize);
            }
        }
        self.xset.push(j);
    }

    pub fn build_reach_L(&mut self, bvec: &ArrayView1<Complex64>) {
        debug!("Building reach(L,B) size {}", self.lower.get_n());
        self.xset = Vec::with_capacity(self.lower.get_n());
        self.xmark = vec![false; self.lower.get_n()];

        for i in 0..bvec.len() {
            if bvec[i] != CZERO {
                if !self.xmark[i] {
                    self.xset_dfs_lower(i, 0);
                }
            }
        }
    }

    pub fn build_reach_U(&mut self, bvec: &ArrayView1<Complex64>) {
        debug!("Building reach(U,B) size {}", self.upper.get_n());
        self.xset = Vec::with_capacity(self.upper.get_n());
        self.xmark = vec![false; self.upper.get_n()];

        for i in 0..bvec.len() {
            if bvec[i] != CZERO {
                if !self.xmark[i] {
                    self.xset_dfs_upper(i, 0);
                }
            }
        }
    }

    pub fn permute_b(&self, bvec: &ArrayView1<Complex64>) -> Array1<Complex64> {
        let mut b_reord = Array1::<Complex64>::zeros(bvec.len());
        for i in 0..bvec.len() {
            b_reord[i] = bvec[self.perm_fill.perm[i]];
        }
        let mut b_perm = Array1::<Complex64>::zeros(bvec.len());
        for i in 0..bvec.len() {
            b_perm[i] = b_reord[self.perm_ppiv[i]];
        }
        b_perm
    }

    pub fn unpermute_x(&self, x: &Array1<Complex64>) -> Array1<Complex64> {
        let mut x_out = Array1::<Complex64>::zeros(x.len());
        for i in 0..x.len() {
            x_out[self.perm_fill.perm[i]] = x[i];
        }
        x_out
    }

    pub fn refine_b(
        &mut self,
        mat: &impl CCMatrixBase,
        bvec: &ArrayView1<Complex64>,
        xvec: &Array1<Complex64>,
        steps: i64,
    ) -> Array1<Complex64> {
        debug!("Iterative refinment...");
        let mut xnew = xvec.to_owned();
        for step in 0..steps {
            let r = bvec - mat.mult_vec(&xnew.view());
            let (dx, err) = self.solve_b(&r.view());
            xnew = xnew + dx
        }
        xnew
    }

    pub fn solve_b(&mut self, bvec: &ArrayView1<Complex64>) -> (Array1<Complex64>, i64) {
        debug!("Solving b.");
        if !self.initialized {
            debug!("Solver not initialized.");
            return (zero_vec(bvec), -1);
        }
        //self.build_reach_L(bvec);
        let b_perm = self.permute_b(bvec);
        let y = solve_drhs_isl(&self.lower, &b_perm.view());
        //self.build_reach_U(&y.view());
        let mut sol = solve_drhs_isu(&self.upper, &y.view());
        sol = self.unpermute_x(&sol);
        (sol, 0)
    }

    pub fn solve_dense(
        &mut self,
        matrix: &mut CCMatrixOwned,
        vec: &ArrayView1<Complex64>,
    ) -> (Array1<Complex64>, i64) {
        // Exit codes
        // -1 = error
        // -0 = Succes
        match matrix.get_mtype() {
            MatrixType::LowerTriangular => (solve_drhs_isl(matrix, vec), 0),
            MatrixType::UpperTriangular => (solve_drhs_isu(matrix, vec), 0),
            MatrixType::Generic => {
                if !self.initialized {
                    self.lu_decomp(matrix);
                }
                let (x, err) = self.solve_b(vec);
                if err != 0 {
                    (x, err)
                } else {
                    let x2 = self.refine_b(matrix, vec, &x, 2);
                    (x2, 0)
                }
            }
            _ => {
                println!("Matrix type {:?} not supported!", matrix.get_mtype());
                (zero_vec(vec), -1)
            }
        }
    }
}
