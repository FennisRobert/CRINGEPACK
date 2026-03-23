use log::{debug, error, info, trace, warn};
use num_complex::{Complex64};
use crate::sparse::{CCMatrixBase, CCMatrixOwned};
use crate::symbolic::CCSymbolic;
const CZERO: Complex64 = Complex64::new(0.0, 0.0);
const CONE: Complex64 = Complex64::new(1.0, 0.0);


pub fn cholesky(matrix: &CCMatrixOwned, symbolic: &CCSymbolic) {
    matrix.printsmall();
    
}