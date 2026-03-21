use std::mem::zeroed;

use ::numpy::PyReadonlyArray1;
use num_complex::Complex64;
use numpy::{ToPyArray, ndarray::ArrayView1};
use pyo3::prelude::*;

mod linsolve;
mod metisperm;
mod pmat;
mod sparse;

use linsolve::SparseSolver;
use sparse::{CCMatrixView, MatrixType};


#[pyclass]
pub struct Cringepack {
    solver: SparseSolver,
}

pub fn gen_mat_view<'a>(
    row: ArrayView1<'a, i64>,
    colptr: ArrayView1<'a, i64>,
    data: ArrayView1<'a, Complex64>,
) -> CCMatrixView<'a> {
    let mtype_enum = MatrixType::Generic;
    CCMatrixView::new(row, colptr, data, mtype_enum)
}

#[pymethods]
impl Cringepack {
    #[new]
    pub fn new() -> Self {
        Cringepack {
            solver: SparseSolver::new(),
        }
    }

    fn factorize(
        &mut self,
        py: Python<'_>,
        row: PyReadonlyArray1<'_, i64>,
        colptr: PyReadonlyArray1<'_, i64>,
        data: PyReadonlyArray1<'_, Complex64>,
    ) {
        let row = row.as_array();
        let colptr = colptr.as_array();
        let data = data.as_array();
        let mut mat = gen_mat_view(row, colptr, data).to_owned();
        self.solver.lu_decomp(&mut mat);
    }

    fn solve_b(
        &mut self,
        py: Python<'_>,
        b: PyReadonlyArray1<'_, Complex64>,
    ) -> PyResult<(Py<numpy::PyArray1<Complex64>>, i64)> {
        let b = b.as_array();
        let (out, code) = self.solver.solve_b(&b);
        Ok((out.to_pyarray(py).into(), code))
    }

    fn solve(
        &mut self,
        py: Python<'_>,
        row: PyReadonlyArray1<'_, i64>,
        colptr: PyReadonlyArray1<'_, i64>,
        data: PyReadonlyArray1<'_, Complex64>,
        b: PyReadonlyArray1<'_, Complex64>,
    ) -> PyResult<(Py<numpy::PyArray1<Complex64>>, i64)> {
        let row = row.as_array();
        let colptr = colptr.as_array();
        let data = data.as_array();
        let mut mat = gen_mat_view(row, colptr, data).to_owned();
        self.solver.lu_decomp(&mut mat);
        let b = b.as_array();
        let (out, code) = self.solver.solve_b(&b);
        Ok((out.to_pyarray(py).into(), code))
    }

    fn metis(
        &mut self,
        py: Python<'_>,
        row: PyReadonlyArray1<'_, i64>,
        colptr: PyReadonlyArray1<'_, i64>,
        data: PyReadonlyArray1<'_, Complex64>,
    ) -> PyResult<(Py<numpy::PyArray1<i64>>, i64)> {
        let row = row.as_array();
        let colptr = colptr.as_array();
        let data = data.as_array();
        let mut mat = gen_mat_view(row, colptr, data).to_owned();
        let out = self.solver.metis(&mut mat);

        Ok((out.to_pyarray(py).into(), 0))
    }
}

#[pymodule]
fn _cringepack(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = env_logger::try_init();
    m.add_class::<Cringepack>()?;
    Ok(())
}
