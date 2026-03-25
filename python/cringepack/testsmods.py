
from time import time
from scipy.sparse import csc_matrix
import numpy as np
from .matrix import SparseMatrix
from rich.console import Console
from rich.table import Table
from rich import box


class SolveMethod:
    name = 'UNNAMED'
    properties = ''
    istruth: bool = False
    def _solve(self, A: csc_matrix, bs: list[np.ndarray]) -> list[np.ndarray]:
        pass

    def _solve_b(self, b: np.ndarray) -> np.ndarray:
        pass

    def _lu(self, A: csc_matrix):
        pass
    
    def _cholesky(self, A: csc_matrix):
        self._lu(A)

    def solve(self, A: csc_matrix, bs: list[np.ndarray]) -> tuple[float, list[np.ndarray]]:
        t0 = time()
        out = self._solve(A, bs)
        return (time()-t0, out)
    
    def solve_b(self, b: np.ndarray) -> tuple[float, np.ndarray]:
        t0 = time()
        out = self._solve_b(b)
        return (time()-t0, out)

    def lu(self, A: csc_matrix) -> float:
        t0 = time()
        self._lu(A)
        return time()-t0
    
    def cholesky(self, A: csc_matrix) -> float:
        t0 = time()
        self._cholesky(A)
        return time()-t0

def run_tests(matrices: list[SparseMatrix], solvers: list[SolveMethod]):
    results = []
    console = Console()

    truth_solver = None
    for s in solvers:
        if getattr(s, 'istruth', False):
            truth_solver = s.name
            break

    for mat in matrices:
        A = mat.A
        bs = mat.bs
        n = A.shape[0]
        nnz = A.nnz
        mtype = mat.mtype
        for solver in solvers:
            try:
                if mtype=='SPD':
                    t_lu = solver.cholesky(A)
                else:
                    t_lu = solver.lu(A)
                lu_ok = True
            except Exception:
                t_lu = float('nan')
                lu_ok = False

            t_solves = []
            errors = []
            solutions = []
            for b in bs:
                try:
                    t_solve, x = solver.solve_b(b)
                    residual = np.max(np.abs(A @ x - b))
                    t_solves.append(t_solve)
                    errors.append(residual)
                    solutions.append(x)
                except Exception:
                    t_solves.append(float('nan'))
                    errors.append(float('nan'))
                    solutions.append(None)

            t_solve_total = np.nansum(t_solves)
            t_total = t_lu + t_solve_total
            n_rhs = len(bs)
            mdof_lu = (n**2) / t_lu / 1e6 if t_lu > 0 else 0
            mdof_solve = (n**2 * n_rhs) / t_solve_total / 1e6 if t_solve_total > 0 else 0
            #mdof_lu = (nnz) / t_lu / 1e6 if t_lu > 0 else 0
            #mdof_solve = (nnz * n_rhs) / t_solve_total / 1e6 if t_solve_total > 0 else 0
            results.append({
                'matrix': mat.name,
                'solver': solver.name,
                'mtype': mat.mtype,
                'n': n,
                'nnz': nnz,
                'n_rhs': n_rhs,
                't_lu': t_lu,
                't_solve_total': t_solve_total,
                't_total': t_total,
                'mdof_lu': mdof_lu,
                'mdof_solve': mdof_solve,
                'max_residual': np.nanmax(errors),
                'lu_ok': lu_ok,
                'solutions': solutions,
            })

    solver_names = list(dict.fromkeys(r['solver'] for r in results))
    matrix_names = list(dict.fromkeys(r['matrix'] for r in results))

    def get_result(matrix, solver):
        for r in results:
            if r['matrix'] == matrix and r['solver'] == solver:
                return r
        return None

    def solution_error(mname, sname):
        if not truth_solver or sname == truth_solver:
            return None
        r = get_result(mname, sname)
        r_truth = get_result(mname, truth_solver)
        if not r or not r_truth:
            return None
        max_err = 0.0
        for x, x_truth in zip(r['solutions'], r_truth['solutions']):
            if x is not None and x_truth is not None:
                max_err = max(max_err, np.max(np.abs(x - x_truth)))
            else:
                return float('nan')
        return max_err

    def fmt_perf(mdof, t):
        return f"{mdof:.1f} MDoF/s ({t:.3f}s)"

    # Table 1: Timing
    t1 = Table(
        title="⚡ CRINGEPACK BENCHMARK — Performance",
        box=box.ROUNDED,
        show_lines=True,
        title_style="bold cyan",
    )
    t1.add_column("Matrix", style="bold white", min_width=12)
    t1.add_column("Type", justify="center", style="dim")
    t1.add_column("n", justify="right", style="dim")
    t1.add_column("nnz", justify="right", style="dim")
    t1.add_column("rhs", justify="right", style="dim")
    for sname in solver_names:
        t1.add_column(f"{sname}\nFactorize", justify="right", style="yellow")
        t1.add_column(f"{sname}\nSolve", justify="right", style="green")

    for mname in matrix_names:
        row = [mname]
        r0 = get_result(mname, solver_names[0])
        row.append(str(r0['mtype']))
        row.append(str(r0['n']))
        row.append(str(r0['nnz']))
        row.append(str(r0['n_rhs']))
        for sname in solver_names:
            r = get_result(mname, sname)
            if r:
                row.append(fmt_perf(r['mdof_lu'], r['t_lu']))
                row.append(fmt_perf(r['mdof_solve'], r['t_solve_total']))
            else:
                row.extend(["—", "—"])
        t1.add_row(*row)

    console.print()
    console.print(t1)

    # Table 2: Accuracy
    t2 = Table(
        title="🎯 CRINGEPACK BENCHMARK — Accuracy",
        box=box.ROUNDED,
        show_lines=True,
        title_style="bold cyan",
    )
    t2.add_column("Matrix", style="bold white", min_width=12)
    t2.add_column("Type", style="dim")
    for sname in solver_names:
        t2.add_column(f"{sname}\nmax ‖Ax-b‖∞", justify="right")
        if truth_solver and sname != truth_solver:
            t2.add_column(f"{sname}\nvs {truth_solver}", justify="right")
        t2.add_column(f"{sname}\n", justify="center")

    for mname in matrix_names:
        row = [mname]
        r0 = get_result(mname, solver_names[0])
        row.append(str(r0['mtype']))
        for sname in solver_names:
            r = get_result(mname, sname)
            if r:
                passed = r['lu_ok'] and r['max_residual'] < 1e-6
                res_str = f"{r['max_residual']:.2e}"
                row.append(f"[green]{res_str}[/green]" if passed else f"[red]{res_str}[/red]")

                if truth_solver and sname != truth_solver:
                    sol_err = solution_error(mname, sname)
                    if sol_err is not None:
                        row.append(f"[green]{sol_err:.2e}[/green]" if sol_err < 1e-6 else f"[red]{sol_err:.2e}[/red]")
                    else:
                        row.append("—")

                row.append("[bold green]✅[/bold green]" if passed else "[bold red]❌[/bold red]")
            else:
                row.append("—")
                if truth_solver and sname != truth_solver:
                    row.append("—")
                row.append("—")
        t2.add_row(*row)

    console.print()
    console.print(t2)

    # Solver descriptions
    t3 = Table(
        title="📋 Solver Descriptions",
        box=box.ROUNDED,
        show_lines=True,
        title_style="bold cyan",
    )
    t3.add_column("Solver", style="bold white", min_width=12)
    t3.add_column("Description", style="dim")
    t3.add_column("Role", justify="center")
    for solver in solvers:
        desc = getattr(solver, 'properties', '—')
        role = "[bold green]📏 Truth[/bold green]" if getattr(solver, 'istruth', False) else ""
        t3.add_row(solver.name, desc, role)

    console.print()
    console.print(t3)

    # Footer
    console.print()
    console.print("[dim]  MDoF/s = n² / time / 1e6  (factorize: single, solve: × n_rhs)[/dim]")
    console.print("[dim]  max ‖Ax-b‖∞ = worst residual across all RHS vectors[/dim]")
    if truth_solver:
        console.print(f"[dim]  vs {truth_solver} = max ‖x - x_truth‖∞ across all RHS vectors[/dim]")
    console.print()