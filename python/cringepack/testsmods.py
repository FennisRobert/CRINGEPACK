
from time import time
from scipy.sparse import csc_matrix
import numpy as np


class SolveMethod:
    name = 'UNNAMED'
    properties = ''

    def _solve(self, A: csc_matrix, bs: list[np.ndarray]) -> list[np.ndarray]:
        pass

    def _solve_b(self, b: np.ndarray) -> np.ndarray:
        pass

    def _lu(self, A: csc_matrix):
        pass

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


def compare(A: csc_matrix, bs: list[np.ndarray], calA: SolveMethod, calB: SolveMethod):
    n = A.shape[0]
    nnz = A.nnz
    MDOF = n * 2 * 1e-6
    nrhs = len(bs)

    print("=" * 70)
    print("  CRINGEPACK Solver Benchmark")
    print("  Complex Rust INefficient Gaussian Elimination PACKage")
    print("=" * 70)
    print(f"  Matrix size:    {n} x {n}")
    print(f"  Nonzeros:       {nnz} ({100*nnz/(n*n):.2f}% dense)")
    print(f"  RHS vectors:    {nrhs}")
    print(f"  MDOF:           {MDOF:.3f}")
    print("-" * 70)

    # Factorization
    print("\n  [FACTORIZATION]")
    print(f"  {'Solver':<20} {'Time (s)':<14} {'MDOF/s':<14} {'Speedup':<10}")
    print(f"  {'-'*56}")

    time_fact_A = calA.lu(A)
    time_fact_B = calB.lu(A)

    perf_A = MDOF / time_fact_A
    perf_B = MDOF / time_fact_B
    speedup = time_fact_B / time_fact_A if time_fact_A > 0 else float('inf')

    print(f"  {calA.name:<20} {time_fact_A:<14.6f} {perf_A:<14.2f} (ref)")
    print(f"  {calB.name:<20} {time_fact_B:<14.6f} {perf_B:<14.2f} {speedup:.2f}x")

    # Solves
    print(f"\n  [TRIANGULAR SOLVES]")
    print(f"  {'#':<4} {'Solver':<20} {'Time (s)':<14} {'MDOF/s':<14} {'||xA-xB||':<14} {'Status'}")
    print(f"  {'-'*78}")

    total_tA = 0.0
    total_tB = 0.0
    max_err = 0.0

    for i, b in enumerate(bs):
        tA, sol_A = calA.solve_b(b)
        tB, sol_B = calB.solve_b(b)

        error = np.linalg.norm(sol_A - sol_B)
        residual_A = np.linalg.norm(A @ sol_A - b)
        residual_B = np.linalg.norm(A @ sol_B - b)

        max_err = max(max_err, error)
        total_tA += tA
        total_tB += tB

        solve_perf_A = MDOF / tA if tA > 0 else float('inf')
        solve_perf_B = MDOF / tB if tB > 0 else float('inf')

        status = "OK" if error < 1e-6 else "WARN" if error < 1e-3 else "FAIL"
        emoji = "  " if status == "OK" else "!!" if status == "WARN" else "XX"

        print(f"  {i:<4} {calA.name:<20} {tA:<14.6f} {solve_perf_A:<14.2f}")
        print(f"       {calB.name:<20} {tB:<14.6f} {solve_perf_B:<14.2f} {error:<14.2e} {emoji} {status}")

    # Summary
    avg_tA = total_tA / nrhs if nrhs > 0 else 0
    avg_tB = total_tB / nrhs if nrhs > 0 else 0
    avg_perf_A = MDOF / avg_tA if avg_tA > 0 else float('inf')
    avg_perf_B = MDOF / avg_tB if avg_tB > 0 else float('inf')
    solve_speedup = avg_tB / avg_tA if avg_tA > 0 else float('inf')

    print(f"\n  [SUMMARY]")
    print(f"  {'-'*56}")
    print(f"  {'Metric':<30} {calA.name:<14} {calB.name:<14}")
    print(f"  {'-'*56}")
    print(f"  {'Factorization (s)':<30} {time_fact_A:<14.6f} {time_fact_B:<14.6f}")
    print(f"  {'Factorization (MDOF/s)':<30} {perf_A:<14.2f} {perf_B:<14.2f}")
    print(f"  {'Avg solve (s)':<30} {avg_tA:<14.6f} {avg_tB:<14.6f}")
    print(f"  {'Avg solve (MDOF/s)':<30} {avg_perf_A:<14.2f} {avg_perf_B:<14.2f}")
    print(f"  {'Solve speedup':<30} {'(ref)':<14} {solve_speedup:.2f}x")
    print(f"  {'Max ||xA-xB||':<30} {max_err:.2e}")
    print(f"  {'Total time (s)':<30} {time_fact_A+total_tA:<14.6f} {time_fact_B+total_tB:<14.6f}")
    print("=" * 70)