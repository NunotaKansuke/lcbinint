"""Linear-limb-darkening speed comparison: VBM vs lcbinint mainstream vs the
experimental algebraic-boundary kernel (primal and with the analytic Jacobian).

Builds and runs ``benchmark_ld_compare`` for the lcbinint numbers, then times
VBMicrolensing's ``BinaryMagDark`` on the identical geometries in-process.
"""

from __future__ import annotations

import csv
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ABS_FLOOR = 1.0e-12


def timed(call, repeats: int) -> tuple[float, float]:
    best = float("inf")
    value = 0.0
    for _ in range(repeats):
        t0 = time.process_time()
        for _ in range(20):
            value = call()
        dt = (time.process_time() - t0) / 20.0
        best = min(best, dt)
    return value, best


def main() -> int:
    repeats = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    exe = ROOT / "build" / "benchmark_ld_compare"
    rows = list(csv.DictReader(
        subprocess.run([str(exe)], check=True, capture_output=True, text=True)
        .stdout.splitlines()))

    import VBMicrolensing

    print(f"{'case':<20} {'tol':>6} {'rho':>7} | "
          f"{'VBM':>10} {'mainstream':>11} {'alg_primal':>11} {'alg+jac':>10} | "
          f"{'alg/VBM':>8} {'jac/VBM':>8}  mp  err")
    for row in rows:
        s = float(row["sep"]); q = float(row["q"]); rho = float(row["rho"])
        x = float(row["mid_x"]); y = float(row["mid_y"])
        tol = float(row["tolerance"]); c = float(row["limb_c"])

        vbm = VBMicrolensing.VBMicrolensing()
        vbm.Tol = ABS_FLOOR
        vbm.RelTol = tol
        vbm.a1 = c
        vbm.SetLDprofile(vbm.LDlinear)
        vbm_value, vbm_s = timed(
            lambda: vbm.BinaryMagDark(s, q, x, y, rho, ABS_FLOOR), repeats)

        alg_primal_value = float(row["alg_primal_mag"])
        err = abs(alg_primal_value - vbm_value) / max(abs(vbm_value), 1.0)
        us = 1.0e6
        vbm_us = vbm_s * us
        main_us = float(row["mainstream_us"])
        prim_us = float(row["alg_primal_us"])
        jac_us = float(row["alg_jacobian_us"])
        print(f"{row['case']:<20} {tol:>6.0e} {rho:>7.4f} | "
              f"{vbm_us:>10.1f} {main_us:>11.1f} {prim_us:>11.1f} {jac_us:>10.1f} | "
              f"{prim_us / vbm_us:>8.2f} {jac_us / vbm_us:>8.2f} "
              f"{row['alg_multipole_hits']:>3} {err:>8.1e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
