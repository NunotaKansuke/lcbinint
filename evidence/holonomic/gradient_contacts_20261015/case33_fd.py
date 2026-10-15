"""Recheck the historical case_033 rho-gradient anomaly through the public API."""
from pathlib import Path
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "evidence/holonomic/lightcurve_matrix_20261015"))
from benchmark import load

np, lc, _ = load()
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
case = json.loads((ROOT / "evidence/holonomic/stress_20261015/case_033.json").read_text())
p = case["params"]
times = np.asarray([-0.15, 0.15])


def curve(tol):
    return lc.LightCurve(
        options=lc.Options(
            coordinates="vbm",
            nbin="auto",
            finite_source_tol=tol,
            finite_source_reltol=tol,
            finite_source_backend="algebraic_boundary",
        ),
        limb_darkening=lc.LimbDarkening.linear(case["limb"]),
    )


gradient_rows = []
for tol in (1e-3, 1e-5, 1e-7):
    result = curve(tol).holonomic_gradient(times, p, mu_atol=tol, mu_rtol=tol)
    column = list(result["param_order"]).index("rho")
    gradient_rows.append(
        {
            "tol": tol,
            "mu": [float(value) for value in result["mu"]],
            "dmu_drho": [float(value) for value in result["grad_mu"][:, column]],
            "gradient_valid": [bool(value) for value in result["gradient_valid"][:, column]],
            "value_converged": [bool(value) for value in result["value_converged"]],
            "support_certified": [bool(value) for value in result["support_certified"]],
        }
    )


reference_tol = 1e-7
reference = curve(reference_tol)
scale = max(p["rho"], 1e-4)
fd_rows = []
for fraction in (1e-2, 3e-3, 1e-3):
    step = scale * fraction
    plus, minus = dict(p), dict(p)
    plus["rho"] += step
    minus["rho"] -= step
    a, b = reference.info(times, plus), reference.info(times, minus)
    fd_rows.append(
        {
            "fraction_of_scale": fraction,
            "step": step,
            "fd_dmu_drho": [float(value) for value in (a["mu"] - b["mu"]) / (2 * step)],
            "plus_value_converged": [bool(value) for value in a["value_converged"]],
            "minus_value_converged": [bool(value) for value in b["value_converged"]],
            "plus_support_certified": [bool(value) for value in a["support_certified"]],
            "minus_support_certified": [bool(value) for value in b["support_certified"]],
        }
    )

print(
    json.dumps(
        {
            "case": case["id"],
            "times": times.tolist(),
            "gradient_tolerance_sweep": gradient_rows,
            "finite_difference_tolerance": reference_tol,
            "finite_differences": fd_rows,
        },
        indent=2,
    )
)
