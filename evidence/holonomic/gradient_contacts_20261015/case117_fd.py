"""Focused public-API gradient/finite-difference check for stress case 117."""
from pathlib import Path
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "evidence/holonomic/lightcurve_matrix_20261015"))
from benchmark import load

np, lc, _ = load()
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
case = json.loads((ROOT / "evidence/holonomic/stress_20261015/case_117.json").read_text())
p = case["params"]
times = np.asarray([-p["rho"] * p["tE"]])


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
for tol in (1e-3, 1e-4, 1e-5, 1e-6):
    result = curve(tol).holonomic_gradient(times, p, mu_atol=tol, mu_rtol=tol)
    order = list(result["param_order"])
    gradient_rows.append(
        {
            "tol": tol,
            "mu": float(result["mu"][0]),
            "value_converged": bool(result["value_converged"][0]),
            "support_certified": bool(result["support_certified"][0]),
            "gradient": {
                key: float(result["grad_mu"][0, order.index(key)])
                for key in ("t0", "rho", "q")
            },
        }
    )


reference_tol = 1e-7
reference = curve(reference_tol)
fd_rows = []
scales = {
    "t0": (p["rho"] * p["tE"], (1e-5, 3e-6, 1e-6)),
    "rho": (p["rho"], (1e-5, 3e-6, 1e-6)),
    "q": (p["q"], (1e-1, 3e-2, 1e-2, 3e-3, 1e-3)),
}
for key, (scale, fractions) in scales.items():
    for fraction in fractions:
        step = scale * fraction
        plus, minus = dict(p), dict(p)
        plus[key] += step
        minus[key] -= step
        a, b = reference.info(times, plus), reference.info(times, minus)
        fd_rows.append(
            {
                "parameter": key,
                "fraction_of_scale": fraction,
                "step": step,
                "fd": float((a["mu"][0] - b["mu"][0]) / (2 * step)),
                "plus_value_converged": bool(a["value_converged"][0]),
                "minus_value_converged": bool(b["value_converged"][0]),
                "plus_support_certified": bool(a["support_certified"][0]),
                "minus_support_certified": bool(b["support_certified"][0]),
            }
        )

print(
    json.dumps(
        {
            "case": case["id"],
            "time": float(times[0]),
            "gradient_tolerance_sweep": gradient_rows,
            "finite_difference_tolerance": reference_tol,
            "finite_differences": fd_rows,
        },
        indent=2,
    )
)
