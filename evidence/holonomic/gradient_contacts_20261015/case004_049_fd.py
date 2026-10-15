"""Recheck formerly invalid symmetric stress rows through the public API."""
from pathlib import Path
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "evidence/holonomic/lightcurve_matrix_20261015"))
from benchmark import load

np, lc, _ = load()
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})


def curve(case, tol):
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


rows = []
for name in ("case_004", "case_049"):
    case = json.loads((ROOT / "evidence/holonomic/stress_20261015" / f"{name}.json").read_text())
    p = case["params"]
    times = np.asarray([0.0])
    gradient = curve(case, 1e-3).holonomic_gradient(times, p, mu_atol=1e-3, mu_rtol=1e-3)
    rho_column = list(gradient["param_order"]).index("rho")
    reference = curve(case, 1e-7)
    differences = []
    for fraction in (1e-2, 3e-3, 1e-3):
        step = p["rho"] * fraction
        plus, minus = dict(p), dict(p)
        plus["rho"] += step
        minus["rho"] -= step
        a, b = reference.info(times, plus), reference.info(times, minus)
        differences.append(
            {
                "fraction_of_rho": fraction,
                "fd_drho": float((a["mu"][0] - b["mu"][0]) / (2 * step)),
                "methods": [str(a["finite_source_method_name"][0]), str(b["finite_source_method_name"][0])],
                "value_converged": [bool(a["value_converged"][0]), bool(b["value_converged"][0])],
                "support_certified": [bool(a["support_certified"][0]), bool(b["support_certified"][0])],
            }
        )
    rows.append(
        {
            "case": name,
            "mu": float(gradient["mu"][0]),
            "gradient_valid": [bool(value) for value in gradient["gradient_valid"][0]],
            "dmu_drho": float(gradient["grad_mu"][0, rho_column]),
            "value_converged": bool(gradient["value_converged"][0]),
            "support_certified": bool(gradient["support_certified"][0]),
            "fd_tolerance": 1e-7,
            "finite_differences": differences,
        }
    )

print(json.dumps({"results": rows}, indent=2))
