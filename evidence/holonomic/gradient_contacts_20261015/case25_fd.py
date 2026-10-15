from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "evidence/holonomic/lightcurve_matrix_20261015"))
from benchmark import load

np, lc, _ = load()
case = json.loads((ROOT / "evidence/holonomic/stress_20261015/case_025.json").read_text())
p = case["params"]
t = np.asarray([-p["rho"] * p["tE"]])

gradients = []
for tol in (1e-3, 1e-4, 1e-5):
    model = lc.LightCurve(
        options=lc.Options(
            coordinates="vbm",
            nbin="auto",
            finite_source_tol=tol,
            finite_source_reltol=tol,
            finite_source_backend="algebraic_boundary",
        ),
        limb_darkening=lc.LimbDarkening.linear(case["limb"]),
    )
    gradient = model.holonomic_gradient(t, p, mu_atol=tol, mu_rtol=tol)
    rho_column = list(gradient["param_order"]).index("rho")
    gradients.append({
        "tol": tol,
        "mu": float(gradient["mu"][0]),
        "gradient_drho": float(gradient["grad_mu"][0, rho_column]),
        "gradient_valid": bool(gradient["gradient_valid"][0, rho_column]),
        "value_converged": bool(gradient["value_converged"][0]),
        "support_certified": bool(gradient["support_certified"][0]),
    })

reference_tol = 1e-5
reference = lc.LightCurve(
    options=lc.Options(
        coordinates="vbm",
        nbin="auto",
        finite_source_tol=reference_tol,
        finite_source_reltol=reference_tol,
        finite_source_backend="algebraic_boundary",
    ),
    limb_darkening=lc.LimbDarkening.linear(case["limb"]),
)
rows = []
for fraction in (1e-7, 1e-8, 1e-9):
    h = p["rho"] * fraction
    plus, minus = dict(p), dict(p)
    plus["rho"] += h
    minus["rho"] -= h
    a, b = reference.info(t, plus), reference.info(t, minus)
    rows.append({
        "fraction_of_rho": fraction,
        "fd_drho": float((a["mu"][0] - b["mu"][0]) / (2 * h)),
        "value_converged": [bool(a["value_converged"][0]), bool(b["value_converged"][0])],
    })
print(json.dumps({
    "case": case["id"],
    "time": float(t[0]),
    "gradient_tolerance_sweep": gradients,
    "finite_difference_tolerance": reference_tol,
    "fd": rows,
}, indent=2))
