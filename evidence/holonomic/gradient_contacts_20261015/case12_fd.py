from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "evidence/holonomic/lightcurve_matrix_20261015"))
from benchmark import load

np, lc, _ = load()
case = json.loads((ROOT / "evidence/holonomic/stress_20261015/case_012.json").read_text())
p = case["params"]
t = np.asarray(np.load(ROOT / "evidence/holonomic/stress_20261015/case_012.npz")["times"][[8]])
tol = 1e-9

def curve():
    return lc.LightCurve(
        options=lc.Options(
            coordinates="vbm",
            finite_source_backend="algebraic_boundary",
            finite_source_tol=tol,
            finite_source_reltol=tol,
        ),
        limb_darkening=lc.LimbDarkening.linear(0.0),
    )

model = curve()
gradient = model.holonomic_gradient(t, p, mu_atol=tol, mu_rtol=tol)
column = list(gradient["param_order"]).index("rho")
rows = []
for fraction in (1e-7, 3e-8, 1e-8, 3e-9, 1e-9, 3e-10):
    h = p["rho"] * fraction
    plus, minus = dict(p), dict(p)
    plus["rho"] += h
    minus["rho"] -= h
    a, b = model.info(t, plus), model.info(t, minus)
    rows.append({
        "fraction_of_rho": fraction,
        "fd_drho": float((a["mu"][0] - b["mu"][0]) / (2 * h)),
        "value_converged": [bool(a["value_converged"][0]), bool(b["value_converged"][0])],
        "fully_converged": [bool(a["fully_converged"][0]), bool(b["fully_converged"][0])],
    })
result = {
    "case": case["id"],
    "time": float(t[0]),
    "tol": tol,
    "gradient_mu": float(gradient["mu"][0]),
    "gradient_drho": float(gradient["grad_mu"][0, column]),
    "gradient_valid": bool(gradient["gradient_valid"][0, column]),
    "value_converged": bool(gradient["value_converged"][0]),
    "support_certified": bool(gradient["support_certified"][0]),
    "fd": rows,
}
out = Path(__file__).with_name("case12_fd.json")
out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
