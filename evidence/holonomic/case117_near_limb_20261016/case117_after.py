"""Reproduce case_117 gradients, work counts, and strict-value finite differences."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "evidence/holonomic/lightcurve_matrix_20261015"))
from benchmark import load

np, lc, _ = load()
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
case = json.loads((ROOT / "evidence/holonomic/stress_20261015/case_117.json").read_text())
p = case["params"]
t = -p["rho"] * p["tE"]


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
    result = curve(tol).holonomic_gradient([t], p, mu_atol=tol, mu_rtol=tol)
    order = list(result["param_order"])
    gradient_rows.append({
        "tol": tol,
        "mu": float(result["mu"][0]),
        "value_converged": bool(result["value_converged"][0]),
        "support_certified": bool(result["support_certified"][0]),
        "gradient_valid": [bool(x) for x in result["gradient_valid"][0]],
        "gradient": {key: float(result["grad_mu"][0, order.index(key)])
                     for key in ("t0", "rho", "q")},
    })

# The compiled epoch API exposes the number of radial/angular evaluations.
from _lcbinint_holonomic_fused import BinarySeededFusedGradient
geometry_curve = curve(1e-3)
geometry = geometry_curve.finite_source_geometry([t], p)
q_solver = float(geometry.mass_ratio[0])
evaluator = BinarySeededFusedGradient(p["s"], q_solver, p["rho"], 1400)
work_rows = []
for tol in (1e-3, 1e-4, 1e-5, 1e-6):
    result = evaluator.evaluate(
        float(geometry.source_x[0]), float(geometry.source_y[0]), case["limb"],
        tol, tol, True,
    )
    work_rows.append({
        "tol": tol,
        "node_evaluations": int(result["node_evaluations"]),
        "value_converged": bool(result["value_converged"]),
        "support_certified": bool(result["support_certified"]),
        "raw_gradient": [float(x) for x in result["grad_mu"]],
    })

# Approximate the near-limb gap from the cached caustic polyline. This is a
# geometry diagnostic, not a support certificate.
source_x, source_y = float(geometry.source_x[0]), float(geometry.source_y[0])
rho = float(geometry.source_radius[0])
branches = geometry_curve.caustics(p)
branch_rows = []
for i, (xx, yy) in enumerate(zip(branches.x, branches.y)):
    x, y = np.asarray(xx), np.asarray(yy)
    start = np.column_stack((x[:-1], y[:-1]))
    delta = np.column_stack((x[1:] - x[:-1], y[1:] - y[:-1]))
    offset = np.column_stack((source_x - start[:, 0], source_y - start[:, 1]))
    denom = np.sum(delta * delta, axis=1)
    fraction = np.clip(np.sum(offset * delta, axis=1) / np.maximum(denom, 1e-300), 0, 1)
    distance = np.hypot(offset[:, 0] - fraction * delta[:, 0],
                        offset[:, 1] - fraction * delta[:, 1])
    branch_rows.append({
        "branch": i,
        "min_polyline_distance": float(np.min(distance)),
        "rho_minus_distance": float(rho - np.min(distance)),
        "bbox_diagonal": float(np.hypot(np.ptp(x), np.ptp(y))),
    })

reference_tol = 1e-7
reference = curve(reference_tol)
fd_rows = []
scales = {
    "t0": (p["rho"] * p["tE"], (1e-5, 3e-6, 1e-6, 3e-7, 1e-7)),
    "rho": (p["rho"], (1e-5, 3e-6, 1e-6, 3e-7, 1e-7)),
    "q": (p["q"], (1e-1, 3e-2, 1e-2, 3e-3, 1e-3)),
}
for key, (scale, fractions) in scales.items():
    for fraction in fractions:
        step = scale * fraction
        plus, minus = dict(p), dict(p)
        plus[key] += step
        minus[key] -= step
        a, b = reference.info([t], plus), reference.info([t], minus)
        fd_rows.append({
            "parameter": key,
            "fraction_of_scale": fraction,
            "step": step,
            "fd": float((a["mu"][0] - b["mu"][0]) / (2 * step)),
            "plus_value_converged": bool(a["value_converged"][0]),
            "minus_value_converged": bool(b["value_converged"][0]),
            "plus_support_certified": bool(a["support_certified"][0]),
            "minus_support_certified": bool(b["support_certified"][0]),
        })

native_path = Path(lc._native.__file__).resolve()
gradient_path = Path(__import__("_lcbinint_holonomic_fused").__file__).resolve()
output = {
    "case": case["id"],
    "time": t,
    "parameter_setup": p,
    "limb_darkening": case["limb"],
    "source_geometry": {"x": source_x, "y": source_y, "rho": rho,
                        "q_solver": q_solver,
                        "center_distance_minus_rho": float(np.hypot(source_x, source_y) - rho)},
    "nearest_caustic_branches": branch_rows,
    "baseline_tol_1e-3": {
        "gradient": {"t0": -18.481774648253097,
                     "rho": 140.99008516184267,
                     "q": 33.97349834378293},
        "node_evaluations": 136,
    },
    "gradient_tolerance_sweep": gradient_rows,
    "node_evaluations_after": work_rows,
    "finite_difference_tolerance": reference_tol,
    "finite_differences": fd_rows,
    "native_extension": str(native_path),
    "native_sha256": hashlib.sha256(native_path.read_bytes()).hexdigest(),
    "gradient_extension": str(gradient_path),
    "gradient_sha256": hashlib.sha256(gradient_path.read_bytes()).hexdigest(),
}
out = Path(__file__).resolve().parent / "case117_after.json"
out.write_text(json.dumps(output, indent=2) + "\n")
print(json.dumps(output, indent=2))
