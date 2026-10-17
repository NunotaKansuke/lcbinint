from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
if hasattr(os, "sched_getaffinity"):
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})

import numpy as np
from scipy.integrate import quad

HERE = Path(__file__).resolve().parent
INPUT = HERE / "corrected_value_gradient_12cases" / "inputs.json"
OUT = Path(os.environ.get(
    "LCBININT_FSPL_LIMIT_OUT", str(HERE / "issue4_single_lens_limit_20261017")
))
INDICES = (6, 7, 8, 20)


def fspl_uniform(u: float, rho: float) -> tuple[float, float]:
    """Disk average of the point-lens magnification in lens-centered polar coordinates."""
    def excess_area_density(r: float) -> float:
        if r == 0.0:
            return 1.0
        return (r * r + 2.0) / np.sqrt(r * r + 4.0) - r

    def integrand(r: float) -> float:
        if r == 0.0:
            if abs(u - rho) <= 1.0e-14:
                angle = np.pi
            else:
                angle = 2.0 * np.pi if u < rho else 0.0
            return angle * excess_area_density(r)
        cosine = (r * r + u * u - rho * rho) / (2.0 * r * u)
        if cosine <= -1.0:
            angle = 2.0 * np.pi
        elif cosine >= 1.0:
            angle = 0.0
        else:
            angle = 2.0 * np.arccos(cosine)
        return excess_area_density(r) * angle

    contact = abs(u - rho)
    upper = u + rho
    points = sorted({point for point in (contact, u) if 0.0 < point < upper})
    integral, error = quad(
        integrand, 0.0, upper, points=points or None,
        epsabs=1.0e-13, epsrel=1.0e-13, limit=1000,
    )
    normalization = np.pi * rho * rho
    return 1.0 + integral / normalization, error / normalization


def fspl_linear_ld(u: float, rho: float, coefficient: float) -> tuple[float, float]:
    """Disk average with I(mu)=1-coefficient*(1-mu), integrated numerically."""
    normalization = np.pi * rho * rho * (1.0 - coefficient / 3.0)

    def angular_weight(r: float) -> float:
        if r == 0.0:
            if abs(u - rho) <= 1.0e-14:
                return np.pi * (1.0 - coefficient)
            if u > rho:
                return 0.0
            mu = np.sqrt(max(0.0, 1.0 - (u / rho) ** 2))
            return 2.0 * np.pi * (1.0 - coefficient * (1.0 - mu))
        cosine = (r * r + u * u - rho * rho) / (2.0 * r * u)
        if cosine >= 1.0:
            return 0.0
        theta_max = np.pi if cosine <= -1.0 else np.arccos(cosine)

        def intensity(theta: float) -> float:
            distance_sq = r * r + u * u - 2.0 * r * u * np.cos(theta)
            mu = np.sqrt(max(0.0, 1.0 - distance_sq / (rho * rho)))
            return 1.0 - coefficient * (1.0 - mu)

        angular, _ = quad(
            intensity, 0.0, theta_max,
            epsabs=2.0e-10, epsrel=2.0e-10, limit=100,
        )
        return 2.0 * angular

    def integrand(r: float) -> float:
        if r == 0.0:
            return angular_weight(r)
        excess = (r * r + 2.0) / np.sqrt(r * r + 4.0) - r
        return excess * angular_weight(r)

    contact = abs(u - rho)
    upper = u + rho
    points = sorted({point for point in (contact, u) if 0.0 < point < upper})
    integral, error = quad(
        integrand, 0.0, upper, points=points or None,
        epsabs=1.0e-9, epsrel=1.0e-9, limit=300,
    )
    return 1.0 + integral / normalization, error / normalization


def main() -> None:
    source = json.loads(INPUT.read_text())
    case = next(item for item in source["cases"] if item["id"] == "near_limb_contact_117")
    params = {key: float(value) for key, value in case["params"].items()}
    times = np.asarray(case["times"], dtype=np.float64)[np.asarray(INDICES)]
    tau = (times - params["t0"]) / params["tE"]
    y1 = params["u0"] * np.sin(params["alpha"]) - tau * np.cos(params["alpha"])
    y2 = -params["u0"] * np.cos(params["alpha"]) - tau * np.sin(params["alpha"])
    u = np.hypot(y1, y2)
    uniform = [fspl_uniform(float(value), params["rho"]) for value in u]
    ld = [fspl_linear_ld(float(value), params["rho"], float(case["limb"])) for value in u]
    rows = []
    for i, epoch in enumerate(times):
        rows.append({
            "source_index": INDICES[i], "time": float(epoch), "u": float(u[i]),
            "rho_minus_u": float(params["rho"] - u[i]),
            "q_zero_uniform": uniform[i][0], "uniform_quad_error_estimate": uniform[i][1],
            "q_zero_ld05": ld[i][0], "ld_quad_error_estimate": ld[i][1],
        })
    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(), "python": sys.version,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "method": "independent one-dimensional quadrature of the exact point-lens finite-source integral in the q->0 limit",
        "source_intensity": "uniform or linear I(mu)=1-u1*(1-mu), normalized by disk-integrated flux",
        "case_id": "near_limb_contact_117", "input_sha256": hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        "indices": list(INDICES), "params": params, "limb_coefficient": float(case["limb"]),
        "rows": rows,
        "limitation": "This is an exact point-lens q=0 reference, not an exact binary-lens reference for q=7.886e-7. It tests the smooth finite-source limb-contact limit and order of magnitude; binary perturbations are assessed separately with the independent binary solvers.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / "raw.npz", times=times, u=u,
        q_zero_uniform=np.asarray([row["q_zero_uniform"] for row in rows]),
        q_zero_ld05=np.asarray([row["q_zero_ld05"] for row in rows]),
        uniform_quad_error=np.asarray([row["uniform_quad_error_estimate"] for row in rows]),
        ld_quad_error=np.asarray([row["ld_quad_error_estimate"] for row in rows]),
    )
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
