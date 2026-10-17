from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
if hasattr(os, "sched_getaffinity"):
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
sys.meta_path = [finder for finder in sys.meta_path if "_editable" not in type(finder).__module__]

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "python/lcbinint"
BUILD = ROOT / "build-algebraic-boundary/python-api/lcbinint"
sys.path.insert(0, str(ROOT / "python"))
spec = importlib.util.spec_from_file_location(
    "lcbinint", SOURCE / "__init__.py",
    submodule_search_locations=[str(SOURCE), str(BUILD)],
)
package = importlib.util.module_from_spec(spec)
sys.modules["lcbinint"] = package
spec.loader.exec_module(package)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=("algebraic_boundary", "default"),
        default="algebraic_boundary",
    )
    parser.add_argument("--with-gradient", action="store_true")
    args = parser.parse_args()
    inputs = json.loads((HERE / "stratified_09_lcbinint_only" / "inputs.json").read_text())
    case = inputs["cases"][0]
    times = np.asarray(case["times"], dtype=np.float64)
    params = {k: float(v) for k, v in case["params"].items()}
    selected = [12, 13, 14]
    results = {}
    output = HERE / "issue5_fallback_route_20261017"
    output.mkdir(parents=True, exist_ok=True)
    payload = {"case_id": case["id"], "params": params, "results": results}
    for backend in (args.backend,):
        options = package.Options(
            coordinates="vbm", nbin="auto", finite_source_tol=1e-3,
            finite_source_reltol=1e-3, finite_source_backend=backend, jax=False,
        )
        curve = package.LightCurve(options=options)
        started = time.perf_counter()
        result = curve(
            times[selected], params, with_gradient=args.with_gradient
        )
        results[backend] = {
            "resolved_backend": curve.options.finite_source_backend,
            "times": times[selected].tolist(),
            "magnifications": np.asarray(result.magnifications, dtype=np.float64).tolist(),
            "status_names": np.asarray(result.status_names).tolist(),
            "value_converged": np.asarray(result.value_converged).tolist(),
            "gradient_valid": None if result.gradient_valid is None else
                np.asarray(result.gradient_valid).tolist(),
            "gradients_finite": None if result.gradients is None else
                bool(np.isfinite(np.asarray(result.gradients)).all()),
            "elapsed_seconds": time.perf_counter() - started,
            "diagnostics": {
                key: np.asarray(result.diagnostics[key]).tolist()
                for key in ("stop", "finite_source_method_name", "error_estimate",
                            "support_certified", "seed_support_proven",
                            "point_source_magnification", "value_fallback_used",
                            "value_fallback_kind")
                if key in result.diagnostics
            },
        }
        mode = "gradient" if args.with_gradient else "value"
        (output / f"metadata_{backend}_{mode}.json").write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n"
        )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
