"""Matched cold-geometry and warm-geometry router benchmark (no gradients).

Load the explicitly named native build, rather than an installed lcbinint.
Fresh worker threads reset lcbinint's thread-local geometry cache; thread
creation and VBM object construction are outside the timed calls.
"""

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import threading
import time

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
import numpy as np
import VBMicrolensing


def fresh_call(function):
    slot = {}

    def worker():
        try:
            start = time.perf_counter()
            slot["value"] = function()
            slot["seconds"] = time.perf_counter() - start
        except Exception as exc:
            slot["error"] = repr(exc)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    if "error" in slot:
        raise RuntimeError(slot["error"])
    return slot["seconds"], slot["value"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=15)
    args = parser.parse_args()
    sys.path.insert(0, str(args.native_dir.resolve()))
    native = importlib.import_module("_lcbinint")
    report = {
        "native": native.__file__,
        "native_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
        "python": sys.version, "host": platform.node(),
        "vbm_version": VBMicrolensing.__version__,
        "repetitions": args.repetitions, "threads": 1,
        "cold_definition": "fresh OS thread / thread-local geometry; thread creation excluded",
        "timing_api": "_evaluate_holonomic (diagnostics, value calculation only); scalar-only additionally when fully converged",
        "rows": [],
    }
    cases = [
        ("ordinary", 1.2, .01, .3, 0., -1., 1.),
        ("wide_planetary", 1.2, .01, .03, -.08, -.2, .8),
        ("resonant", 1., .1, .02, .7, -.4, .4),
        ("close", .7, .3, .08, 1.1, -.5, .5),
    ]
    for label, s, q, u0, alpha, lo, hi in cases:
        for ld in [0., .5]:
            for tolerance in [1e-3, 1e-4]:
                times = np.linspace(lo, hi, 201)
                params = dict(t0=0., tE=1., u0=u0, alpha=alpha, s=s, q=q, rho=.003)
                curve = native.LightCurve(lens="binary", param_type="vbm", nbin="auto",
                    ld_c=ld, finite_source_tol=tolerance, finite_source_reltol=tolerance)
                vbm = VBMicrolensing.VBMicrolensing()
                vbm.Tol = vbm.RelTol = tolerance
                vbm.a1 = ld
                vp = [math.log(s), math.log(q), u0, alpha, math.log(params["rho"]), 0., 0.]
                vt = times.tolist()
                diagnostic = lambda: curve._evaluate_holonomic(times, params, tolerance, tolerance)
                diagnostic()  # code warmup; this API clears geometry again on every call
                vbm.BinaryLightCurve(vp, vt)
                lc_samples, vb_samples = [], []
                for repetition in range(args.repetitions):
                    order = ("lc", "vbm") if repetition % 2 == 0 else ("vbm", "lc")
                    for route in order:
                        start = time.perf_counter()
                        if route == "lc":
                            values = diagnostic()
                            lc_samples.append(time.perf_counter() - start)
                        else:
                            reference = np.asarray(vbm.BinaryLightCurve(vp, vt)[0])
                            vb_samples.append(time.perf_counter() - start)
                mu = np.asarray(values["mu"])
                methods = list(values["finite_source_method_name"])
                row = dict(case=label, params=params, ld=ld, atol=tolerance, rtol=tolerance,
                    times=times.tolist(), values=mu.tolist(), methods=methods,
                    error_estimate=np.asarray(values["error_estimate"]).tolist(),
                    fully_converged=np.asarray(values["fully_converged"]).tolist(),
                    value_converged=np.asarray(values["value_converged"]).tolist(),
                    support_certified=np.asarray(values["support_certified"]).tolist(),
                    seed_support_proven=np.asarray(values["seed_support_proven"]).tolist(),
                    stop=list(values["stop"]),
                    diagnostic_samples_seconds=lc_samples, vbm_samples_seconds=vb_samples,
                    diagnostic_median_ms=1000*statistics.median(lc_samples),
                    vbm_median_ms=1000*statistics.median(vb_samples),
                    vbm_max_relative_difference=float(np.max(abs(mu-reference)/np.maximum(abs(reference),1e-12))))
                if all(values["fully_converged"]):
                    scalar = lambda: curve._magnification_preplanned(times, params, [8]*201, [1]*201, False)
                    cold = [fresh_call(scalar)[0] for _ in range(args.repetitions)]
                    scalar()
                    warm = []
                    for _ in range(args.repetitions):
                        start = time.perf_counter(); scalar(); warm.append(time.perf_counter()-start)
                    row.update(scalar_cold_samples_seconds=cold, scalar_warm_samples_seconds=warm,
                        scalar_cold_median_ms=1000*statistics.median(cold),
                        scalar_warm_median_ms=1000*statistics.median(warm))
                    if hasattr(curve, "_magnification_holonomic"):
                        cold_value = []
                        for _ in range(args.repetitions):
                            start = time.perf_counter()
                            checked = curve._magnification_holonomic(times, params, tolerance, tolerance)
                            cold_value.append(time.perf_counter()-start)
                        np.testing.assert_allclose(checked, mu, rtol=2e-12, atol=2e-12)
                        row.update(scalar_api_cold_samples_seconds=cold_value,
                            scalar_api_cold_median_ms=1000*statistics.median(cold_value))
                report["rows"].append(row)
                print(label, ld, tolerance, f"diag={row['diagnostic_median_ms']:.3f}ms",
                      f"vbm={row['vbm_median_ms']:.3f}ms",
                      {name: methods.count(name) for name in sorted(set(methods))}, flush=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
