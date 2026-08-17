"""Small, direct value-only comparison of the algebraic C++ kernel and microJAX.

This is intentionally not a public API or FFI benchmark.  The C++ timings are
read from ``benchmark_algebraic_boundary`` (which measures the native
experimental kernel after a warm-up); microJAX is timed through the production
one-shot CPU scheduler after JIT warm-up.  VBM is used only as an accuracy
oracle and is outside both timings.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path


CASES = {
    "smooth_planetary": (1.1, 1.0e-3, 0.08, 0.03, 1.0e-2),
    "central_high_mag": (1.1, 1.0e-3, 0.002, 0.0007, 1.0e-3),
    "tiny_source": (1.1, 1.0e-3, 0.0015, -0.0004, 3.0e-4),
    "wide_binary": (2.0, 0.1, 0.10, 0.05, 1.0e-2),
    "equal_resonant": (1.0, 1.0, 0.01, 0.02, 2.0e-2),
}
PROFILES = ("uniform", "linear")


def _load_microjax(source: Path):
    source = source.resolve()
    sys.path.insert(0, str(source))
    from microjax.inverse_ray.cpu import (  # pylint: disable=import-outside-toplevel
        mag_limb_dark_cpu_one_shot,
        mag_uniform_cpu_one_shot,
    )

    return mag_uniform_cpu_one_shot, mag_limb_dark_cpu_one_shot


def _vbm_value(case_name: str, profile: str) -> float:
    import VBMicrolensing  # pylint: disable=import-outside-toplevel

    s, q, x, y, rho = CASES[case_name]
    solver = VBMicrolensing.VBMicrolensing()
    solver.Tol = 1.0e-10
    solver.RelTol = 1.0e-9
    if profile == "uniform":
        return float(solver.BinaryMag(s, q, x, y, rho, 1.0e-10))
    solver.a1 = 0.5
    solver.minannuli = 16
    solver.SetLDprofile(solver.LDlinear)
    return float(solver.BinaryMagDark(s, q, x, y, rho, 1.0e-10))


def _time_micro(function, argument, repeats: int):
    import jax  # pylint: disable=import-outside-toplevel

    compiled_start = time.perf_counter()
    first = function(argument)
    first = jax.block_until_ready(first)
    compile_seconds = time.perf_counter() - compiled_start
    samples = []
    last = first
    for _ in range(repeats):
        started = time.perf_counter()
        last = jax.block_until_ready(function(argument))
        samples.append(time.perf_counter() - started)
    return compile_seconds, statistics.median(samples), last


def _scalar(value) -> float:
    return float(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--native-csv",
        type=Path,
        default=Path("/tmp/algebraic-boundary-core-20260818.csv"),
    )
    parser.add_argument(
        "--microjax-src",
        type=Path,
        default=Path("/tmp/microjax-develop-review.DGK5Hu/src"),
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rtol", type=float, default=1.0e-3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import jax  # pylint: disable=import-outside-toplevel
    import jax.numpy as jnp  # pylint: disable=import-outside-toplevel

    jax.config.update("jax_enable_x64", True)
    mag_uniform_cpu_one_shot, mag_limb_dark_cpu_one_shot = _load_microjax(
        args.microjax_src
    )
    with args.native_csv.open(newline="") as stream:
        native_rows = list(csv.DictReader(
            line for line in stream if not line.startswith("#")
        ))

    native_by_key = {
        (row["case"], row["profile"], float(row["tolerance"])): row
        for row in native_rows
    }
    rows = []
    for profile in PROFILES:
        micro_base = (
            mag_uniform_cpu_one_shot
            if profile == "uniform"
            else mag_limb_dark_cpu_one_shot
        )
        for case_name in CASES:
            s, q, x, y, rho = CASES[case_name]
            argument = jnp.asarray(x + 1.0j * y, dtype=jnp.complex128)
            if profile == "uniform":
                def evaluate(w):
                    return micro_base(w, rho, s=s, q=q, n_limb=64)
            else:
                def evaluate(w):
                    return micro_base(w, rho, s=s, q=q, u1=0.5, n_limb=64)
            jitted = jax.jit(evaluate)
            compile_seconds, micro_seconds, result = _time_micro(
                jitted, argument, max(1, args.repeats)
            )
            native = native_by_key[(case_name, profile, args.rtol)]
            reference = _vbm_value(case_name, profile)
            micro_value = _scalar(result.magnification)
            micro_status = int(result.status)
            rows.append({
                "case": case_name,
                "profile": profile,
                "reference_vbm": reference,
                "native_value": float(native["polar_algebraic"]),
                "microjax_value": micro_value,
                "native_error": float(native["polar_algebraic_error"]),
                "microjax_error": abs(micro_value - reference) / max(abs(reference), 1.0),
                "native_ms": 1.0e3 * float(native["polar_algebraic_seconds"]),
                "microjax_ms": 1.0e3 * micro_seconds,
                "microjax_over_native": micro_seconds / float(native["polar_algebraic_seconds"]),
                "microjax_status": micro_status,
                "microjax_estimated_error": _scalar(result.estimated_error),
                "microjax_compile_ms": 1.0e3 * compile_seconds,
            })

    print("case,profile,native_ms,microjax_ms,microjax_over_native,"
          "native_error,microjax_error,microjax_status")
    for row in rows:
        print(
            f"{row['case']},{row['profile']},{row['native_ms']:.6g},"
            f"{row['microjax_ms']:.6g},{row['microjax_over_native']:.6g},"
            f"{row['native_error']:.6g},{row['microjax_error']:.6g},"
            f"{row['microjax_status']}"
        )
    valid = [row for row in rows if row["microjax_status"] == 0]
    print(
        "# summary,total,valid,median_microjax_over_native,"
        "native_median_ms,microjax_median_ms"
    )
    print(
        f"# summary,{len(rows)},{len(valid)},"
        f"{statistics.median(row['microjax_over_native'] for row in valid):.6g},"
        f"{statistics.median(row['native_ms'] for row in valid):.6g},"
        f"{statistics.median(row['microjax_ms'] for row in valid):.6g}"
    )
    if args.output:
        args.output.write_text(json.dumps({
            "native_csv": str(args.native_csv.resolve()),
            "microjax_source": str(args.microjax_src.resolve()),
            "microjax_rtol": args.rtol,
            "rows": rows,
        }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
