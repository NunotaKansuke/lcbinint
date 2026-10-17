#!/usr/bin/env python3
"""Matched limb-darkening timing and scalar-vs-vector Jacobian audit."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

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
ROOT = HERE
OUT = HERE / "issue6_ld_scalar_output_audit_20261017"
REPETITIONS = 15
TOL = 1.0e-3
LD_CASES = ("large_source_uniform", "large_source_ld05_user")
SCALAR_CASE = "near_limb_contact_117"
CONTACT_EPOCH = -1.670997233467463

SOURCES = [
    Path("/rogue1_8/nunota/microjax/src"),
    Path("/rogue1_8/nunota/microlux/src"),
    Path("/rogue1_8/nunota/moasarc-ecosystem/integration/autodiff-vbm/src"),
]
for source in reversed(SOURCES):
    if not source.is_dir():
        raise RuntimeError(f"missing pinned source path: {source}")
    sys.path.insert(0, str(source))

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
import microlux.model as microlux_model
from autodiff_vbm import AutodiffVBM


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def block(value) -> None:
    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def call_timed(fn):
    start = time.perf_counter()
    result = fn()
    block(result)
    return result, (time.perf_counter() - start) * 1.0e3


def paired_times(first_fn, second_fn):
    first_result, first_ms = call_timed(first_fn)
    second_result, second_ms = call_timed(second_fn)
    samples = {"first": [], "second": []}
    for i in range(REPETITIONS):
        order = (("first", first_fn), ("second", second_fn))
        if i % 2:
            order = tuple(reversed(order))
        for label, fn in order:
            _, elapsed = call_timed(fn)
            samples[label].append(elapsed)
    return {
        "first_call_ms": {"first": first_ms, "second": second_ms},
        "samples_ms": samples,
        "median_ms": {
            key: float(np.median(values)) for key, values in samples.items()
        },
        "p95_ms": {
            key: float(np.percentile(values, 95)) for key, values in samples.items()
        },
        "second_over_first": float(
            np.median(samples["second"]) / np.median(samples["first"])
        ),
        "warmup_first": np.asarray(first_result).tolist(),
        "warmup_second": [np.asarray(x).tolist() for x in second_result]
        if isinstance(second_result, tuple)
        else np.asarray(second_result).tolist(),
    }


def params_vector(params: dict[str, float]) -> np.ndarray:
    return np.asarray([
        np.log(params["s"]), np.log(params["q"]), params["u0"],
        params["alpha"], np.log(params["rho"]), np.log(params["tE"]),
        params["t0"],
    ], dtype=np.float64)


def autodiff_curve(ad, p, times, limb):
    return ad.BinaryLightCurve(p, times, a1=limb)[0]


def microlux_curve(p, times, limb):
    return microlux_model.binary_mag(
        p[6], p[2], jnp.exp(p[5]), jnp.exp(p[4]), jnp.exp(p[1]),
        jnp.exp(p[0]), (p[3] + jnp.pi) * (180.0 / jnp.pi), times,
        tol=TOL, retol=TOL, limb_darkening_coeff=limb, analytic=True,
    )


def run_ld_pair(cases: dict[str, dict], ad: AutodiffVBM) -> dict:
    plain, dark = (cases[key] for key in LD_CASES)
    plain_times = np.asarray(plain["times"], dtype=np.float64)
    dark_times = np.asarray(dark["times"], dtype=np.float64)
    if plain["params"] != dark["params"] or not np.allclose(
        plain_times, dark_times, rtol=0.0, atol=1.0e-15
    ):
        raise ValueError("limb-darkening control cases do not have matched inputs")

    p = jnp.asarray(params_vector(plain["params"]))
    times = jnp.asarray(plain_times)
    results = {}
    raw = {"times": plain_times}
    for profile, limb in (("uniform", None), ("linear_ld_0p5", 0.5)):
        curve = lambda pp, tt, ll, _limb=limb: autodiff_curve(ad, pp, tt, _limb)

        def with_aux(pp, tt, ll):
            value = curve(pp, tt, ll)
            return value, value

        value_fn = jax.jit(curve, static_argnums=(2,))
        grad_fn = jax.jit(
            jax.jacfwd(with_aux, argnums=0, has_aux=True), static_argnums=(2,)
        )
        measured = paired_times(
            lambda: value_fn(p, times, limb),
            lambda: grad_fn(p, times, limb),
        )
        values = np.asarray(value_fn(p, times, limb), dtype=np.float64).reshape(-1)
        gradient, aux = grad_fn(p, times, limb)
        gradient = np.asarray(gradient, dtype=np.float64).reshape(times.size, 7)
        aux = np.asarray(aux, dtype=np.float64).reshape(-1)
        measured["output_shapes"] = {
            "value": list(values.shape),
            "full_jacobian": list(gradient.shape),
        }
        measured["value_consistency_max_abs"] = float(np.max(np.abs(values - aux)))
        measured["value_consistency_max_rel"] = float(
            np.max(np.abs(values - aux) / np.maximum(np.abs(values), 1e-300))
        )
        measured["gradient_finite"] = bool(np.all(np.isfinite(gradient)))
        results[profile] = measured
        raw[f"value__{profile}"] = values
        raw[f"value_plus_gradient__{profile}"] = aux
        raw[f"jacfwd__{profile}"] = gradient
    np.savez_compressed(OUT / "matched_ld_raw.npz", **raw)
    return {
        "case_ids": list(LD_CASES),
        "same_parameters": True,
        "original_epoch_arrays_equal_bitwise": bool(np.array_equal(plain_times, dark_times)),
        "original_epoch_arrays_max_abs_difference": float(np.max(np.abs(plain_times - dark_times))),
        "benchmark_uses_one_shared_epoch_array": True,
        "shared_epoch_array_source": LD_CASES[0],
        "n_epochs": int(times.size),
        "params": plain["params"],
        "profiles": {"uniform": None, "linear_ld_0p5": 0.5},
        "value_output_shape": [int(times.size)],
        "gradient_output_shape": [int(times.size), 7],
        "method": "15 interleaved pairs of value-only and value-plus-full-jacfwd calls; x64 CPU; block_until_ready",
        "results": results,
    }


def run_scalar_modes(case: dict, ad: AutodiffVBM, previous: dict) -> dict:
    params = {key: float(value) for key, value in case["params"].items()}
    times = np.asarray(case["times"], dtype=np.float64)
    prior_raw_path = (
        HERE / "issue6_7_gradient_api_audit_20261017" / "raw"
        / "issue7_near_limb_contact_117.npz"
    )
    with np.load(prior_raw_path) as prior_raw:
        prior_times = np.asarray(prior_raw["times"], dtype=np.float64)
    if not np.array_equal(times, prior_times):
        raise ValueError("scalar test epochs differ from the previous vector benchmark")
    index = int(np.argmin(np.abs(times - CONTACT_EPOCH)))
    selected_time = float(times[index])
    one_time = jnp.asarray([selected_time], dtype=jnp.float64)
    limb_value = float(case["limb"])
    limb = limb_value if limb_value > 0 else None
    p = jnp.asarray(params_vector(params))

    prior_case = previous["issue7"][SCALAR_CASE]
    results = {}
    for library in ("autodiff_vbm", "microlux"):
        if library == "autodiff_vbm":
            scalar_fn = lambda pp: autodiff_curve(ad, pp, one_time, limb)[0]
        else:
            scalar_fn = lambda pp: microlux_curve(pp, one_time, limb)[0]
        fwd = jax.jit(jax.jacfwd(scalar_fn))
        rev = jax.jit(jax.jacrev(scalar_fn))
        timing = paired_times(lambda: fwd(p), lambda: rev(p))
        fwd_grad = np.asarray(fwd(p), dtype=np.float64)
        rev_grad = np.asarray(rev(p), dtype=np.float64)
        timing["gradient_output_shapes"] = {
            "jacfwd": list(fwd_grad.shape),
            "jacrev": list(rev_grad.shape),
        }
        timing["jacobian_parity"] = {
            "max_abs": float(np.max(np.abs(fwd_grad - rev_grad))),
            "max_rel_nonzero": float(np.max(
                np.abs(fwd_grad - rev_grad) / np.maximum(np.abs(fwd_grad), 1e-12)
            )),
        }
        timing["jacrev_over_jacfwd"] = timing["second_over_first"]
        old = prior_case[library]
        timing["matched_vector_output_reference"] = {
            "output_shape": [int(times.size)],
            "jacobian_shape": [int(times.size), 7],
            "jacfwd_median_ms": old["jacfwd"]["median_ms"],
            "jacrev_median_ms": old["jacrev"]["median_ms"],
            "jacrev_over_jacfwd": old["parity"]["jacrev_over_jacfwd_median"],
            "source": "issue6_7_gradient_api_audit_20261017/metadata.json",
        }
        results[library] = timing
    return {
        "case_id": SCALAR_CASE,
        "selected_epoch_index": index,
        "selected_time": selected_time,
        "requested_contact_time": CONTACT_EPOCH,
        "limb_darkening_coefficient": limb,
        "input_parameter_count": 7,
        "underlying_public_curve_output_shape": [1],
        "selected_output_index": 0,
        "function_output_shape": [],
        "gradient_output_shape": [7],
        "same_epochs_as_previous_vector_benchmark": True,
        "method": "each public vector light-curve call receives one epoch; the wrapper returns mag[0] as a scalar; 15 alternating jacfwd/jacrev pairs; x64 CPU; block_until_ready",
        "results": results,
    }


def main() -> None:
    input_data = json.loads((ROOT / "corrected_value_gradient_12cases" / "inputs.json").read_text())
    cases = {case["id"]: case for case in input_data["cases"]}
    by_id = cases
    plain, dark = (by_id[key] for key in LD_CASES)
    if plain["params"] != dark["params"] or not np.allclose(
        plain["times"], dark["times"], rtol=0.0, atol=1.0e-15
    ):
        raise ValueError("saved limb-darkening control cases are not matched")
    ad = AutodiffVBM()
    ad.Tol = ad.RelTol = TOL
    previous = json.loads((OUT.parent / "issue6_7_gradient_api_audit_20261017" / "metadata.json").read_text())
    scalar_case = by_id[SCALAR_CASE]

    OUT.mkdir(parents=True, exist_ok=False)

    result = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "jax": jax.__version__,
        "jax_backend": jax.default_backend(),
        "x64": bool(jax.config.read("jax_enable_x64")),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "tolerance": TOL,
        "repetitions": REPETITIONS,
        "library_routes": {
            "autodiff_vbm": "BinaryLightCurve; a1=None for uniform, a1=0.5 for LD",
            "microlux": "binary_mag(..., analytic=True, limb_darkening_coeff=0.5)",
        },
        "matched_limb_darkening_value_gradient": run_ld_pair(by_id, ad),
        "scalar_output_forward_reverse": run_scalar_modes(scalar_case, ad, previous),
        "provenance": {
            "inputs": "corrected_value_gradient_12cases/inputs.json plus the existing user large-source case and near_limb_contact_117 epoch",
            "previous_vector_comparison": "issue6_7_gradient_api_audit_20261017/metadata.json",
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256(Path(__file__).resolve()),
            "autodiff_vbm_revision": "69002603035eaddcdbad3d3b62e4f570a7cf9b85",
            "microlux_revision": "711245b6f557e5b1548b8e27d85ced7384c13418",
        },
    }
    (OUT / "metadata.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (OUT / "inputs.json").write_text(json.dumps({
        "ld_pair": [by_id[key] for key in LD_CASES],
        "scalar_case": scalar_case,
        "scalar_time_index": result["scalar_output_forward_reverse"]["selected_epoch_index"],
        "scalar_time": result["scalar_output_forward_reverse"]["selected_time"],
    }, indent=2, allow_nan=False) + "\n")
    print(f"saved {OUT / 'metadata.json'}", flush=True)
    for profile, measured in result["matched_limb_darkening_value_gradient"]["results"].items():
        print("LD_PROFILE", profile, measured["median_ms"], measured["second_over_first"], flush=True)
    for library, measured in result["scalar_output_forward_reverse"]["results"].items():
        print("SCALAR_MODE", library, measured["median_ms"], measured["jacrev_over_jacfwd"], flush=True)


if __name__ == "__main__":
    main()
