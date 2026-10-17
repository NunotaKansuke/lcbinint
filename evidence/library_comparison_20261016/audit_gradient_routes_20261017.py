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
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
if hasattr(os, "sched_getaffinity"):
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
sys.meta_path = [finder for finder in sys.meta_path if "_editable" not in type(finder).__module__]

HERE = Path(__file__).resolve().parent
INPUTS = HERE / "corrected_value_gradient_12cases" / "inputs.json"
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

OUT = Path(os.environ.get(
    "LCBININT_GRADIENT_AUDIT_OUT", str(HERE / "issue6_7_gradient_api_audit_20261017")
))
TOL = 1.0e-3
REPETITIONS = 15
ISSUE6_CASES = (
    "ordinary_point", "large_source_ld05_user",
    "near_limb_contact_117", "near_double_root",
)
ISSUE7_CASES = ("near_limb_contact_117", "near_double_root")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parameter_vector(params: dict[str, float]) -> np.ndarray:
    return np.asarray([
        np.log(params["s"]), np.log(params["q"]), params["u0"], params["alpha"],
        np.log(params["rho"]), np.log(params["tE"]), params["t0"],
    ], dtype=np.float64)


def block(value) -> None:
    leaves = jax.tree_util.tree_leaves(value)
    for leaf in leaves:
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def call_timed(fn) -> tuple[object, float]:
    start = time.perf_counter()
    result = fn()
    block(result)
    return result, (time.perf_counter() - start) * 1.0e3


def paired_timing(value_fn, gradient_fn) -> dict:
    value_first, value_first_ms = call_timed(value_fn)
    gradient_first, gradient_first_ms = call_timed(gradient_fn)
    value_samples: list[float] = []
    gradient_samples: list[float] = []
    for i in range(REPETITIONS):
        order = (("value", value_fn), ("gradient", gradient_fn))
        if i % 2:
            order = tuple(reversed(order))
        for name, fn in order:
            _, elapsed = call_timed(fn)
            (value_samples if name == "value" else gradient_samples).append(elapsed)
    return {
        "first_call_ms": {
            "value_only": value_first_ms,
            "value_plus_gradient": gradient_first_ms,
        },
        "steady_ms": {
            "value_only": value_samples,
            "value_plus_gradient": gradient_samples,
        },
        "median_ms": {
            "value_only": float(np.median(value_samples)),
            "value_plus_gradient": float(np.median(gradient_samples)),
        },
        "p95_ms": {
            "value_only": float(np.percentile(value_samples, 95)),
            "value_plus_gradient": float(np.percentile(gradient_samples, 95)),
        },
        "combined_over_value_median": float(
            np.median(gradient_samples) / np.median(value_samples)
        ),
        "interleaving": "15 pairs; even pairs value then gradient, odd pairs gradient then value",
        "warmup_values": {
            "value_only": np.asarray(value_first[0] if isinstance(value_first, tuple) else value_first).tolist(),
            "value_plus_gradient": np.asarray(gradient_first[1] if isinstance(gradient_first, tuple) else gradient_first).tolist(),
        },
    }


def main() -> None:
    source = json.loads(INPUTS.read_text())
    by_id = {case["id"]: case for case in source["cases"]}
    by_id["large_source_ld05_user"] = {
        "id": "large_source_ld05_user",
        "params": {
            "s": 0.6, "q": 1.0e-3, "rho": 0.5, "t0": 0.0,
            "tE": 10.0, "u0": -1.0e-5, "alpha": 0.5,
        },
        "limb": 0.5,
        "times": np.linspace(-0.8, 0.8, 401, dtype=np.float64).tolist(),
        "source": "user-provided:401-point curve, 2026-10-16",
    }
    stratified = json.loads((HERE / "stratified_09_lcbinint_only" / "inputs.json").read_text())
    by_id["stratified_09"] = stratified["cases"][0]

    autodiff = AutodiffVBM()
    autodiff.Tol = autodiff.RelTol = TOL

    def autodiff_curve(p, times, limb):
        return autodiff.BinaryLightCurve(p, times, a1=limb)[0]

    def microlux_curve(p, times, limb):
        return microlux_model.binary_mag(
            p[6], p[2], jnp.exp(p[5]), jnp.exp(p[4]), jnp.exp(p[1]),
            jnp.exp(p[0]), (p[3] + jnp.pi) * (180.0 / jnp.pi), times,
            tol=TOL, retol=TOL, limb_darkening_coeff=limb,
            analytic=True,
        )

    functions = {
        "autodiff_vbm": autodiff_curve,
        "microlux": microlux_curve,
    }
    compiled = {}
    for library, fn in functions.items():
        def with_aux(p, times, limb, _fn=fn):
            value = _fn(p, times, limb)
            return value, value
        compiled[library] = {
            "value": jax.jit(fn, static_argnums=(2,)),
            "jacfwd": jax.jit(
                jax.jacfwd(with_aux, argnums=0, has_aux=True), static_argnums=(2,)
            ),
            "jacrev": jax.jit(
                jax.jacrev(with_aux, argnums=0, has_aux=True), static_argnums=(2,)
            ),
        }

    OUTPUTS = OUT / "raw"
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    issue6: dict[str, dict] = {}
    issue7: dict[str, dict] = {}
    for identifier in ISSUE6_CASES:
        case = by_id[identifier]
        params = {key: float(value) for key, value in case["params"].items()}
        limb_value = float(case["limb"])
        limb = limb_value if limb_value > 0 else None
        p = jnp.asarray(parameter_vector(params))
        times = jnp.asarray(np.asarray(case["times"], dtype=np.float64))
        arrays: dict[str, np.ndarray] = {"times": np.asarray(times)}
        case_result: dict[str, dict] = {"source": case.get("source"), "n_epochs": int(times.size)}
        for library in ("autodiff_vbm",):
            value_fn = lambda fn=compiled[library]["value"]: fn(p, times, limb)
            gradient_fn = lambda fn=compiled[library]["jacfwd"]: fn(p, times, limb)
            try:
                paired = paired_timing(value_fn, gradient_fn)
                value, grad_and_aux = paired["warmup_values"]["value_only"], paired["warmup_values"]["value_plus_gradient"]
                # Re-evaluate once after warmup to save raw outputs used in parity checks.
                plain = np.asarray(value_fn(), dtype=np.float64).reshape(-1)
                (grad, combined_value), _ = call_timed(gradient_fn)
                grad = np.asarray(grad, dtype=np.float64).reshape(times.size, 7)
                combined_value = np.asarray(combined_value, dtype=np.float64).reshape(-1)
                delta = np.abs(plain - combined_value)
                rel = delta / np.maximum(np.abs(plain), 1.0e-300)
                arrays[f"value__{library}"] = plain
                arrays[f"value_plus_gradient__{library}"] = combined_value
                arrays[f"jacfwd__{library}"] = grad
                paired["value_consistency"] = {
                    "max_abs": float(np.max(delta)), "max_rel": float(np.max(rel)),
                }
                case_result[library] = paired
            except Exception as error:
                case_result[library] = {"error": f"{type(error).__name__}: {error}"}
        np.savez_compressed(OUTPUTS / f"issue6_{identifier}.npz", **arrays)
        issue6[identifier] = case_result
        print("ISSUE6_DONE", identifier, flush=True)

    for identifier in ISSUE7_CASES:
        case = by_id[identifier]
        params = {key: float(value) for key, value in case["params"].items()}
        limb_value = float(case["limb"])
        limb = limb_value if limb_value > 0 else None
        p = jnp.asarray(parameter_vector(params))
        times = jnp.asarray(np.asarray(case["times"], dtype=np.float64))
        arrays: dict[str, np.ndarray] = {"times": np.asarray(times)}
        case_result: dict[str, dict] = {"source": case.get("source"), "n_epochs": int(times.size)}
        for library in ("autodiff_vbm", "microlux"):
            value_fn = lambda fn=compiled[library]["value"]: fn(p, times, limb)
            outputs = {}
            case_result[library] = {}
            try:
                for mode in ("jacfwd", "jacrev"):
                    grad_fn = lambda fn=compiled[library][mode]: fn(p, times, limb)
                    output, first_ms = call_timed(grad_fn)
                    outputs[mode] = (output, first_ms, grad_fn)
                (grad_fwd, value_fwd), first_fwd = outputs["jacfwd"][0], outputs["jacfwd"][1]
                (grad_rev, value_rev), first_rev = outputs["jacrev"][0], outputs["jacrev"][1]
                grad_fwd = np.asarray(grad_fwd, dtype=np.float64).reshape(times.size, 7)
                grad_rev = np.asarray(grad_rev, dtype=np.float64).reshape(times.size, 7)
                value_fwd = np.asarray(value_fwd, dtype=np.float64).reshape(-1)
                value_rev = np.asarray(value_rev, dtype=np.float64).reshape(-1)
                samples_by_mode = {"jacfwd": [], "jacrev": []}
                for i in range(REPETITIONS):
                    modes = ("jacfwd", "jacrev") if i % 2 == 0 else ("jacrev", "jacfwd")
                    for mode in modes:
                        _, _, fn = outputs[mode]
                        _, elapsed = call_timed(fn)
                        samples_by_mode[mode].append(elapsed)
                for mode in ("jacfwd", "jacrev"):
                    samples = samples_by_mode[mode]
                    case_result[library][mode] = {
                        "first_call_ms": outputs[mode][1],
                        "steady_samples_ms": samples,
                        "median_ms": float(np.median(samples)),
                        "p95_ms": float(np.percentile(samples, 95)),
                    }
                dv = np.abs(value_fwd - value_rev)
                dg = np.abs(grad_fwd - grad_rev)
                case_result.setdefault(library, {})["parity"] = {
                    "value_max_abs": float(np.max(dv)),
                    "value_max_rel": float(np.max(dv / np.maximum(np.abs(value_fwd), 1.0e-300))),
                    "gradient_max_abs": float(np.max(dg)),
                    "gradient_max_rel_nonzero": float(
                        np.max(dg / np.maximum(np.abs(grad_fwd), 1.0e-12))
                    ),
                    "jacrev_over_jacfwd_median": float(
                        np.median(case_result[library]["jacrev"]["steady_samples_ms"])
                        / np.median(case_result[library]["jacfwd"]["steady_samples_ms"])
                    ),
                }
                arrays[f"jacfwd__{library}"] = grad_fwd
                arrays[f"jacrev__{library}"] = grad_rev
                arrays[f"value_jacfwd__{library}"] = value_fwd
                arrays[f"value_jacrev__{library}"] = value_rev
                case_result[library]["errors"] = None
            except Exception as error:
                case_result[library] = {"error": f"{type(error).__name__}: {error}"}
            print("ISSUE7_ROUTE", identifier, library,
                  case_result.get(library, {}).get("parity", case_result.get(library)), flush=True)
        np.savez_compressed(OUTPUTS / f"issue7_{identifier}.npz", **arrays)
        issue7[identifier] = case_result

    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(), "platform": platform.platform(),
        "python": sys.version,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "tolerance": TOL, "x64": True, "repetitions": REPETITIONS,
        "timing_policy": "synchronize every result; first call separate; paired alternating order for issue 6",
        "issue6": issue6,
        "issue7": issue7,
        "routes": {
            "autodiff_vbm": {
                "value": "AutodiffVBM.BinaryLightCurve",
                "jacfwd": "jax.jacfwd over public vector light curve; underlying custom_jvp FFI",
                "jacrev": "jax.jacrev over same public vector light curve",
                "source_revision": "69002603035eaddcdbad3d3b62e4f570a7cf9b85",
            },
            "microlux": {
                "value": "microlux.model.binary_mag, analytic=True",
                "jacfwd": "jax.jacfwd over public vector light curve",
                "jacrev": "jax.jacrev over public vector light curve",
                "source_revision": "711245b6f557e5b1548b8e27d85ced7384c13418",
            },
            "microjax": {
                "supported_route": "jax.jacfwd (CPU public API)",
                "reverse_mode": "not run: project documentation says reverse mode through CPU scheduler is not part of public API",
                "source_revision": "326cb0dd14617e5716a9d652b510709902d27b5b",
            },
            "lcbinint": {
                "supported_route": "LightCurve(..., with_gradient=True), current unified API",
                "backend": "branch default algebraic_boundary for static binary source",
                "source_revision": "40e66e510799f9d3d15b459be2b38694daad52ee",
            },
        },
        "inputs_sha256": sha256(INPUTS),
        "warnings": [
            "The timing audit measures requested APIs and complete Jacobians, not numerical certification.",
            "Reverse mode is tested only where the public implementation advertises/supports it.",
            "MicroJAX forward mode and lcbinint fused timing for stratified_09 are stored by the current API comparison run, not duplicated here.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print("SAVED", OUT / "metadata.json", flush=True)


if __name__ == "__main__":
    main()
