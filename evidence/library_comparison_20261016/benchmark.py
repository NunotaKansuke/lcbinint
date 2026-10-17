from __future__ import annotations

import hashlib
import importlib
import importlib.metadata as metadata
import json
import os
import platform
import subprocess
import sys
import time
import warnings
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

# Ignore PEP-660 finders so explicit source/build paths below determine imports.
sys.meta_path = [finder for finder in sys.meta_path if "_editable" not in type(finder).__module__]

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_leaves

jax.config.update("jax_enable_x64", True)

import microlux.model as microlux_model
from autodiff_vbm import AutodiffVBM
from microjax.inverse_ray.lightcurve import mag_binary as microjax_mag_binary
from VBMicrolensing import VBMicrolensing as VBM
from lcbinint import LightCurve, LimbDarkening, Options
import lcbinint._lcbinint as lcbinint_native
import autodiff_vbm._autodiff_vbm_native as autodiff_vbm_native
import _lcbinint_holonomic_fused as lcbinint_gradient_native


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("LCBININT_BENCHMARK_OUT", Path(__file__).resolve().parent))
EVIDENCE = ROOT / "evidence/holonomic/additional_bug_perf_20261016/post_axis_skip"
PERF_CASES = ROOT / "evidence/holonomic/additional_bug_perf_20261016/post_patch/performance_cases.json"
TOL = 1.0e-3
REPETITIONS = 3
PARAM_ORDER = ["log_s", "log_q", "u0", "alpha_rad", "log_rho", "log_tE", "t0"]


def digest(path: str | Path) -> str | None:
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def git_value(path: str | Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def json_write(path: Path, value: object) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def finite_number(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def load_cases() -> list[dict]:
    performance = json.loads(PERF_CASES.read_text())
    cases: list[dict] = []
    for config in performance:
        archive = np.load(EVIDENCE / f"{config['id']}_performance.npz", allow_pickle=False)
        cases.append({
            "id": config["id"],
            "params": {key: float(value) for key, value in config["params"].items()},
            "limb": float(config["limb"]),
            "times": np.asarray(archive["times"], dtype=np.float64),
            "source": f"existing:{EVIDENCE.name}/{config['id']}_performance.npz",
        })

    extra = {
        "id": "large_source_ld05_user",
        "params": {
            "s": 0.6, "q": 1.0e-3, "rho": 0.5, "t0": 0.0,
            "tE": 10.0, "u0": -1.0e-5, "alpha": 0.5,
        },
        "limb": 0.5,
        "times": np.linspace(-0.8, 0.8, 401, dtype=np.float64),
        "source": "user-provided:401-point curve, 2026-10-16",
    }
    cases.append(extra)

    stress_configs = json.loads((EVIDENCE / "cases.json").read_text())
    selected = [
        "near_limb_contact_117",
        "near_double_root",
        "stratified_00",
        "stratified_02",
        "stratified_09",
    ]
    by_id = {config["id"]: config for config in stress_configs}
    for identifier in selected:
        config = by_id[identifier]
        archive = np.load(EVIDENCE / f"{identifier}.npz", allow_pickle=False)
        cases.append({
            "id": identifier,
            "params": {key: float(value) for key, value in config["params"].items()},
            "limb": float(config["limb"]),
            "times": np.asarray(archive["times"], dtype=np.float64),
            "source": f"existing:{EVIDENCE.name}/{identifier}.npz",
        })
    return cases


def parameter_vector(params: dict[str, float]) -> np.ndarray:
    return np.asarray([
        np.log(params["s"]), np.log(params["q"]), params["u0"], params["alpha"],
        np.log(params["rho"]), np.log(params["tE"]), params["t0"],
    ], dtype=np.float64)


def block_tree(value: object) -> None:
    for leaf in tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def measure(fn, repetitions: int = REPETITIONS) -> tuple[np.ndarray, dict]:
    start = time.perf_counter()
    output = fn()
    block_tree(output)
    first_ms = (time.perf_counter() - start) * 1.0e3
    samples = []
    for _ in range(repetitions):
        start = time.perf_counter()
        output = fn()
        block_tree(output)
        samples.append((time.perf_counter() - start) * 1.0e3)
    return output, {
        "first_call_ms": first_ms,
        "steady_median_ms": float(np.median(samples)),
        "steady_p95_ms": float(np.percentile(samples, 95)),
        "repetitions": repetitions,
    }


def get_value_consensus(values: dict[str, np.ndarray], n: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    missing = np.full(n, np.nan)
    vbm_members = [values.get(name, missing) for name in ("vbm", "autodiff_vbm")]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        vbm_family = np.nanmedian(np.stack(vbm_members), axis=0)
    groups = {
        "microlux": values.get("microlux", missing),
        "microjax": values.get("microjax", missing),
        "lcbinint": values.get("lcbinint", missing),
        "vbm_family": vbm_family,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        consensus = np.nanmedian(np.stack(list(groups.values())), axis=0)
    return consensus, groups


def value_deviation(values: np.ndarray, consensus: np.ndarray) -> dict:
    mask = np.isfinite(values) & np.isfinite(consensus)
    if not np.any(mask):
        return {"finite_epochs": 0, "median_abs": None, "p95_abs": None,
                "max_abs": None, "median_rel": None, "p95_rel": None}
    delta = np.abs(values[mask] - consensus[mask])
    relative = delta / np.maximum(np.abs(consensus[mask]), 1.0e-12)
    return {
        "finite_epochs": int(mask.sum()),
        "median_abs": finite_number(np.median(delta)),
        "p95_abs": finite_number(np.percentile(delta, 95)),
        "max_abs": finite_number(np.max(delta)),
        "median_rel": finite_number(np.median(relative)),
        "p95_rel": finite_number(np.percentile(relative, 95)),
    }


def value_consistency(values_only: np.ndarray, values_with_gradient: np.ndarray) -> dict:
    mask = np.isfinite(values_only) & np.isfinite(values_with_gradient)
    if not np.any(mask):
        return {"finite_epochs": 0, "median_abs": None, "p95_abs": None,
                "max_abs": None, "max_rel": None}
    delta = np.abs(values_only[mask] - values_with_gradient[mask])
    relative = delta / np.maximum(np.abs(values_only[mask]), 1.0e-12)
    return {
        "finite_epochs": int(mask.sum()),
        "median_abs": finite_number(np.median(delta)),
        "p95_abs": finite_number(np.percentile(delta, 95)),
        "max_abs": finite_number(np.max(delta)),
        "max_rel": finite_number(np.max(relative)),
    }


def gradient_consensus(gradients: dict[str, np.ndarray], n: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    missing = np.full((n, 7), np.nan)
    groups = {
        name: gradients.get(name, missing)
        for name in ("microlux", "microjax", "autodiff_vbm", "lcbinint")
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        consensus = np.nanmedian(np.stack(list(groups.values())), axis=0)
    return consensus, groups


def gradient_deviation(values: np.ndarray, consensus: np.ndarray) -> dict:
    result = {"parameters": {}}
    for index, name in enumerate(PARAM_ORDER):
        v = values[:, index]
        c = consensus[:, index]
        mask = np.isfinite(v) & np.isfinite(c)
        if not np.any(mask):
            result["parameters"][name] = {
                "finite_epochs": 0, "median_abs": None, "p95_abs": None,
                "max_abs": None, "median_rel_nonzero": None, "p95_rel_nonzero": None,
            }
            continue
        delta = np.abs(v[mask] - c[mask])
        cabs = np.abs(c[mask])
        nonzero = cabs > 1.0e-8
        relative = delta[nonzero] / cabs[nonzero]
        result["parameters"][name] = {
            "finite_epochs": int(mask.sum()),
            "median_abs": finite_number(np.median(delta)),
            "p95_abs": finite_number(np.percentile(delta, 95)),
            "max_abs": finite_number(np.max(delta)),
            "median_rel_nonzero": finite_number(np.median(relative)) if relative.size else None,
            "p95_rel_nonzero": finite_number(np.percentile(relative, 95)) if relative.size else None,
        }
    return result


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = load_cases()
    case_filter = {
        value for value in os.environ.get("LCBININT_BENCH_CASES", "").split(",") if value
    }
    if case_filter:
        missing = case_filter - {case["id"] for case in cases}
        if missing:
            raise ValueError(f"unknown benchmark cases: {sorted(missing)}")
        cases = [case for case in cases if case["id"] in case_filter]
    skip_libraries = {
        value for value in os.environ.get("LCBININT_SKIP_LIBRARIES", "").split(",") if value
    }
    skip_value_gradient = {
        value for value in os.environ.get("LCBININT_SKIP_VALUE_PLUS_GRADIENT", "").split(",") if value
    }
    skip_reason = os.environ.get("LCBININT_SKIP_REASON", "skipped by configuration")
    print("SELECTED_CASES", [case["id"] for case in cases], flush=True)
    cases_for_hash = [{
        "id": case["id"], "params": case["params"], "limb": case["limb"],
        "times": case["times"].tolist(), "source": case["source"],
    } for case in cases]
    input_hash = hashlib.sha256(
        json.dumps(cases_for_hash, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    script_hash = digest(__file__)
    run_signature = hashlib.sha256(f"{input_hash}:{script_hash}".encode()).hexdigest()
    json_write(OUT / "inputs.json", {
        "input_hash": input_hash, "tolerance": TOL, "parameter_order": PARAM_ORDER,
        "cases": cases_for_hash,
        "case_filter": sorted(case_filter),
        "skipped_libraries": sorted(skip_libraries),
        "value_plus_gradient_skipped_libraries": sorted(skip_value_gradient),
    })

    def microlux_curve(p, t, ld):
        return microlux_model.binary_mag(
            p[6], p[2], jnp.exp(p[5]), jnp.exp(p[4]), jnp.exp(p[1]), jnp.exp(p[0]),
            p[3] * (180.0 / jnp.pi), t,
            tol=TOL, retol=TOL, limb_darkening_coeff=ld,
        )

    def autodiff_curve(p, t, ld):
        return autodiff.BinaryLightCurve(p, t, a1=ld)[0]

    def jacobian_with_value(fn):
        def fn_with_aux(p, t, ld):
            value = fn(p, t, ld)
            return value, value
        return jax.jit(
            jax.jacfwd(fn_with_aux, argnums=0, has_aux=True),
            static_argnums=(2,),
        )

    microlux_value = jax.jit(microlux_curve, static_argnums=(2,))
    microlux_value_gradient = jacobian_with_value(microlux_curve)

    def microjax_curve(p, t, ld):
        tau = (t - p[6]) * jnp.exp(-p[5])
        alpha, u0 = p[3], p[2]
        w = (
            -u0 * jnp.sin(alpha) + tau * jnp.cos(alpha)
            + 1j * (u0 * jnp.cos(alpha) + tau * jnp.sin(alpha))
        )
        return microjax_mag_binary(
            w, jnp.exp(p[4]), s=jnp.exp(p[0]), q=jnp.exp(p[1]),
            u1=ld, backend="cpu",
        )

    microjax_value = jax.jit(microjax_curve, static_argnums=(2,))
    microjax_value_gradient = jacobian_with_value(microjax_curve)

    autodiff = AutodiffVBM()
    autodiff.Tol = autodiff.RelTol = TOL
    autodiff_value = jax.jit(autodiff_curve, static_argnums=(2,))
    autodiff_value_gradient = jacobian_with_value(autodiff_curve)

    metadata_record = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "cpu": platform.processor() or "Intel Xeon Gold 6530 (lscpu)",
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "threads": {key: os.environ.get(key) for key in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "JAX_PLATFORMS", "JAX_ENABLE_X64",
        )},
        "tolerance": {"common": TOL, "microjax": "CPU public backend exposes fixed settings; no tolerance argument"},
        "trajectory_convention": {
            "input_cases": "VBMicrolensing convention: y1=u0*sin(alpha)-tau*cos(alpha), y2=-u0*cos(alpha)-tau*sin(alpha)",
            "microlux_microjax": "add pi to alpha to map the same physical trajectory to their (tau+i*u0)*exp(i*alpha) convention",
            "autodiff_vbm_lcbinint": "use input alpha directly",
        },
        "value_api": {
            "microlux": "microlux.model.binary_mag(times and physical parameters, vector call)",
            "microjax": "microjax.inverse_ray.lightcurve.mag_binary(w_points,...,backend='cpu'); vectorized JAX time-to-w transform in same call",
            "vbm": "VBMicrolensing.VBMicrolensing.BinaryLightCurve(params,times)",
            "autodiff_vbm": "AutodiffVBM.BinaryLightCurve(params,times)",
            "lcbinint": "lcbinint.LightCurve(times,params), algebraic_boundary",
        },
        "value_plus_gradient_api": {
            "microlux": "jax.jacfwd(has_aux=True) of the vector-valued public binary_mag call",
            "microjax": "jax.jacfwd(has_aux=True) of the vector-valued public mag_binary CPU call",
            "autodiff_vbm": "jax.jacfwd(has_aux=True) of the vector-valued public BinaryLightCurve call",
            "lcbinint": "single public holonomic_gradient batch call returning both mu and grad_mu",
            "vbm": None,
        },
        "workload_modes": ["value_only", "value_plus_gradient"],
        "versions": {
            "microlux": {"version": metadata.version("microlux"), "branch": git_value("/rogue1_8/nunota/microlux", "branch", "--show-current"), "head": git_value("/rogue1_8/nunota/microlux", "rev-parse", "HEAD"), "module": str(Path(microlux_model.__file__).resolve()), "module_sha256": digest(microlux_model.__file__)},
            "microjax": {"version": importlib.import_module("microjax").__version__, "distribution": metadata.version("microjaxx"), "branch": git_value("/rogue1_8/nunota/microjax", "branch", "--show-current"), "head": git_value("/rogue1_8/nunota/microjax", "rev-parse", "HEAD"), "module": str(Path(importlib.import_module("microjax").__file__).resolve())},
            "vbm": {"version": importlib.import_module("VBMicrolensing").__version__, "head": git_value("/rogue1_8/nunota/moasarc-ecosystem/forks/VBMicrolensing", "rev-parse", "HEAD"), "module": str(Path(importlib.import_module("VBMicrolensing").__file__).resolve()), "native": str(Path(VBM.__module__ and importlib.import_module("VBMicrolensing.VBMicrolensing").__file__).resolve()) if importlib.util.find_spec("VBMicrolensing.VBMicrolensing") else None},
            "autodiff_vbm": {"version": metadata.version("autodiff-vbm"), "branch": git_value("/rogue1_8/nunota/moasarc-ecosystem/integration/autodiff-vbm", "branch", "--show-current"), "head": git_value("/rogue1_8/nunota/moasarc-ecosystem/integration/autodiff-vbm", "rev-parse", "HEAD"), "module": str(Path(importlib.import_module("autodiff_vbm").__file__).resolve()), "native": str(Path(autodiff_vbm_native.__file__).resolve()), "native_sha256": digest(autodiff_vbm_native.__file__), "vendored_vbm": "5.5"},
            "lcbinint": {"branch": git_value(ROOT, "branch", "--show-current"), "head": git_value(ROOT, "rev-parse", "HEAD"), "native": str(Path(lcbinint_native.__file__).resolve()), "native_sha256": digest(lcbinint_native.__file__), "gradient_native": str(Path(lcbinint_gradient_native.__file__).resolve()), "gradient_native_sha256": digest(lcbinint_gradient_native.__file__), "python_source_sha256": digest(ROOT / "python/lcbinint/__init__.py"), "gradient_source_sha256": digest(ROOT / "python/lcbinint/holonomic_fused.py")},
        },
        "input_hash": input_hash,
        "script_sha256": script_hash,
        "repetitions": REPETITIONS,
        "consensus_is_reference": False,
        "comparison_modes": ["value_only", "value_plus_gradient"],
        "selected_cases": [case["id"] for case in cases],
        "skipped_libraries": sorted(skip_libraries),
        "value_plus_gradient_skipped_libraries": sorted(skip_value_gradient),
        "note": "VBM and autodiff-vbm share the VBM solver family; they are collapsed to one value family for consensus.",
    }
    json_write(OUT / "metadata.json", metadata_record)

    progress_path = OUT / "progress.json"
    if progress_path.exists():
        prior = json.loads(progress_path.read_text())
        if prior.get("run_signature") != run_signature:
            raise RuntimeError("existing progress has a different input/script hash; preserving it")
        progress = prior
    else:
        progress = {"run_signature": run_signature, "input_hash": input_hash, "completed": {}, "errors": {}}

    compiled_graphs: set[str] = set()
    raw_dir = OUT / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for case in cases:
        identifier = case["id"]
        raw_path = raw_dir / f"{identifier}.npz"
        if identifier in progress["completed"] and raw_path.is_file():
            print("SKIP", identifier, flush=True)
            continue

        params = case["params"]
        limb = case["limb"]
        limb_arg = limb if limb > 0.0 else None
        times = case["times"]
        p_numpy = parameter_vector(params)
        p_vbm_jax = jnp.asarray(p_numpy)
        p_standard = p_numpy.copy()
        p_standard[3] += np.pi
        p_standard_jax = jnp.asarray(p_standard)
        t_jax = jnp.asarray(times)
        n = int(times.size)
        ld_tag = "uniform" if limb == 0 else f"{limb:g}"
        tag = f"n={n},ld={ld_tag}"
        arrays: dict[str, np.ndarray] = {"times": times}
        result: dict[str, object] = {
            "n_epochs": n, "limb": limb, "params": params,
            "source": case["source"], "timing": {}, "status": {},
        }
        values: dict[str, np.ndarray] = {}
        gradients: dict[str, np.ndarray] = {}

        if "vbm" in skip_libraries:
            result["status"]["vbm_skipped"] = skip_reason
            print("SKIP", identifier, "vbm", skip_reason, flush=True)
        else:
            # The ordinary VBM API is value-only.
            print("START", identifier, "vbm value_only", flush=True)
            try:
                vbm = VBM()
                vbm.Tol = vbm.RelTol = TOL
                vbm.a1 = limb
                vbm.a2 = 0.0
                value, timing = measure(lambda: vbm.BinaryLightCurve(p_numpy.tolist(), times)[0])
                values["vbm"] = np.asarray(value, dtype=np.float64).reshape(-1)
                arrays["value__vbm"] = values["vbm"]
                result["timing"]["vbm_value"] = timing
                result["status"]["vbm_value_finite"] = int(np.isfinite(values["vbm"]).sum())
                print("MEASURED", identifier, "vbm value_only", timing, flush=True)
            except Exception as error:
                result["status"]["vbm_value_error"] = f"{type(error).__name__}: {error}"
        task_specs = [
            ("microlux", microlux_value, microlux_value_gradient, (p_standard_jax, t_jax, limb_arg)),
            ("microjax", microjax_value, microjax_value_gradient, (p_standard_jax, t_jax, limb)),
            ("autodiff_vbm", autodiff_value, autodiff_value_gradient, (p_vbm_jax, t_jax, limb_arg)),
        ]
        for name, value_fn, value_gradient_fn, args in task_specs:
            if name in skip_libraries:
                result["status"][f"{name}_skipped"] = skip_reason
                print("SKIP", identifier, name, skip_reason, flush=True)
                continue
            print("START", identifier, name, "value_only", flush=True)
            try:
                compile_key = f"{name}:value:{tag}"
                value, timing = measure(lambda fn=value_fn, aa=args: fn(*aa))
                timing["first_call_includes_new_jit_graph"] = compile_key not in compiled_graphs
                compiled_graphs.add(compile_key)
                values[name] = np.asarray(value, dtype=np.float64).reshape(-1)
                arrays[f"value__{name}"] = values[name]
                result["timing"][f"{name}_value"] = timing
                result["status"][f"{name}_value_finite"] = int(np.isfinite(values[name]).sum())
                print("MEASURED", identifier, name, "value_only", timing, flush=True)
            except Exception as error:
                result["status"][f"{name}_value_error"] = f"{type(error).__name__}: {error}"
                print("VALUE_ERROR", identifier, name, result["status"][f"{name}_value_error"], flush=True)
                continue
            if name in skip_value_gradient:
                result["status"][f"{name}_value_plus_gradient_skipped"] = skip_reason
                print("SKIP", identifier, name, "value_plus_gradient", skip_reason, flush=True)
                continue
            print("START", identifier, name, "value_plus_gradient", flush=True)
            try:
                compile_key = f"{name}:value_plus_gradient:{tag}"
                (grad, combined_value), timing = measure(
                    lambda fn=value_gradient_fn, aa=args: fn(*aa)
                )
                timing["first_call_includes_new_jit_graph"] = compile_key not in compiled_graphs
                compiled_graphs.add(compile_key)
                combined_values = np.asarray(combined_value, dtype=np.float64).reshape(-1)
                gradients[name] = np.asarray(grad, dtype=np.float64).reshape(n, 7)
                arrays[f"value_plus_gradient__{name}"] = combined_values
                arrays[f"gradient__{name}"] = gradients[name]
                result["timing"][f"{name}_value_plus_gradient"] = timing
                result["status"][f"{name}_value_plus_gradient_finite"] = int(
                    np.isfinite(combined_values).sum()
                )
                result["status"][f"{name}_gradient_finite"] = int(
                    np.isfinite(gradients[name]).sum()
                )
                print("MEASURED", identifier, name, "value_plus_gradient", timing, flush=True)
            except Exception as error:
                result["status"][f"{name}_value_plus_gradient_error"] = (
                    f"{type(error).__name__}: {error}"
                )
                print("VALUE_PLUS_GRADIENT_ERROR", identifier, name,
                      result["status"][f"{name}_value_plus_gradient_error"], flush=True)

        try:
            curve_kwargs = {
                "options": Options(
                    coordinates="vbm", nbin="auto",
                    finite_source_tol=TOL, finite_source_reltol=TOL,
                    finite_source_backend="algebraic_boundary",
                )
            }
            if limb > 0.0:
                curve_kwargs["limb_darkening"] = LimbDarkening.linear(limb)
            curve = LightCurve(**curve_kwargs)
            lc_params = {
                "t0": params["t0"], "u0": params["u0"], "tE": params["tE"],
                "alpha": params["alpha"], "rho": params["rho"],
                "q": params["q"], "s": params["s"],
            }
            print("START", identifier, "lcbinint value_only", flush=True)
            value, timing = measure(lambda: curve(times, lc_params))
            values["lcbinint"] = np.asarray(value, dtype=np.float64).reshape(-1)
            arrays["value__lcbinint"] = values["lcbinint"]
            result["timing"]["lcbinint_value"] = timing
            result["status"]["lcbinint_value_finite"] = int(np.isfinite(values["lcbinint"]).sum())
            print("MEASURED", identifier, "lcbinint value_only", timing, flush=True)
            print("START", identifier, "lcbinint value_plus_gradient", flush=True)
            grad, timing = measure(lambda: curve.holonomic_gradient(
                times, lc_params, mu_atol=TOL, mu_rtol=TOL
            ))
            combined_values = np.asarray(grad["mu"], dtype=np.float64).reshape(-1)
            arrays["value_plus_gradient__lcbinint"] = combined_values
            result["timing"]["lcbinint_value_plus_gradient"] = timing
            result["status"]["lcbinint_value_plus_gradient_finite"] = int(
                np.isfinite(combined_values).sum()
            )
            raw_gradient = np.asarray(grad["grad_mu"], dtype=np.float64).reshape(n, 7)
            valid = np.asarray(grad["gradient_valid"], dtype=bool)
            if valid.shape == (n,):
                valid = np.broadcast_to(valid[:, None], raw_gradient.shape)
            # lcbinint returns physical derivatives in t0,u0,tE,alpha,rho,q,s order.
            converted = np.column_stack((
                raw_gradient[:, 6] * params["s"],
                raw_gradient[:, 5] * params["q"],
                raw_gradient[:, 1],
                raw_gradient[:, 3],
                raw_gradient[:, 4] * params["rho"],
                raw_gradient[:, 2] * params["tE"],
                raw_gradient[:, 0],
            ))
            gradients["lcbinint"] = converted
            arrays["gradient__lcbinint"] = converted
            arrays["gradient_valid__lcbinint"] = valid
            result["timing"]["lcbinint_value_plus_gradient"]["first_call_includes_new_jit_graph"] = False
            result["status"]["lcbinint_gradient_valid"] = int(valid.sum())
            result["status"]["lcbinint_gradient_components"] = int(valid.size)
            result["status"]["lcbinint_gradient_finite_valid"] = int((np.isfinite(converted) & valid).sum())
        except Exception as error:
            result["status"]["lcbinint_error"] = f"{type(error).__name__}: {error}"
            print("LCBININT_ERROR", identifier, result["status"]["lcbinint_error"], flush=True)

        temp_path = raw_dir / f"{identifier}.tmp.npz"
        np.savez_compressed(temp_path, **arrays)
        temp_path.replace(raw_path)
        progress["completed"][identifier] = result
        json_write(progress_path, progress)
        print("DONE", identifier, n, "epochs", "values", {
            name: result["status"].get(f"{name}_value_finite")
            for name in ("vbm", "microlux", "microjax", "autodiff_vbm", "lcbinint")
        }, "value_plus_gradient_ms", {
            name: result["timing"].get(f"{name}_value_plus_gradient", {}).get("steady_median_ms")
            for name in ("microlux", "microjax", "autodiff_vbm", "lcbinint")
        }, flush=True)

    # Build cohort-median diagnostics from the per-case raw outputs.
    case_summaries = {}
    totals: dict[str, dict[str, float]] = {}
    for case in cases:
        identifier = case["id"]
        raw_path = raw_dir / f"{identifier}.npz"
        if not raw_path.exists() or identifier not in progress["completed"]:
            continue
        with np.load(raw_path, allow_pickle=False) as raw:
            n = len(raw["times"])
            vals = {
                name: np.asarray(raw[f"value__{name}"])
                for name in ("vbm", "microlux", "microjax", "autodiff_vbm", "lcbinint")
                if f"value__{name}" in raw.files
            }
            combined_values = {
                name: np.asarray(raw[f"value_plus_gradient__{name}"])
                for name in ("microlux", "microjax", "autodiff_vbm", "lcbinint")
                if f"value_plus_gradient__{name}" in raw.files
            }
            grads = {
                name: np.asarray(raw[f"gradient__{name}"])
                for name in ("microlux", "microjax", "autodiff_vbm", "lcbinint")
                if f"gradient__{name}" in raw.files
            }
            if "lcbinint" in grads and "gradient_valid__lcbinint" in raw.files:
                mask = np.asarray(raw["gradient_valid__lcbinint"], dtype=bool)
                grads["lcbinint"] = np.where(mask, grads["lcbinint"], np.nan)
            vcons, vgroups = get_value_consensus(vals, n)
            gcons, ggroups = gradient_consensus(grads, n)
            summary = {
                "n_epochs": n,
                "value_consensus": {
                    "definition": "per-epoch median of microlux, microjax, lcbinint, and one VBM-family median (VBM + autodiff-vbm)",
                    "finite_epochs": int(np.isfinite(vcons).sum()),
                    "by_library": {
                        name: value_deviation(vals.get(name, np.full(n, np.nan)), vcons)
                        for name in ("vbm", "microlux", "microjax", "autodiff_vbm", "lcbinint")
                    },
                    "pairwise_family_max_rel": None,
                },
                "value_plus_gradient_consistency": {
                    name: value_consistency(
                        vals.get(name, np.full(n, np.nan)), combined_values.get(name, np.full(n, np.nan))
                    )
                    for name in ("microlux", "microjax", "autodiff_vbm", "lcbinint")
                },
                "gradient_consensus": {
                    "definition": "per-epoch/parameter median of finite gradients; invalid lcbinint components are excluded",
                    "finite_components": int(np.isfinite(gcons).sum()),
                    "by_library": {
                        name: gradient_deviation(
                            np.where(np.isfinite(ggroups[name]), ggroups[name], np.nan), gcons
                        )
                        for name in ("microlux", "microjax", "autodiff_vbm", "lcbinint")
                    },
                },
                "timing": progress["completed"][identifier]["timing"],
                "status": progress["completed"][identifier]["status"],
            }
            # Consensus spread is descriptive, not a truth estimate.
            group_values = np.stack(list(vgroups.values()))
            pair_max = 0.0
            for i in range(group_values.shape[0]):
                for j in range(i + 1, group_values.shape[0]):
                    mask = np.isfinite(group_values[i]) & np.isfinite(group_values[j])
                    if np.any(mask):
                        pair_max = max(pair_max, float(np.max(
                            np.abs(group_values[i, mask] - group_values[j, mask])
                            / np.maximum(np.abs(vcons[mask]), 1.0e-12)
                        )))
            summary["value_consensus"]["pairwise_family_max_rel"] = pair_max
            case_summaries[identifier] = summary
            n_epochs = float(n)
            for name, phase in (
                ("vbm", "value"), ("microlux", "value"), ("microjax", "value"),
                ("autodiff_vbm", "value"), ("lcbinint", "value"),
                ("microlux", "value_plus_gradient"), ("microjax", "value_plus_gradient"),
                ("autodiff_vbm", "value_plus_gradient"), ("lcbinint", "value_plus_gradient"),
            ):
                key = f"{name}_{phase}"
                timing = summary["timing"].get(key)
                if timing:
                    bucket = totals.setdefault(key, {"epochs": 0.0, "median_call_ms_sum": 0.0})
                    bucket["epochs"] += n_epochs
                    bucket["median_call_ms_sum"] += timing["steady_median_ms"]

    overall = {}
    for key, data in totals.items():
        total_ms = data["median_call_ms_sum"]
        overall[key] = {
            "epochs": int(data["epochs"]),
            "sum_of_per_curve_medians_ms": total_ms,
            "weighted_epochs_per_second": data["epochs"] / (total_ms / 1000.0) if total_ms else None,
        }

    final = {
        "input_hash": input_hash,
        "script_sha256": script_hash,
        "metadata": metadata_record,
        "case_count": len(case_summaries),
        "total_epochs": int(sum(summary["n_epochs"] for summary in case_summaries.values())),
        "overall": overall,
        "cases": case_summaries,
        "errors": {
            case_id: result["status"]
            for case_id, result in progress["completed"].items()
            if any("error" in key for key in result["status"])
        },
    }
    json_write(OUT / "summary.json", final)
    print("SUMMARY", OUT / "summary.json", "cases=", len(case_summaries),
          "epochs=", final["total_epochs"], flush=True)


if __name__ == "__main__":
    run()
