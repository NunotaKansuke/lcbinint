from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import sys
import time
import warnings
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

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCES = [
    Path("/rogue1_8/nunota/microjax/src"),
    Path("/rogue1_8/nunota/microlux/src"),
    Path("/rogue1_8/nunota/moasarc-ecosystem/integration/autodiff-vbm/src"),
    HERE / "vbm561",
]
for source in reversed(SOURCES):
    if not source.is_dir():
        raise RuntimeError(f"missing pinned source path: {source}")
    sys.path.insert(0, str(source))
sys.path.insert(0, str(ROOT / "python"))

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from VBMicrolensing import VBMicrolensing as VBM
import microlux.model as microlux_model
from autodiff_vbm import AutodiffVBM
from microjax.inverse_ray.lightcurve import mag_binary as microjax_mag_binary

lc_source = ROOT / "python/lcbinint"
lc_build = ROOT / "build-algebraic-boundary/python-api/lcbinint"
if not any(lc_build.glob("_lcbinint*.so")):
    raise RuntimeError(f"missing built lcbinint extension under {lc_build}")
spec = importlib.util.spec_from_file_location(
    "lcbinint", lc_source / "__init__.py",
    submodule_search_locations=[str(lc_source), str(lc_build)],
)
if spec is None or spec.loader is None:
    raise RuntimeError("could not load the in-tree lcbinint API")
lcbinint = importlib.util.module_from_spec(spec)
sys.modules["lcbinint"] = lcbinint
spec.loader.exec_module(lcbinint)

OUT = Path(os.environ.get(
    "LCBININT_ISSUE_AUDIT_OUT", str(HERE / "issue4_tolerance_audit_20261017")
))
TOLERANCES = (1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6)
SOURCE_INPUT = HERE / "corrected_value_gradient_12cases" / "inputs.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    case = next(
        item for item in json.loads(SOURCE_INPUT.read_text())["cases"]
        if item["id"] == "near_limb_contact_117"
    )
    all_times = np.asarray(case["times"], dtype=np.float64)
    selected = np.asarray([6, 7, 8, 19, 20, 21], dtype=np.int64)
    times = all_times[selected]
    params = {key: float(value) for key, value in case["params"].items()}
    limb = float(case["limb"])
    p = np.asarray([
        np.log(params["s"]), np.log(params["q"]), params["u0"],
        params["alpha"], np.log(params["rho"]), np.log(params["tE"]),
        params["t0"],
    ], dtype=np.float64)
    p_standard = p.copy()
    p_standard[3] += np.pi
    p_jax = jnp.asarray(p)
    p_standard_jax = jnp.asarray(p_standard)
    t_jax = jnp.asarray(times)
    lc_params = {
        "t0": params["t0"], "u0": params["u0"], "tE": params["tE"],
        "alpha": params["alpha"], "rho": params["rho"],
        "q": params["q"], "s": params["s"],
    }
    arrays: dict[str, np.ndarray] = {"times": times, "source_indices": selected}
    convergence: dict[str, dict] = {}
    for tol in TOLERANCES:
        key = f"{tol:.0e}"
        print("TOLERANCE", key, flush=True)

        vbm = VBM()
        vbm.Tol = vbm.RelTol = tol
        vbm.a1 = limb
        vbm.a2 = 0.0
        start = time.perf_counter()
        vbm_values = np.asarray(
            vbm.BinaryLightCurve(p.tolist(), times)[0], dtype=np.float64
        ).reshape(-1)
        vbm_ms = (time.perf_counter() - start) * 1.0e3
        arrays[f"vbm561__tol_{key}"] = vbm_values
        convergence.setdefault("vbm561", {})[key] = {
            "finite_epochs": int(np.isfinite(vbm_values).sum()),
            "elapsed_ms": vbm_ms,
        }
        print("MEASURED vbm561", key, "finite", int(np.isfinite(vbm_values).sum()),
              "ms", round(vbm_ms, 3), flush=True)

        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            microlux_values = microlux_model.binary_mag(
                params["t0"], params["u0"], params["tE"], params["rho"],
                params["q"], params["s"], float((params["alpha"] + np.pi) * 180.0 / np.pi),
                t_jax, tol=tol, retol=tol, limb_darkening_coeff=limb,
            )
        microlux_values.block_until_ready()
        microlux_values = np.asarray(microlux_values, dtype=np.float64).reshape(-1)
        microlux_ms = (time.perf_counter() - start) * 1.0e3
        arrays[f"microlux__tol_{key}"] = microlux_values
        convergence.setdefault("microlux", {})[key] = {
            "finite_epochs": int(np.isfinite(microlux_values).sum()),
            "elapsed_ms_including_compile": microlux_ms,
            "warnings": [str(item.message) for item in caught],
        }
        print("MEASURED microlux", key, "finite", int(np.isfinite(microlux_values).sum()),
              "ms", round(microlux_ms, 3), flush=True)

        curve_options = lcbinint.Options(
            coordinates="vbm", nbin="auto", finite_source_tol=tol,
            finite_source_reltol=tol,
        )
        curve = lcbinint.LightCurve(
            options=curve_options,
            limb_darkening=lcbinint.LimbDarkening.linear(limb),
        )
        if curve.options.finite_source_backend != "algebraic_boundary":
            raise RuntimeError("current LightCurve default did not resolve to algebraic_boundary")
        start = time.perf_counter()
        lc_result = curve(times, lc_params, with_gradient=False)
        lcbinint_values = np.asarray(lc_result.magnifications, dtype=np.float64).reshape(-1)
        lc_ms = (time.perf_counter() - start) * 1.0e3
        arrays[f"lcbinint__tol_{key}"] = lcbinint_values
        statuses = np.asarray(lc_result.status, dtype=np.int8)
        convergence.setdefault("lcbinint", {})[key] = {
            "finite_epochs": int(np.isfinite(lcbinint_values).sum()),
            "status_counts": {
                str(int(code)): int(count)
                for code, count in zip(*np.unique(statuses, return_counts=True))
            },
            "elapsed_ms_including_first_call": lc_ms,
        }
        arrays[f"lcbinint_status__tol_{key}"] = statuses
        if isinstance(lc_result.diagnostics, dict):
            method = lc_result.diagnostics.get("finite_source_method_name")
            if method is not None:
                arrays[f"lcbinint_method__tol_{key}"] = np.asarray(method, dtype="U32")
        print("MEASURED lcbinint", key, "finite", int(np.isfinite(lcbinint_values).sum()),
              "status", convergence["lcbinint"][key]["status_counts"],
              "ms", round(lc_ms, 3), flush=True)

        autodiff = AutodiffVBM()
        autodiff.Tol = autodiff.RelTol = tol
        start = time.perf_counter()
        autodiff_values = autodiff.BinaryLightCurve(
            p_jax, t_jax, a1=limb
        )[0]
        autodiff_values.block_until_ready()
        autodiff_values = np.asarray(autodiff_values, dtype=np.float64).reshape(-1)
        autodiff_ms = (time.perf_counter() - start) * 1.0e3
        arrays[f"autodiff_vbm__tol_{key}"] = autodiff_values
        convergence.setdefault("autodiff_vbm_vbm55", {})[key] = {
            "finite_epochs": int(np.isfinite(autodiff_values).sum()),
            "elapsed_ms_including_compile": autodiff_ms,
        }
        print("MEASURED autodiff_vbm", key,
              "finite", int(np.isfinite(autodiff_values).sum()),
              "ms", round(autodiff_ms, 3), flush=True)

        # The CPU microJAX API does not expose a tolerance knob; keep its
        # default result as a fixed-setting point of comparison.
        if key == f"{TOLERANCES[0]:.0e}":
            alpha_standard = params["alpha"] + np.pi
            tau = (times - params["t0"]) / params["tE"]
            microjax_values = microjax_mag_binary(
                jnp.asarray(
                    -params["u0"] * np.sin(alpha_standard) + tau * np.cos(alpha_standard)
                    + 1j * (params["u0"] * np.cos(alpha_standard) + tau * np.sin(alpha_standard))
                ),
                params["rho"], s=params["s"], q=params["q"], u1=limb,
                backend="cpu",
            )
            microjax_values.block_until_ready()
            arrays["microjax__fixed_cpu_settings"] = np.asarray(
                microjax_values, dtype=np.float64
            ).reshape(-1)

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "raw.npz", **arrays)
    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(), "platform": platform.platform(),
        "python": sys.version, "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "case_id": "near_limb_contact_117", "case_source": str(SOURCE_INPUT),
        "input_sha256": sha256(SOURCE_INPUT), "selected_source_indices": selected.tolist(),
        "selected_times": times.tolist(), "params": params, "limb": limb,
        "tolerances": list(TOLERANCES), "convergence": convergence,
        "libraries": {
            "vbm561": {
                "version": importlib.import_module("VBMicrolensing").__version__,
                "module": str(Path(importlib.import_module("VBMicrolensing").__file__).resolve()),
                "native": str(Path(importlib.import_module("VBMicrolensing.VBMicrolensing").__file__).resolve()),
                "native_sha256": sha256(Path(importlib.import_module("VBMicrolensing.VBMicrolensing").__file__).resolve()),
            },
            "microlux": {"module": str(Path(microlux_model.__file__).resolve())},
            "autodiff_vbm": {
                "version": importlib.import_module("autodiff_vbm").__version__,
                "module": str(Path(importlib.import_module("autodiff_vbm").__file__).resolve()),
                "vendored_vbm": "5.5",
            },
            "microjax": {
                "version": importlib.import_module("microjax").__version__,
                "module": str(Path(importlib.import_module("microjax").__file__).resolve()),
                "backend": "cpu; fixed public settings",
            },
            "lcbinint": {
                "branch": "experiment/algebraic_boundary",
                "head": "40e66e510799f9d3d15b459be2b38694daad52ee",
                "native_sha256": sha256(Path(lcbinint._lcbinint.__file__).resolve()),
                "backend": "algebraic_boundary via default for static binary LightCurve",
            },
        },
        "warnings": [
            "This tolerance ladder is a convergence probe, not a certified truth reference.",
            "autodiff-vbm shares the VBM 5.5 solver family and is not independent of VBM.",
            "microJAX has no public CPU tolerance argument; it is shown at fixed settings.",
            "Selected epochs are the three near-contact samples on both sides of t=0.",
        ],
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print("SAVED", OUT / "raw.npz", flush=True)


if __name__ == "__main__":
    main()
