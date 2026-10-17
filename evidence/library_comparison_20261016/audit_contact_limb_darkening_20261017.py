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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for source in reversed([
    Path("/rogue1_8/nunota/microjax/src"),
    Path("/rogue1_8/nunota/microlux/src"),
    Path("/rogue1_8/nunota/moasarc-ecosystem/integration/autodiff-vbm/src"),
]):
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
spec = importlib.util.spec_from_file_location(
    "lcbinint", lc_source / "__init__.py",
    submodule_search_locations=[str(lc_source), str(lc_build)],
)
if spec is None or spec.loader is None:
    raise RuntimeError("could not load in-tree lcbinint")
lcbinint = importlib.util.module_from_spec(spec)
sys.modules["lcbinint"] = lcbinint
spec.loader.exec_module(lcbinint)

INPUT = HERE / "corrected_value_gradient_12cases" / "inputs.json"
OUT = Path(os.environ.get(
    "LCBININT_CONTACT_LD_OUT", str(HERE / "issue4_contact_ld_audit_20261017")
))
TOL = 1.0e-3


def main() -> None:
    case = next(c for c in json.loads(INPUT.read_text())["cases"] if c["id"] == "near_limb_contact_117")
    indices = np.asarray([6, 7, 8, 20], dtype=np.int64)
    all_times = np.asarray(case["times"], dtype=np.float64)
    times = all_times[indices]
    params = {k: float(v) for k, v in case["params"].items()}
    p = np.asarray([
        np.log(params["s"]), np.log(params["q"]), params["u0"], params["alpha"],
        np.log(params["rho"]), np.log(params["tE"]), params["t0"],
    ], dtype=np.float64)
    p_jax = jnp.asarray(p)
    standard_alpha = params["alpha"] + np.pi
    tau = (times - params["t0"]) / params["tE"]
    w = jnp.asarray(-params["u0"] * np.sin(standard_alpha) + tau * np.cos(standard_alpha)
                    + 1j * (params["u0"] * np.cos(standard_alpha) + tau * np.sin(standard_alpha)))
    lc_params = {
        "t0": params["t0"], "u0": params["u0"], "tE": params["tE"],
        "alpha": params["alpha"], "rho": params["rho"], "q": params["q"], "s": params["s"],
    }
    values: dict[str, np.ndarray] = {"times": times, "source_indices": indices}
    results: dict[str, dict] = {}
    for limb in (0.0, 0.5):
        tag = "uniform" if limb == 0 else "ld05"
        print("LIMB", tag, flush=True)
        vbm = VBM()
        vbm.Tol = vbm.RelTol = TOL
        vbm.a1 = limb
        vbm.a2 = 0.0
        values[f"vbm55__{tag}"] = np.asarray(vbm.BinaryLightCurve(p.tolist(), times)[0], dtype=np.float64)

        autodiff = AutodiffVBM()
        autodiff.Tol = autodiff.RelTol = TOL
        ad = autodiff.BinaryLightCurve(p_jax, jnp.asarray(times), a1=limb)[0]
        ad.block_until_ready()
        values[f"autodiff_vbm55__{tag}"] = np.asarray(ad, dtype=np.float64)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            lux = microlux_model.binary_mag(
                params["t0"], params["u0"], params["tE"], params["rho"], params["q"], params["s"],
                float(standard_alpha * 180.0 / np.pi), jnp.asarray(times),
                tol=TOL, retol=TOL, limb_darkening_coeff=(limb if limb else None),
            )
        lux.block_until_ready()
        values[f"microlux__{tag}"] = np.asarray(lux, dtype=np.float64)
        results.setdefault("microlux_warnings", {})[tag] = [str(x.message) for x in caught]

        microjax = microjax_mag_binary(
            w, params["rho"], s=params["s"], q=params["q"], u1=limb, backend="cpu",
        )
        microjax.block_until_ready()
        values[f"microjax__{tag}"] = np.asarray(microjax, dtype=np.float64)

        kwargs = {"options": lcbinint.Options(
            coordinates="vbm", nbin="auto", finite_source_tol=TOL,
            finite_source_reltol=TOL,
        )}
        if limb:
            kwargs["limb_darkening"] = lcbinint.LimbDarkening.linear(limb)
        curve = lcbinint.LightCurve(**kwargs)
        result = curve(times, lc_params, with_gradient=False)
        values[f"lcbinint__{tag}"] = np.asarray(result.magnifications, dtype=np.float64)
        values[f"lcbinint_status__{tag}"] = np.asarray(result.status, dtype=np.int8)
        results[tag] = {
            "lcbinint_status_counts": {
                str(int(code)): int(count)
                for code, count in zip(*np.unique(np.asarray(result.status), return_counts=True))
            }
        }
        print("VALUES", tag, {
            key: np.asarray(value).tolist()
            for key, value in values.items() if key.endswith(f"__{tag}")
        }, flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "raw.npz", **values)
    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(), "python": sys.version,
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "case_id": "near_limb_contact_117",
        "input_sha256": hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        "source_indices": indices.tolist(), "times": times.tolist(), "params": params,
        "tolerance": TOL, "results": results,
        "versions": {
            "official_vbm": {"version": importlib.import_module("VBMicrolensing").__version__,
                              "module": str(Path(importlib.import_module("VBMicrolensing").__file__).resolve())},
            "autodiff_vbm": {"version": importlib.import_module("autodiff_vbm").__version__,
                              "vendored_vbm": "5.5"},
            "microjax": {"version": importlib.import_module("microjax").__version__,
                          "backend": "cpu fixed settings"},
            "lcbinint": {"head": "40e66e510799f9d3d15b459be2b38694daad52ee",
                          "backend": "algebraic_boundary default"},
        },
        "note": "Compares uniform and linear-LD cases at the same near-limb-contact epochs; does not establish a truth reference.",
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
