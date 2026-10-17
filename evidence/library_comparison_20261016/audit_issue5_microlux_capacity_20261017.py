from __future__ import annotations

import argparse
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
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

HERE = Path(__file__).resolve().parent
sys.path = [p for p in sys.path if "_editable" not in p]
sys.path.insert(0, "/rogue1_8/nunota/microlux/src")
microlux_model = importlib.import_module("microlux.model")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--capacity", type=int,
        help="run only one diagnostic strategy with this total sample capacity",
    )
    parser.add_argument("--return-info", action="store_true")
    args = parser.parse_args()
    case = next(
        c for c in json.loads((HERE / "stratified_09_lcbinint_only/inputs.json").read_text())["cases"]
        if c["id"] == "stratified_09"
    )
    params = {k: float(v) for k, v in case["params"].items()}
    all_times = np.asarray(case["times"], dtype=np.float64)
    indices = [12, 13, 14]
    times = jnp.asarray(all_times[indices])
    p = np.asarray([
        np.log(params["s"]), np.log(params["q"]), params["u0"],
        params["alpha"] + np.pi, np.log(params["rho"]),
        np.log(params["tE"]), params["t0"],
    ], dtype=np.float64)
    if args.capacity is None:
        strategies = {
            "saved_default": (30, 30, 60, 120, 240),
            "expanded_capacity": (30, 30, 60, 120, 2160),
        }
        output = HERE / "issue5_microlux_capacity_20261017"
    else:
        if args.capacity <= 240:
            raise ValueError("capacity must exceed the fixed 240-sample prefix")
        strategies = {
            f"diagnostic_capacity_{args.capacity}":
                (30, 30, 60, 120, args.capacity - 240),
        }
        info_suffix = "_info" if args.return_info else ""
        output = HERE / f"issue5_microlux_capacity_{args.capacity}{info_suffix}_20261017"
    rows = {}
    output.mkdir(parents=True, exist_ok=True)
    for name, strategy in strategies.items():
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            output_value = microlux_model.binary_mag(
                p[6], p[2], np.exp(p[5]), np.exp(p[4]), np.exp(p[1]),
                np.exp(p[0]), p[3] * (180.0 / np.pi), times,
                tol=1.0e-3, retol=1.0e-3,
                default_strategy=strategy,
                limb_darkening_coeff=case["limb"],
                return_info=args.return_info,
            )
            for leaf in jax.tree_util.tree_leaves(output_value):
                if hasattr(leaf, "block_until_ready"):
                    leaf.block_until_ready()
        if args.return_info:
            values, info = output_value
            root_state, error_state = info[-2:]
            mag = np.squeeze(np.asarray(error_state.mag, dtype=np.float64))
            error_hist = np.squeeze(
                np.asarray(error_state.error_hist, dtype=np.float64), axis=-1
            )
            epsilon = np.squeeze(np.asarray(error_state.epsilon, dtype=np.float64))
            epsilon_rel = np.squeeze(
                np.asarray(error_state.epsilon_rel, dtype=np.float64)
            )
            sample_n = np.squeeze(
                np.asarray(root_state.sample_num, dtype=np.float64)
            )
            abs_error_sum = np.nansum(error_hist, axis=-1)
            rel_threshold = epsilon_rel / np.sqrt(sample_n / 2.0)
            rel_ratios = error_hist / np.maximum(np.abs(mag)[..., None], 1.0e-300)
            rel_ratios = rel_ratios / np.maximum(
                rel_threshold[..., None], 1.0e-300
            )
            state_summary = {
                "sample_num": np.asarray(root_state.sample_num).tolist(),
                "mag": mag.tolist(),
                "estimated_abs_error_sum": abs_error_sum.tolist(),
                "epsilon": epsilon.tolist(),
                "epsilon_rel": epsilon_rel.tolist(),
                "abs_error_over_epsilon": (abs_error_sum / epsilon).tolist(),
                "max_relative_interval_ratio": np.nanmax(rel_ratios, axis=-1).tolist(),
                "mag_no_diff": np.asarray(error_state.mag_no_diff).tolist(),
                "outloop": np.asarray(error_state.outloop).tolist(),
                "exceed_flag": np.asarray(error_state.exceed_flag).tolist(),
            }
        else:
            values = output_value
            state_summary = None
        rows[name] = {
            "default_strategy": list(strategy),
            "max_length": sum(strategy),
            "values": np.asarray(values, dtype=np.float64).tolist(),
            "elapsed_seconds_including_compile": time.perf_counter() - started,
            "warnings": [str(item.message) for item in caught],
            "return_info_state": state_summary,
        }
        print(name, rows[name], flush=True)
        metadata = {
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "host": platform.node(), "python": sys.version,
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "case_id": "stratified_09", "indices": indices,
            "times": all_times[indices].tolist(), "params": params,
            "limb_darkening_coefficient": float(case["limb"]),
            "tolerance": 1.0e-3,
            "source": str(Path(microlux_model.__file__).resolve()),
            "source_revision": "711245b6f557e5b1548b8e27d85ced7384c13418",
            "rows": rows,
            "note": "The expanded sampling capacity is a diagnostic rerun, not a new validated tolerance guarantee.",
        }
        (output / "metadata.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n"
        )


if __name__ == "__main__":
    main()
