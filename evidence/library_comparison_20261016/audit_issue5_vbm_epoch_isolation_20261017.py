from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
if hasattr(os, "sched_getaffinity"):
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})

HERE = Path(__file__).resolve().parent
INPUT = HERE / "stratified_09_lcbinint_only" / "inputs.json"
CHILD = r'''
import json, sys, numpy as np
library, source, p_json, t_text, tol_text, limb_text = sys.argv[1:]
p=np.asarray(json.loads(p_json),dtype=np.float64); t=float(t_text)
tol=float(tol_text); limb=float(limb_text)
if library == "vbm":
    sys.path.insert(0,source)
    from VBMicrolensing import VBMicrolensing
    model=VBMicrolensing(); model.Tol=model.RelTol=tol
    model.a1=limb; model.a2=0.0
    value=model.BinaryLightCurve(p.tolist(),[t])[0][0]
else:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    sys.path.insert(0,source)
    from autodiff_vbm import AutodiffVBM
    model=AutodiffVBM(); model.Tol=model.RelTol=tol
    out=model.BinaryLightCurve(jnp.asarray(p),jnp.asarray([t]),a1=limb)[0]
    out.block_until_ready()
    value=np.asarray(out,dtype=np.float64).reshape(-1)[0]
print(json.dumps({"value":float(value)}))
'''


def main() -> None:
    cases = json.loads(INPUT.read_text())["cases"]
    case = next(c for c in cases if c["id"] == "stratified_09")
    params = {k: float(v) for k, v in case["params"].items()}
    values = np.asarray([
        np.log(params["s"]), np.log(params["q"]), params["u0"],
        params["alpha"], np.log(params["rho"]), np.log(params["tE"]),
        params["t0"],
    ], dtype=np.float64)
    times = np.asarray(case["times"], dtype=np.float64)
    rows = []
    output = HERE / "issue5_vbm_epoch_isolation_20261017"
    output.mkdir(parents=True, exist_ok=True)
    source_paths = {
        "vbm": str(HERE / "vbm561"),
        "autodiff_vbm": "/rogue1_8/nunota/moasarc-ecosystem/integration/autodiff-vbm/src",
    }
    timeouts = {"vbm": 2.0, "autodiff_vbm": 5.0}
    for index in range(times.size):
        for library in ("vbm", "autodiff_vbm"):
            started = time.perf_counter()
            row = {"index": index, "time": float(times[index]), "library": library}
            try:
                proc = subprocess.run(
                    [sys.executable, "-c", CHILD, library, source_paths[library],
                     json.dumps(values.tolist(), separators=(",", ":")),
                     repr(float(times[index])), "1e-3", repr(float(case["limb"]))],
                    check=True, capture_output=True, text=True,
                    timeout=timeouts[library],
                )
                row["value"] = json.loads(proc.stdout.strip())["value"]
                row["status"] = "completed"
            except subprocess.TimeoutExpired:
                row["status"] = f"timeout_after_{timeouts[library]:g}s"
            except subprocess.CalledProcessError as error:
                row["status"] = "error"
                row["stderr_tail"] = error.stderr[-2000:]
            row["elapsed_seconds"] = time.perf_counter() - started
            rows.append(row)
            metadata = {
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "host": platform.node(), "python": sys.version,
                "cpu_affinity": sorted(os.sched_getaffinity(0)),
                "case_id": "stratified_09", "parameters": params,
                "limb": float(case["limb"]), "tolerance": 1.0e-3,
                "sampled_indices": list(range(times.size)),
                "fresh_process_per_call": True,
                "source_paths": source_paths, "per_library_timeouts": timeouts,
                "rows": rows,
            }
            (output / "metadata.json").write_text(
                json.dumps(metadata, indent=2, allow_nan=False) + "\n"
            )
            print(row, flush=True)

    by_index = {}
    for row in rows:
        by_index.setdefault(row["index"], {})[row["library"]] = row
    common = [
        index for index, pair in by_index.items()
        if all(pair.get(name, {}).get("status") == "completed"
               for name in ("vbm", "autodiff_vbm"))
    ]
    differences = [
        abs(by_index[index]["vbm"]["value"] -
            by_index[index]["autodiff_vbm"]["value"])
        for index in common
    ]
    relative_differences = [
        difference / max(abs(by_index[index]["vbm"]["value"]), 1.0e-12)
        for index, difference in zip(common, differences)
    ]
    summary = {
        "case_id": "stratified_09",
        "epoch_count": int(times.size),
        "completed_epochs": {
            name: sorted(
                index for index, pair in by_index.items()
                if pair.get(name, {}).get("status") == "completed"
            )
            for name in ("vbm", "autodiff_vbm")
        },
        "timed_out_epochs": {
            name: sorted(
                index for index, pair in by_index.items()
                if pair.get(name, {}).get("status", "").startswith("timeout")
            )
            for name in ("vbm", "autodiff_vbm")
        },
        "both_completed_epochs": len(common),
        "max_abs_difference_vbm_vs_autodiff_vbm": max(differences, default=None),
        "max_relative_difference_vbm_vs_autodiff_vbm": max(relative_differences, default=None),
        "note": "The two VBM-family implementations are not independent references; the central three epochs timed out in both.",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print("SUMMARY", json.dumps(summary, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
