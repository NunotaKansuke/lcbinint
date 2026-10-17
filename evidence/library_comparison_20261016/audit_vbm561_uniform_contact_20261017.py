from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
if hasattr(os, "sched_getaffinity"):
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vbm561"))
import numpy as np
from VBMicrolensing import VBMicrolensing as VBM

INPUT = HERE / "corrected_value_gradient_12cases" / "inputs.json"
OUT = Path(os.environ.get(
    "LCBININT_VBM561_UNIFORM_OUT", str(HERE / "issue4_vbm561_uniform_20261017")
))
INDICES = np.asarray([6, 7, 8, 20], dtype=np.int64)
TOLERANCES = (1.0e-3, 1.0e-6)


def main() -> None:
    case = next(c for c in json.loads(INPUT.read_text())["cases"] if c["id"] == "near_limb_contact_117")
    params = case["params"]
    times = np.asarray(case["times"], dtype=np.float64)[INDICES]
    p = [
        np.log(params["s"]), np.log(params["q"]), float(params["u0"]),
        float(params["alpha"]), np.log(params["rho"]), np.log(params["tE"]),
        float(params["t0"]),
    ]
    arrays: dict[str, np.ndarray] = {"times": times, "source_indices": INDICES}
    timing = {}
    for tol in TOLERANCES:
        key = f"{tol:.0e}"
        vbm = VBM()
        vbm.Tol = vbm.RelTol = tol
        vbm.a1 = 0.0
        vbm.a2 = 0.0
        start = time.perf_counter()
        values = np.asarray(vbm.BinaryLightCurve(p, times)[0], dtype=np.float64)
        timing[key] = (time.perf_counter() - start) * 1.0e3
        arrays[f"vbm561_uniform__tol_{key}"] = values
        print("VBM561_UNIFORM", key, "ms", round(timing[key], 3), "values", values.tolist(), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "raw.npz", **arrays)
    module = __import__("VBMicrolensing")
    native = Path(__import__("VBMicrolensing.VBMicrolensing", fromlist=["x"]).__file__).resolve()
    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(), "python": sys.version,
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "version": module.__version__,
        "source_revision": "db4f95267d74ad5f4dc592ab1e6d46c05f6cc1b7",
        "native": str(native), "native_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
        "case_id": "near_limb_contact_117", "source_indices": INDICES.tolist(),
        "times": times.tolist(), "params": params, "limb": 0.0,
        "tolerances": list(TOLERANCES), "elapsed_ms": timing,
        "note": "Official VBM 5.6.1 uniform-source control at the same near-limb-contact epochs.",
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
