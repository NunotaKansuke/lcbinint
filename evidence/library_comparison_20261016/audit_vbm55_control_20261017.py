from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
if hasattr(os, "sched_getaffinity"):
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
sys.meta_path = [finder for finder in sys.meta_path if "_editable" not in type(finder).__module__]

import numpy as np
from VBMicrolensing import VBMicrolensing

HERE = Path(__file__).resolve().parent
AUDIT = HERE / "issue4_tolerance_audit_20261017"
OUT = Path(os.environ.get(
    "LCBININT_VBM55_OUT", str(HERE / "issue4_vbm55_control_20261017")
))
TOLERANCES = (1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6)


def main() -> None:
    metadata = json.loads((AUDIT / "metadata.json").read_text())
    times = np.asarray(metadata["selected_times"], dtype=np.float64)
    params = metadata["params"]
    limb = float(metadata["limb"])
    p = [
        np.log(params["s"]), np.log(params["q"]), params["u0"],
        params["alpha"], np.log(params["rho"]), np.log(params["tE"]),
        params["t0"],
    ]
    values: dict[str, np.ndarray] = {"times": times}
    timing: dict[str, float] = {}
    for tol in TOLERANCES:
        key = f"{tol:.0e}"
        vbm = VBMicrolensing()
        vbm.Tol = vbm.RelTol = tol
        vbm.a1 = limb
        vbm.a2 = 0.0
        start = time.perf_counter()
        mag = np.asarray(vbm.BinaryLightCurve(p, times)[0], dtype=np.float64).reshape(-1)
        timing[key] = (time.perf_counter() - start) * 1.0e3
        values[f"vbm55__tol_{key}"] = mag
        print("VBM55", key, "ms", round(timing[key], 3), "values", mag.tolist(), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "raw.npz", **values)
    module = __import__("VBMicrolensing")
    metadata_out = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(), "platform": platform.platform(), "python": sys.version,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "version": module.__version__, "module": str(Path(module.__file__).resolve()),
        "native": str(Path(__import__("VBMicrolensing.VBMicrolensing", fromlist=["x"]).__file__).resolve()),
        "case_id": "near_limb_contact_117", "selected_times": times.tolist(),
        "params": params, "limb": limb, "tolerances": list(TOLERANCES),
        "elapsed_ms": timing,
        "note": "Official VBMicrolensing 5.5 wheel from the active myenv; used only as a version control against saved official 5.6.1.",
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata_out, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
