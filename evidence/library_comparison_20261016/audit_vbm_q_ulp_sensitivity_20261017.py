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
INPUT = HERE / "corrected_value_gradient_12cases" / "inputs.json"
OUT = Path(os.environ.get(
    "LCBININT_VBM_Q_ULP_OUT", str(HERE / "issue4_vbm_q_ulp_20261017")
))
CHILD = r'''
import sys, json, numpy as np
sys.path.insert(0, sys.argv[1])
from VBMicrolensing import VBMicrolensing
q = float(sys.argv[2]); mode = sys.argv[3]; tol = float(sys.argv[4])
p = json.loads(sys.argv[5]); t = float(sys.argv[6])
alpha = p["alpha"]; tau=(t-p["t0"])/p["tE"]
y1=p["u0"]*np.sin(alpha)-tau*np.cos(alpha)
y2=-p["u0"]*np.cos(alpha)-tau*np.sin(alpha)
v=VBMicrolensing(); v.Tol=v.RelTol=tol; v.a1=0.5; v.a2=0.0
if mode == "direct_BinaryMag2":
    value=v.BinaryMag2(p["s"],q,y1,y2,p["rho"])
elif mode == "direct_BinaryMagDark":
    value=v.BinaryMagDark(p["s"],q,y1,y2,p["rho"],tol)
else:
    params=[np.log(p["s"]),np.log(q),p["u0"],p["alpha"],np.log(p["rho"]),np.log(p["tE"]),p["t0"]]
    value=v.BinaryLightCurve(params,[t])[0][0]
print(json.dumps({"q_input":q,"q_after_log_exp":float(np.exp(np.log(q))),"mode":mode,"tol":tol,"value":float(value)}))
'''


def main() -> None:
    case = next(c for c in json.loads(INPUT.read_text())["cases"] if c["id"] == "near_limb_contact_117")
    p = {key: float(value) for key, value in case["params"].items()}
    time_value = float(np.asarray(case["times"], dtype=np.float64)[8])
    q = p["q"]
    q_values = {
        "physical_input": q,
        "nextafter_down": float(np.nextafter(q, -np.inf)),
        "nextafter_up": float(np.nextafter(q, np.inf)),
        "log_exp_roundtrip": float(np.exp(np.log(q))),
        "relative_plus_1e-12": float(q * (1.0 + 1.0e-12)),
        "relative_minus_1e-12": float(q * (1.0 - 1.0e-12)),
    }
    root = str(HERE / "vbm561")
    rows = []
    for name, value in q_values.items():
        for tol in (1.0e-3, 1.0e-6):
            for mode in ("direct_BinaryMagDark", "direct_BinaryMag2", "BinaryLightCurve_log_q"):
                result = subprocess.run(
                    [sys.executable, "-c", CHILD, root, repr(value), mode, repr(tol),
                     json.dumps(p, separators=(",", ":")), repr(time_value)],
                    check=True, capture_output=True, text=True,
                )
                row = json.loads(result.stdout.strip())
                row["q_case"] = name
                rows.append(row)
                print(name, mode, f"tol={tol:.0e}", "q_roundtrip", row["q_after_log_exp"],
                      "value", row["value"], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(), "python": sys.version,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "case_id": "near_limb_contact_117", "source_index": 8,
        "time": time_value, "params": p,
        "version": "5.6.1", "source_revision": "db4f95267d74ad5f4dc592ab1e6d46c05f6cc1b7",
        "fresh_process_per_call": True,
        "q_values": q_values, "rows": rows,
        "warning": "This probes floating-point sensitivity in the VBM 5.6.1 public calculation; it does not certify the q>0 true value.",
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
