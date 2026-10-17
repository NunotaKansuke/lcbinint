from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
INPUT = HERE / "corrected_value_gradient_12cases" / "inputs.json"
CHILD = r'''
import json, sys, numpy as np
sys.path.insert(0, sys.argv[1])
from VBMicrolensing import VBMicrolensing
p=json.loads(sys.argv[2]); t=float(sys.argv[3]); variant=sys.argv[4]
def rt(x): return float(np.exp(np.log(float(x))))
s,q,rho=p["s"],p["q"],p["rho"]
tE_inv=1.0/p["tE"]
if variant in ("s_roundtrip", "all_roundtrip", "all_roundtrip_BinaryMagDark"):
    s=rt(s)
if variant in ("q_roundtrip", "all_roundtrip", "all_roundtrip_BinaryMagDark"):
    q=rt(q)
if variant in ("rho_roundtrip", "all_roundtrip", "all_roundtrip_BinaryMagDark"):
    rho=rt(rho)
if variant in ("all_roundtrip", "all_roundtrip_BinaryMagDark"):
    tE_inv=float(np.exp(-np.log(p["tE"])))
alpha=p["alpha"]
tau=(t-p["t0"])*tE_inv
y1=p["u0"]*np.sin(alpha)-tau*np.cos(alpha)
y2=-p["u0"]*np.cos(alpha)-tau*np.sin(alpha)
v=VBMicrolensing(); v.Tol=v.RelTol=1e-6; v.a1=0.5; v.a2=0.0
if variant == "high_level":
    pr=[np.log(p["s"]),np.log(p["q"]),p["u0"],alpha,np.log(p["rho"]),np.log(p["tE"]),p["t0"]]
    value=v.BinaryLightCurve(pr,[t])[0][0]
elif variant == "all_roundtrip_BinaryMagDark":
    value=v.BinaryMagDark(s,q,y1,y2,rho,1.0e-6)
else:
    value=v.BinaryMag2(s,q,y1,y2,rho)
print(json.dumps({"variant":variant,"s":s,"q":q,"rho":rho,"y1":y1,"y2":y2,"value":float(value)}))
'''


def main() -> None:
    case = next(c for c in json.loads(INPUT.read_text())["cases"]
                if c["id"] == "near_limb_contact_117")
    params = {k: float(v) for k, v in case["params"].items()}
    t = float(np.asarray(case["times"], dtype=np.float64)[8])
    root = str(HERE / "vbm561")
    rows = []
    for variant in ("physical", "s_roundtrip", "q_roundtrip", "rho_roundtrip",
                    "all_roundtrip", "all_roundtrip_BinaryMagDark", "high_level"):
        output = subprocess.run(
            [sys.executable, "-c", CHILD, root,
             json.dumps(params, separators=(",", ":")), repr(t), variant],
            check=True, capture_output=True, text=True,
        )
        rows.append(json.loads(output.stdout.strip()))
    OUT = HERE / "issue4_vbm_roundtrip_components_20261017"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metadata.json").write_text(json.dumps({
        "case_id": "near_limb_contact_117", "time": t, "params": params,
        "version": "5.6.1", "source_revision": "db4f95267d74ad5f4dc592ab1e6d46c05f6cc1b7",
        "fresh_process_per_call": True,
        "rows": rows,
        "interpretation": "Separate log/exp round-trip effects in BinaryMag2 from the public BinaryLightCurve path.",
    }, indent=2, allow_nan=False) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    main()
