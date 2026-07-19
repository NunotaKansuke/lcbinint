#!/usr/bin/env python3
"""Replay changed triple fast-path routing against frozen Cartesian references."""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import time
from pathlib import Path
from typing import Any

import lcbinint


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, allow_nan=True, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def worker(connection, case: dict[str, Any], row: dict[str, Any]) -> None:
    try:
        curve=lcbinint.LightCurve(lens="triple", options=lcbinint.Options(
            param_type="lcbinint", caustic_bins=1400, inverse_ray_grid="auto", nbin="auto",
            point_source_threshold=20.0, hexadecapole_threshold=3.0,
            adaptive_hex_threshold=1e-3, max_source_bins=512),
            limb_darkening=lcbinint.LimbDarkening.linear(row["limb_c"]))
        params=dict(t0=0.,tE=1.,u0=row["source_y"],alpha=0.,s=case["separation"],
            q=case["mass_ratio"],q2=case["tertiary_mass_ratio"],
            sep2=case["tertiary_separation"],ang=case["tertiary_angle"],rho=case["source_radius"])
        started=time.perf_counter_ns(); info=curve.info([row["source_x"]],params)
        connection.send({"value":float(info.magnifications[0]),"elapsed_ns":time.perf_counter_ns()-started,
                         "method":info.finite_source_method_names[0],
                         "reported_error":float(info.finite_source_error_estimates[0])})
    except Exception as exc:
        connection.send({"error":repr(exc)})
    finally: connection.close()


def evaluate(case: dict[str, Any], row: dict[str, Any], timeout: float) -> dict[str, Any]:
    context=multiprocessing.get_context("fork"); parent,child=context.Pipe(duplex=False)
    process=context.Process(target=worker,args=(child,case,row),daemon=True)
    process.start();child.close();process.join(timeout)
    if process.is_alive():
        process.terminate();process.join(2)
        if process.is_alive():process.kill();process.join()
        parent.close();return {"timeout":True,"timeout_seconds":timeout}
    try:return parent.recv() if parent.poll() else {"error":f"worker exited {process.exitcode}"}
    finally:parent.close()


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument("--reference-dir",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--timeout",type=float,default=30.)
    p.add_argument("--shard-count",type=int,default=1);p.add_argument("--shard-index",type=int,default=0);a=p.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=True); rows=[]
    for path in sorted(a.reference_dir.glob("sample-*.json")):
        d=json.loads(path.read_text());r=d["result"]
        if "reference_512" in r and "value" in r["reference_512"]:
            rows.append((d["case"],d["sample"],r["reference_256"],r["reference_512"]))
    rows=[x for i,x in enumerate(rows) if i%a.shard_count==a.shard_index]
    for i,(case,row,r256,r512) in enumerate(rows):
        path=a.output_dir/f"sample-{a.shard_index:02d}-{i:05d}.json"
        if path.exists():continue
        atomic_json(path,{"case":case,"sample":row,"reference_256":r256,"reference_512":r512,
                          "auto":evaluate(case,row,a.timeout)})
        print(f"shard={a.shard_index}/{a.shard_count} sample={i+1}/{len(rows)}",flush=True)

if __name__ == "__main__": main()
