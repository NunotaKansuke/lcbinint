#!/usr/bin/env python3
"""Calibrate triple point-source/hex routing against forced Cartesian tails.

Rows are replayed from the fixed-grid triple discovery set so source positions
and limb profiles are frozen independently of the production auto decision.
The reference disables point/hex exits and requires Cartesian 256/512 tails.
Each row is isolated and checkpointed, making the sweep resumable and safe
against pathological triple solves.
"""
from __future__ import annotations

import argparse
import json
import math
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


def evaluate_curve(case: dict[str, Any], row: dict[str, Any], kind: str, bins: int | None) -> dict[str, Any]:
    common = dict(param_type="lcbinint", caustic_bins=1400, max_source_bins=512)
    if kind == "auto":
        options = lcbinint.Options(
            **common, inverse_ray_grid="auto", nbin="auto",
            point_source_threshold=20.0, hexadecapole_threshold=3.0,
            adaptive_hex_threshold=1.0e-3)
    else:
        # A large kinji threshold prevents the triple point-source exit;
        # zero hex thresholds force the Cartesian inverse-ray reference.
        options = lcbinint.Options(
            **common, inverse_ray_grid="cartesian", source_bins=bins,
            polar_source_bins=bins, nbin=bins, point_source_threshold=1.0e9,
            hexadecapole_threshold=0.0, adaptive_hex_threshold=0.0)
    curve = lcbinint.LightCurve(lens="triple", options=options,
                                limb_darkening=lcbinint.LimbDarkening.linear(row["limb_c"]))
    params = dict(t0=0.0, tE=1.0, u0=row["source_y"], alpha=0.0,
                  s=case["separation"], q=case["mass_ratio"],
                  q2=case["tertiary_mass_ratio"], sep2=case["tertiary_separation"],
                  ang=case["tertiary_angle"], rho=case["source_radius"])
    started = time.perf_counter_ns()
    info = curve.info([row["source_x"]], params)
    return {"value": float(info.magnifications[0]),
            "elapsed_ns": time.perf_counter_ns() - started,
            "method": info.finite_source_method_names[0],
            "point_magnification": float(info.point_source_magnifications[0]),
            "reported_error": float(info.finite_source_error_estimates[0])}


def worker(connection, case: dict[str, Any], row: dict[str, Any], kind: str, bins: int | None) -> None:
    try:
        connection.send(evaluate_curve(case, row, kind, bins))
    except Exception as exc:
        connection.send({"error": repr(exc)})
    finally:
        connection.close()


def timed_evaluation(case: dict[str, Any], row: dict[str, Any], kind: str,
                     bins: int | None, timeout: float) -> dict[str, Any]:
    context = multiprocessing.get_context("fork")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=worker, args=(child, case, row, kind, bins), daemon=True)
    process.start(); child.close(); process.join(timeout)
    if process.is_alive():
        process.terminate(); process.join(2)
        if process.is_alive(): process.kill(); process.join()
        parent.close()
        return {"timeout": True, "timeout_seconds": timeout}
    try:
        return parent.recv() if parent.poll() else {"error": f"worker exited {process.exitcode}"}
    finally:
        parent.close()


def evaluate(case: dict[str, Any], row: dict[str, Any], timeout: float) -> dict[str, Any]:
    auto = timed_evaluation(case, row, "auto", None, min(timeout, 20.0))
    if auto.get("method") not in {"point_source", "hexadecapole"}:
        return {"auto": auto, "reference_skipped": "auto did not select a fast path"}
    return {"auto": auto,
            "reference_256": timed_evaluation(case, row, "reference", 256, timeout),
            "reference_512": timed_evaluation(case, row, "reference", 512, timeout)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--distance-min", type=float, default=2.0)
    parser.add_argument("--distance-max", type=float, default=30.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates=[]
    for path in sorted(args.input_dir.glob("case-*.json")):
        document=json.loads(path.read_text())
        if "error" in document: continue
        for row in document["rows"]:
            distance=float(row["caustic_distance_over_rho"])
            if args.distance_min <= distance <= args.distance_max:
                candidates.append((document["case"], row))
    candidates=[item for index,item in enumerate(candidates) if index % args.shard_count == args.shard_index]
    for index,(case,row) in enumerate(candidates):
        path=args.output_dir / f"sample-{args.shard_index:02d}-{index:05d}.json"
        if path.exists(): continue
        atomic_json(path, {"case": case, "sample": row, "result": evaluate(case,row,args.timeout)})
        print(f"shard={args.shard_index}/{args.shard_count} sample={index + 1}/{len(candidates)}", flush=True)


if __name__ == "__main__":
    main()
