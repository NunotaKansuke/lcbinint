#!/usr/bin/env python3
"""Replay the frozen triple-polar target set after seed augmentation.

The input rows already contain independent Cartesian 256/384/512 references.
This program only evaluates production-resolution explicit polar mode, so it
does not spend time recomputing those references.  Every row runs in an
isolated child and is checkpointed atomically.
"""
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


def worker(connection, frozen: dict[str, Any], caustic_bins: int) -> None:
    try:
        case = frozen["case"]
        sample = frozen["sample"]
        params = dict(
            t0=0.0,
            tE=1.0,
            u0=sample["source_y"],
            alpha=0.0,
            s=case["separation"],
            q=case["mass_ratio"],
            q2=case["tertiary_mass_ratio"],
            sep2=case["tertiary_separation"],
            ang=case["tertiary_angle"],
            rho=case["source_radius"],
        )
        curve = lcbinint.LightCurve(
            lens="triple",
            options=lcbinint.Options(
                param_type="lcbinint",
                caustic_bins=caustic_bins,
                inverse_ray_grid="polar",
                nbin="auto",
                max_source_bins=512,
                polar_grid_ratio=12.0,
                point_source_threshold=1.0e9,
                hexadecapole_threshold=0.0,
                adaptive_hex_threshold=0.0,
            ),
            limb_darkening=lcbinint.LimbDarkening.linear(sample["limb_c"]),
        )
        started = time.perf_counter_ns()
        info = curve.info([sample["source_x"]], params)
        connection.send({
            "value": float(info.magnifications[0]),
            "elapsed_ns": time.perf_counter_ns() - started,
            "method": info.finite_source_method_names[0],
            "converged": bool(info.finite_source_converged[0]),
        })
    except Exception as exc:
        connection.send({"error": repr(exc)})
    finally:
        connection.close()


def evaluate(frozen: dict[str, Any], caustic_bins: int, timeout: float) -> dict[str, Any]:
    context = multiprocessing.get_context("fork")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=worker, args=(child, frozen, caustic_bins), daemon=True)
    process.start()
    child.close()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(2)
        if process.is_alive():
            process.kill()
            process.join()
        parent.close()
        return {"timeout": True, "timeout_seconds": timeout}
    try:
        return parent.recv() if parent.poll() else {"error": f"worker exited {process.exitcode}"}
    finally:
        parent.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--caustic-bins", type=int, default=1200)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    paths = sorted(args.input.glob("sample-*.json"))
    for index, path in enumerate(paths):
        if index % args.shard_count != args.shard_index:
            continue
        destination = args.output / path.name
        if destination.exists():
            continue
        frozen = json.loads(path.read_text())
        atomic_json(destination, {
            "case": frozen["case"],
            "sample": frozen["sample"],
            "reference": frozen.get("result", {}).get("reference", []),
            "polar": evaluate(frozen, args.caustic_bins, args.timeout),
        })


if __name__ == "__main__":
    main()
