#!/usr/bin/env python3
"""Long-timeout validation of triple auto-grid decisions on frozen samples.

The discovery sweep intentionally evaluates explicit fixed grids.  This runner
uses exactly the production auto settings (including auto polar's grid-ratio
floor) on the high-magnification rows selected from that independent sweep,
while retaining a fixed Cartesian convergence sequence as its reference.
Each source/limb row is checkpointed independently and can be resumed.
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


def options(kind: str, bins: int | None, limb: float, caustic_bins: int) -> lcbinint.LightCurve:
    common = dict(
        param_type="lcbinint", caustic_bins=caustic_bins,
        point_source_threshold=20.0, hexadecapole_threshold=0.0,
        adaptive_hex_threshold=0.0, max_source_bins=512,
    )
    if kind == "cart_fixed":
        kwargs = dict(common, inverse_ray_grid="cartesian", source_bins=bins,
                      polar_source_bins=bins, nbin=bins)
    elif kind == "cart_auto":
        kwargs = dict(common, inverse_ray_grid="cartesian", source_bins=50, nbin="auto")
    elif kind == "polar_auto":
        kwargs = dict(common, inverse_ray_grid="auto", source_bins=50, nbin="auto")
    elif kind.startswith("polar_ratio_"):
        # Probe the angular-grid ratios directly.  mode=4 historically forced
        # 12; explicit polar defaults to the base ratio 4.
        ratio = float(kind.rsplit("_", 1)[1])
        kwargs = dict(common, inverse_ray_grid="polar", source_bins=50, nbin="auto",
                      polar_grid_ratio=ratio)
    else:
        raise ValueError(kind)
    return lcbinint.LightCurve(lens="triple", options=lcbinint.Options(**kwargs),
                                limb_darkening=lcbinint.LimbDarkening.linear(limb))


def worker(connection, row: dict[str, Any], caustic_bins: int, reference_bins: tuple[int, ...]) -> None:
    try:
        case, sample = row["case"], row["sample"]
        params = dict(t0=0.0, tE=1.0, u0=sample["source_y"], alpha=0.0,
                      s=case["separation"], q=case["mass_ratio"],
                      q2=case["tertiary_mass_ratio"], sep2=case["tertiary_separation"],
                      ang=case["tertiary_angle"], rho=case["source_radius"])
        x = sample["source_x"]
        def run(kind: str, bins: int | None = None) -> dict[str, Any]:
            curve = options(kind, bins, sample["limb_c"], caustic_bins)
            started = time.perf_counter_ns()
            info = curve.info([x], params)
            return {"value": float(info.magnifications[0]),
                    "elapsed_ns": time.perf_counter_ns() - started,
                    "method": info.finite_source_method_names[0],
                    "point_magnification": float(info.point_source_magnifications[0])}
        connection.send({"reference": [{"bins": b, **run("cart_fixed", b)} for b in reference_bins],
                         "cart_auto": run("cart_auto"),
                         "polar_auto": run("polar_auto"),
                         "polar_ratio_4": run("polar_ratio_4"),
                         "polar_ratio_8": run("polar_ratio_8"),
                         "polar_ratio_12": run("polar_ratio_12")})
    except Exception as exc:
        connection.send({"error": repr(exc)})
    finally:
        connection.close()


def evaluate(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    context = multiprocessing.get_context("fork")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=worker,
        args=(child, row, args.caustic_bins, args.reference_bins), daemon=True)
    process.start(); child.close(); process.join(args.timeout)
    if process.is_alive():
        process.terminate(); process.join(2)
        if process.is_alive(): process.kill(); process.join()
        parent.close()
        return {"timeout": True, "timeout_seconds": args.timeout}
    try:
        return parent.recv() if parent.poll() else {"error": f"worker exited {process.exitcode}"}
    finally:
        parent.close()


def candidates(directory: Path, cutoff: float, distance: float) -> list[dict[str, Any]]:
    output = []
    for path in sorted(directory.glob("case-*.json")):
        doc = json.loads(path.read_text())
        if "error" in doc:
            continue
        for sample in doc["rows"]:
            if (abs(float(sample["point_magnification"])) >= cutoff and
                    float(sample["caustic_distance_over_rho"]) >= distance):
                output.append({"case": doc["case"], "sample": sample})
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--point-mag-cutoff", type=float, default=100.0)
    parser.add_argument("--distance-over-rho-min", type=float, default=3.0)
    parser.add_argument("--reference-bins", type=lambda x: tuple(int(v) for v in x.split(",")),
                        default=(256, 384, 512))
    parser.add_argument("--caustic-bins", type=int, default=1800)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = candidates(args.discovery_dir, args.point_mag_cutoff, args.distance_over_rho_min)
    rows = [row for index, row in enumerate(rows) if index % args.shard_count == args.shard_index]
    if args.max_samples is not None:
        rows = rows[:args.max_samples]
    for index, row in enumerate(rows):
        path = args.output_dir / f"sample-{args.shard_index:02d}-{index:05d}.json"
        if path.exists():
            continue
        result = {"case": row["case"], "sample": row["sample"], "result": evaluate(row, args)}
        atomic_json(path, result)
        print(f"shard={args.shard_index}/{args.shard_count} sample={index + 1}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
