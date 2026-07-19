#!/usr/bin/env python3
"""Replay triple topology-aware auto routing against frozen Cartesian tails.

Input is a fixed-grid triple calibration directory.  Each production-auto and
forced-Cartesian 512-bin evaluation runs in its own process with a timeout and
is checkpointed per row, so a pathological topology cannot stop the sweep.
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
import numpy as np


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, allow_nan=True, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def evaluate_direct(case: dict[str, Any], row: dict[str, Any], kind: str) -> dict[str, Any]:
    common = dict(param_type="lcbinint", caustic_bins=1400, max_source_bins=512)
    if kind == "auto":
        options = lcbinint.Options(
            **common, inverse_ray_grid="auto", nbin="auto",
            point_source_threshold=20.0, hexadecapole_threshold=3.0,
            adaptive_hex_threshold=1.0e-3)
    else:
        options = lcbinint.Options(
            **common, mode=1, inverse_ray_grid="cartesian", nbin=512,
            source_bins=512, polar_source_bins=512,
            point_source_threshold=1.0e9, hexadecapole_threshold=0.0,
            adaptive_hex_threshold=0.0)
    curve = lcbinint.LightCurve(
        lens="triple", options=options,
        limb_darkening=lcbinint.LimbDarkening.linear(row["limb_c"]))
    params = dict(
        t0=0.0, tE=1.0, u0=row["source_y"], alpha=0.0,
        s=case["separation"], q=case["mass_ratio"],
        q2=case["tertiary_mass_ratio"], sep2=case["tertiary_separation"],
        ang=case["tertiary_angle"], rho=case["source_radius"])
    started = time.perf_counter_ns()
    info = curve.info([row["source_x"]], params)
    return {
        "value": float(info.magnifications[0]),
        "elapsed_ns": time.perf_counter_ns() - started,
        "method": info.finite_source_method_names[0],
        "reported_error": float(info.finite_source_error_estimates[0]),
        "refinement_level": int(info.finite_source_refinement_levels[0]),
        "converged": bool(info.finite_source_converged[0]),
        "point_magnification": float(info.point_source_magnifications[0]),
    }


def evaluate_chord_direct(
        case: dict[str, Any], row: dict[str, Any], order: int) -> dict[str, Any]:
    """Independent high-order source-plane disk integral for disputed tails."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    curve = lcbinint.LightCurve(
        lens="triple",
        options=lcbinint.Options(param_type="lcbinint", mode=0))
    weighted = 0.0
    brightness_norm = 0.0
    started = time.perf_counter_ns()
    for eta, wy in zip(nodes, weights):
        half_chord = math.sqrt(max(0.0, 1.0 - eta * eta))
        if half_chord == 0.0:
            continue
        xs = row["source_x"] + case["source_radius"] * half_chord * nodes
        params = dict(
            t0=0.0, tE=1.0,
            u0=row["source_y"] + case["source_radius"] * eta,
            alpha=0.0, s=case["separation"], q=case["mass_ratio"],
            q2=case["tertiary_mass_ratio"], sep2=case["tertiary_separation"],
            ang=case["tertiary_angle"], rho=0.0)
        magnifications = np.asarray(curve(xs, params), dtype=float)
        xi = half_chord * nodes
        radius2 = xi * xi + eta * eta
        mu = np.sqrt(np.maximum(0.0, 1.0 - radius2))
        brightness = 1.0 - row["limb_c"] * (1.0 - mu)
        weighted += wy * half_chord * float(np.sum(weights * brightness * magnifications))
        brightness_norm += wy * half_chord * float(np.sum(weights * brightness))
    return {
        "value": weighted / brightness_norm,
        "elapsed_ns": time.perf_counter_ns() - started,
        "method": "independent_source_plane_chord",
        "order": order,
    }


def worker(connection, case: dict[str, Any], row: dict[str, Any], kind: str) -> None:
    try:
        if kind.startswith("chord-"):
            connection.send(evaluate_chord_direct(case, row, int(kind.split("-")[1])))
        else:
            connection.send(evaluate_direct(case, row, kind))
    except Exception as exc:
        connection.send({"error": repr(exc)})
    finally:
        connection.close()


def timed(case: dict[str, Any], row: dict[str, Any], kind: str, timeout: float) -> dict[str, Any]:
    context = multiprocessing.get_context("fork")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=worker, args=(child, case, row, kind), daemon=True)
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
        if parent.poll():
            try:
                return parent.recv()
            except EOFError:
                return {"error": f"worker exited {process.exitcode} without a result"}
        return {"error": f"worker exited {process.exitcode}"}
    finally:
        parent.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--distance-min", type=float, default=0.0)
    parser.add_argument("--distance-max", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--skip-cartesian-reference", action="store_true")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for path in sorted(args.input_dir.glob("case-*.json")):
        document = json.loads(path.read_text())
        case = document.get("case")
        if case is None:
            continue
        for row in document.get("rows", []):
            ratio = float(row["caustic_distance_over_rho"])
            if args.distance_min <= ratio <= args.distance_max:
                rows.append((case, row))
    rows = [item for index, item in enumerate(rows)
            if index % args.shard_count == args.shard_index]

    for index, (case, row) in enumerate(rows):
        destination = args.output_dir / f"sample-{args.shard_index:02d}-{index:05d}.json"
        if destination.exists():
            continue
        auto = timed(case, row, "auto", min(args.timeout, 60.0))
        result = {
            "case": case,
            "sample": {key: value for key, value in row.items()
                       if key != "fixed_sequences"},
            "auto": auto,
        }
        cartesian = [entry for entry in row["fixed_sequences"]["cartesian"]
                     if "value" in entry]
        polar = [entry for entry in row["fixed_sequences"]["polar"]
                 if "value" in entry]
        if cartesian:
            result["input_cartesian_tail"] = cartesian[-1]
        if polar:
            result["input_polar_tail"] = polar[-1]
        # Cartesian auto is bit-identical to the already stored fixed-256 row
        # and was covered by the resolution sweep.  A new 512-bin solve is
        # needed only when topology routing changed the numerical method.
        if (auto.get("method") == "source_plane_quadrature" and
                not args.skip_cartesian_reference):
            result["reference_512"] = timed(case, row, "reference", args.timeout)
        else:
            result["reference_skipped"] = "topology routing left Cartesian unchanged"
        if (auto.get("method") == "source_plane_quadrature" and cartesian and polar):
            tolerance = 1.0e-4 + 1.0e-3 * max(abs(cartesian[-1]["value"]), 1.0)
            if abs(cartesian[-1]["value"] - polar[-1]["value"]) > tolerance:
                result["chord_160"] = timed(case, row, "chord-160", args.timeout)
                result["chord_256"] = timed(case, row, "chord-256", args.timeout)
        atomic_json(destination, result)
        print(f"shard={args.shard_index}/{args.shard_count} "
              f"sample={index + 1}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
