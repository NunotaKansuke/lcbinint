#!/usr/bin/env python3
"""Production triple-lens Cartesian/polar calibration sweep.

Every grid/resolution evaluation runs in an isolated process.  Output is
checkpointed per lens geometry, is resumable, and supports independent shards.
No automatic grid or automatic nbin result is used as a numerical reference.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import lcbinint


DEFAULT_BINS = (16, 24, 32, 40, 50, 64, 80, 100, 128, 160, 200, 256)
DISTANCE_FACTORS = (0.0, 0.2, 0.5, 0.8, 1.0, 1.2, 1.6, 2.0, 3.0, 5.0, 10.0, 30.0)


@dataclass(frozen=True)
class TripleCase:
    case_id: int
    separation: float
    mass_ratio: float
    tertiary_mass_ratio: float
    tertiary_separation: float
    tertiary_angle: float
    source_radius: float


def log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(10.0 ** rng.uniform(math.log10(low), math.log10(high)))


def make_cases(count: int, seed: int) -> list[TripleCase]:
    """Cover topology boundaries and then fill the six-dimensional interior."""
    rng = np.random.default_rng(seed)
    s_values = (0.3, 0.6, 0.8, 0.95, 1.0, 1.05, 1.3, 2.0, 4.0)
    q_values = (1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5)
    q2_values = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1)
    d2_values = (0.08, 0.2, 0.5, 1.0, 2.0)
    angle_values = (0.0, math.pi / 6, math.pi / 2, 5 * math.pi / 6)
    rho_values = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
    output: list[TripleCase] = []
    anchors = min(count, 120)
    for i in range(anchors):
        output.append(TripleCase(
            i,
            s_values[i % len(s_values)],
            q_values[(3 * i + i // 9) % len(q_values)],
            q2_values[(5 * i + i // 6) % len(q2_values)],
            d2_values[(7 * i + i // 5) % len(d2_values)],
            angle_values[(11 * i + i // 4) % len(angle_values)],
            rho_values[(13 * i + i // 8) % len(rho_values)],
        ))
    while len(output) < count:
        i = len(output)
        output.append(TripleCase(
            i,
            log_uniform(rng, 0.25, 4.0),
            log_uniform(rng, 1e-5, 0.5),
            log_uniform(rng, 1e-6, 0.1),
            log_uniform(rng, 0.05, 2.5),
            float(rng.uniform(0.0, math.pi)),
            log_uniform(rng, 1e-5, 3e-2),
        ))
    return output


def caustic_geometry(case: TripleCase, bins: int) -> dict[str, Any]:
    curve = lcbinint.LightCurve(
        lens="triple", options=lcbinint.Options(param_type="lcbinint", caustic_bins=bins))
    result = curve.caustics(
        s=case.separation, q=case.mass_ratio, q2=case.tertiary_mass_ratio,
        sep2=case.tertiary_separation, ang=case.tertiary_angle, n_points=bins)
    branches = [
        np.column_stack((np.asarray(xs, float), np.asarray(ys, float)))
        for xs, ys in zip(result.x, result.y) if len(xs) >= 2
    ]
    starts = np.concatenate(branches)
    ends = np.concatenate([np.roll(branch, -1, axis=0) for branch in branches])
    points = np.concatenate(branches)
    return {"branches": branches, "starts": starts, "ends": ends, "points": points,
            "bbox_min": points.min(axis=0), "bbox_max": points.max(axis=0)}


def nearest_distance(point: np.ndarray, geometry: dict[str, Any]) -> float:
    starts, vectors = geometry["starts"], geometry["ends"] - geometry["starts"]
    length2 = np.einsum("ij,ij->i", vectors, vectors)
    projection = np.divide(np.einsum("ij,ij->i", point - starts, vectors), length2,
                           out=np.zeros_like(length2), where=length2 > 0)
    closest = starts + np.clip(projection, 0, 1)[:, None] * vectors
    return math.sqrt(float(np.min(np.einsum("ij,ij->i", point - closest, point - closest))))


def source_points(case: TripleCase, geometry: dict[str, Any], count: int, seed: int):
    rng = np.random.default_rng(seed + 104729 * (case.case_id + 1))
    points = geometry["points"]
    output = []
    near_count = max(1, count - max(2, count // 6))
    for i in range(near_count):
        anchor = points[int(rng.integers(len(points)))]
        factor = DISTANCE_FACTORS[i % len(DISTANCE_FACTORS)]
        angle = float(rng.uniform(0, 2 * math.pi))
        sample = anchor + factor * case.source_radius * np.array((math.cos(angle), math.sin(angle)))
        output.append((float(sample[0]), float(sample[1]), "caustic_offset", factor))
    pad = max(0.25, 50 * case.source_radius)
    while len(output) < count:
        sample = rng.uniform(geometry["bbox_min"] - pad, geometry["bbox_max"] + pad)
        output.append((float(sample[0]), float(sample[1]), "field", math.nan))
    return output


def worker(connection, case: TripleCase, x: float, y: float, limb: float,
           grid: str, bins: int, caustic_bins: int) -> None:
    try:
        curve = lcbinint.LightCurve(
            lens="triple",
            options=lcbinint.Options(
                param_type="lcbinint", source_bins=bins, polar_source_bins=bins,
                # A fixed-grid sweep must bypass the production auto-nbin
                # selector.  Otherwise its labels feed back into themselves.
                nbin=bins,
                inverse_ray_grid=grid, caustic_bins=caustic_bins,
                point_source_threshold=20.0, hexadecapole_threshold=0.0,
                adaptive_hex_threshold=0.0, max_source_bins=bins),
            limb_darkening=lcbinint.LimbDarkening.linear(limb))
        params = dict(t0=0., tE=1., u0=y, alpha=0., s=case.separation,
                      q=case.mass_ratio, q2=case.tertiary_mass_ratio,
                      sep2=case.tertiary_separation, ang=case.tertiary_angle,
                      rho=case.source_radius)
        started = time.perf_counter_ns()
        info = curve.info([x], params)
        connection.send({
            "value": float(info.magnifications[0]),
            "elapsed_ns": time.perf_counter_ns() - started,
            "method": info.finite_source_method_names[0],
            "reported_error": float(info.finite_source_error_estimates[0]),
            "point_magnification": float(info.point_source_magnifications[0]),
        })
    except Exception as exc:
        connection.send({"error": repr(exc)})
    finally:
        connection.close()


def timed_evaluation(case: TripleCase, x: float, y: float, limb: float,
                     grid: str, bins: int, caustic_bins: int, timeout: float) -> dict[str, Any]:
    context = multiprocessing.get_context("fork")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=worker,
        args=(child, case, x, y, limb, grid, bins, caustic_bins), daemon=True)
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


def evaluate_case(case: TripleCase, args: argparse.Namespace) -> dict[str, Any]:
    geometry = caustic_geometry(case, args.caustic_bins)
    rows = []
    for point_id, (x, y, sampling, requested_factor) in enumerate(
            source_points(case, geometry, args.points_per_case, args.seed)):
        distance = nearest_distance(np.asarray((x, y)), geometry)
        for limb in args.limb_coefficients:
            sequences = {}
            for grid in ("cartesian", "polar"):
                sequence = []
                timed_out_at = None
                for bins in args.bins:
                    if timed_out_at is None:
                        result = timed_evaluation(case, x, y, limb, grid, bins,
                                                  args.caustic_bins, args.timeout)
                        if result.get("timeout"): timed_out_at = bins
                    else:
                        result = {"skipped_after_timeout": True, "lower_timeout_bins": timed_out_at}
                    result["bins"] = bins
                    sequence.append(result)
                sequences[grid] = sequence
            point_mag = next((float(r["point_magnification"])
                              for seq in sequences.values() for r in seq
                              if "point_magnification" in r), math.nan)
            rows.append({
                "point_id": point_id, "source_x": x, "source_y": y,
                "sampling": sampling, "requested_distance_factor": requested_factor,
                "limb_c": limb, "caustic_distance": distance,
                "caustic_distance_over_rho": distance / case.source_radius,
                "point_magnification": point_mag, "fixed_sequences": sequences,
            })
    return {"schema_version": 1, "case": asdict(case), "configuration": {
        "bins": args.bins, "limb_coefficients": args.limb_coefficients,
        "caustic_bins": args.caustic_bins, "timeout": args.timeout,
        "seed": args.seed}, "rows": rows}


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, allow_nan=True, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lens-cases", type=int, default=256)
    parser.add_argument("--points-per-case", type=int, default=12)
    parser.add_argument("--bins", type=lambda s: tuple(int(x) for x in s.split(",")), default=DEFAULT_BINS)
    parser.add_argument("--limb-coefficients", type=lambda s: tuple(float(x) for x in s.split(",")), default=(0., .5, .8))
    parser.add_argument("--caustic-bins", type=int, default=1200)
    parser.add_argument("--timeout", type=float, default=5.)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assigned = [c for c in make_cases(args.lens_cases, args.seed)
                if c.case_id % args.shard_count == args.shard_index]
    completed = 0; started = time.monotonic()
    for case in assigned:
        destination = args.output_dir / f"case-{case.case_id:06d}.json"
        if destination.exists() and not args.overwrite:
            completed += 1; continue
        case_started = time.monotonic()
        try: result = evaluate_case(case, args)
        except Exception as exc: result = {"schema_version": 1, "case": asdict(case), "error": repr(exc)}
        result["case_elapsed_seconds"] = time.monotonic() - case_started
        atomic_json(destination, result); completed += 1
        if completed % args.progress_every == 0 or completed == len(assigned):
            print(f"shard={args.shard_index}/{args.shard_count} cases={completed}/{len(assigned)} "
                  f"elapsed_s={time.monotonic()-started:.1f}", flush=True)


if __name__ == "__main__":
    main()
