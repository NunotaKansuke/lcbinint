#!/usr/bin/env python3
"""Generate calibration data for finite-source resolution and engine choice.

This is deliberately independent of automatic resolution.  Every
resolution is evaluated from scratch, and the output retains the complete
convergence sequence rather than deciding which implementation is correct
during data collection.

The unit of checkpointing is one lens case.  Multiple processes can safely
work in the same output directory when they use different shard indices.
Completed case files are skipped, so interrupted sweeps are resumable.
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
from typing import Any, Iterable

import numpy as np

import lcbinint
from VBMicrolensing import VBMicrolensing


DEFAULT_BINS = (16, 24, 32, 40, 50, 64, 80, 100, 128, 160, 200, 256)
DISTANCE_FACTORS = (0.25, 0.5, 0.8, 1.0, 1.2, 1.6, 2.0, 3.0, 5.0, 10.0, 30.0)


@dataclass(frozen=True)
class LensCase:
    case_id: int
    separation: float
    mass_ratio: float
    source_radius: float


def _log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(10.0 ** rng.uniform(math.log10(low), math.log10(high)))


def make_lens_cases(count: int, seed: int) -> list[LensCase]:
    """Cover topology boundaries first, then fill the log-space interior."""
    rng = np.random.default_rng(seed)
    separations = (0.1, 0.3, 0.6, 0.8, 0.95, 1.0, 1.05, 1.3, 2.0, 4.0)
    mass_ratios = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0)
    source_radii = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1)
    cases: list[LensCase] = []
    anchor_count = min(count, 90)
    for i in range(anchor_count):
        cases.append(LensCase(
            i,
            separations[i % len(separations)],
            mass_ratios[(3 * i + i // len(separations)) % len(mass_ratios)],
            source_radii[(5 * i + i // len(mass_ratios)) % len(source_radii)],
        ))
    while len(cases) < count:
        i = len(cases)
        cases.append(LensCase(
            i,
            _log_uniform(rng, 0.1, 4.0),
            _log_uniform(rng, 1e-6, 1.0),
            _log_uniform(rng, 1e-5, 0.1),
        ))
    return cases


def _flatten_branches(caustics: Any) -> list[np.ndarray]:
    return [
        np.column_stack((np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)))
        for xs, ys in zip(caustics.x, caustics.y)
        if len(xs) >= 2
    ]


def _segments(
    branches: Iterable[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    component_ids: list[np.ndarray] = []
    for component_id, branch in enumerate(branches):
        starts.append(branch)
        ends.append(np.roll(branch, -1, axis=0))
        component_ids.append(np.full(len(branch), component_id, dtype=int))
    return np.concatenate(starts), np.concatenate(ends), np.concatenate(component_ids)


def caustic_geometry(case: LensCase, bins: int) -> dict[str, Any]:
    geometry = lcbinint.LightCurve(
        options=lcbinint.Options(coordinates="center_of_mass", caustic_bins=bins)
    )
    branches = _flatten_branches(geometry.caustics(
        s=case.separation, q=case.mass_ratio, n_points=bins
    ))
    starts, ends, component_ids = _segments(branches)
    all_points = np.concatenate(branches)
    lengths = np.linalg.norm(ends - starts, axis=1)
    component_diagonals = np.asarray([
        np.linalg.norm(branch.max(axis=0) - branch.min(axis=0))
        for branch in branches
    ])
    component_arc_lengths = np.asarray([
        lengths[component_ids == component_id].sum()
        for component_id in range(len(branches))
    ])
    bbox_min = all_points.min(axis=0)
    bbox_max = all_points.max(axis=0)
    return {
        "branches": branches,
        "starts": starts,
        "ends": ends,
        "lengths": lengths,
        "component_ids": component_ids,
        "component_diagonals": component_diagonals,
        "component_arc_lengths": component_arc_lengths,
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "bbox_width": float(bbox_max[0] - bbox_min[0]),
        "bbox_height": float(bbox_max[1] - bbox_min[1]),
        "bbox_diagonal": float(np.linalg.norm(bbox_max - bbox_min)),
        "arc_length": float(lengths.sum()),
        "component_count": len(branches),
        "largest_component_diagonal": float(component_diagonals.max()),
    }


def nearest_caustic_features(point: np.ndarray, geometry: dict[str, Any]) -> dict[str, float]:
    starts = geometry["starts"]
    vectors = geometry["ends"] - starts
    length2 = np.einsum("ij,ij->i", vectors, vectors)
    projection = np.divide(
        np.einsum("ij,ij->i", point - starts, vectors),
        length2,
        out=np.zeros_like(length2),
        where=length2 > 0.0,
    )
    projection = np.clip(projection, 0.0, 1.0)
    closest = starts + projection[:, None] * vectors
    distance2 = np.einsum("ij,ij->i", point - closest, point - closest)
    index = int(np.argmin(distance2))
    component_id = int(geometry["component_ids"][index])
    segment_length = math.sqrt(max(float(length2[index]), 0.0))
    return {
        "caustic_distance": math.sqrt(max(float(distance2[index]), 0.0)),
        "nearest_segment_length": segment_length,
        "nearest_segment_fraction": float(projection[index]),
        "nearest_component_diagonal": float(geometry["component_diagonals"][component_id]),
        "nearest_component_arc_length": float(geometry["component_arc_lengths"][component_id]),
    }


def source_points(
    case: LensCase,
    geometry: dict[str, Any],
    count: int,
    seed: int,
    include_exact_caustic: bool,
) -> list[tuple[float, float, str, float]]:
    """Sample folds/cusps from both sides plus a small far-field control set."""
    rng = np.random.default_rng(seed + 104729 * (case.case_id + 1))
    points = np.concatenate(geometry["branches"])
    samples: list[tuple[float, float, str, float]] = []
    near_count = max(1, count - max(2, count // 6))
    factors = ((0.0,) + DISTANCE_FACTORS) if include_exact_caustic else DISTANCE_FACTORS
    for i in range(near_count):
        anchor_index = int(rng.integers(len(points)))
        anchor = points[anchor_index]
        factor = factors[i % len(factors)]
        angle = float(rng.uniform(0.0, 2.0 * math.pi))
        offset = factor * case.source_radius * np.array([math.cos(angle), math.sin(angle)])
        sample = anchor + offset
        samples.append((float(sample[0]), float(sample[1]), "caustic_offset", factor))

    bbox_min = geometry["bbox_min"]
    bbox_max = geometry["bbox_max"]
    pad = max(0.25, 50.0 * case.source_radius, geometry["bbox_diagonal"] * 0.25)
    while len(samples) < count:
        sample = rng.uniform(bbox_min - pad, bbox_max + pad)
        samples.append((float(sample[0]), float(sample[1]), "field", math.nan))
    return samples


def _finite_solver(
    bins: int,
    grid: str,
    caustic_bins: int,
    limb_c: float,
):
    # Automatic nbin is intentionally disabled.  Zero hex thresholds
    # reject the fast approximations without making explicit polar mode fall
    # back merely because a huge artificial distance threshold was requested.
    return lcbinint.LightCurve(
        options=lcbinint.Options(
            coordinates="center_of_mass",
            caustic_bins=caustic_bins,
            source_bins=bins,
            polar_source_bins=bins,
            inverse_ray_grid=grid,
            point_source_threshold=0.0,
            hexadecapole_threshold=0.0,
            adaptive_hex_threshold=0.0,
            max_source_bins=bins,
        ),
        limb_darkening=lcbinint.LimbDarkening.linear(limb_c),
    )


def _auto_solver(caustic_bins: int, limb_c: float):
    return lcbinint.LightCurve(
        options=lcbinint.Options(
            coordinates="center_of_mass",
            caustic_bins=caustic_bins,
            nbin="auto",
            inverse_ray_grid="auto",
        ),
        limb_darkening=lcbinint.LimbDarkening.linear(limb_c),
    )


def _lc_worker(
    connection: Any,
    grid: str,
    bins: int,
    caustic_bins: int,
    case: LensCase,
    x: float,
    y: float,
    limb_c: float,
    use_default_switch: bool,
) -> None:
    solver = (
        _auto_solver(caustic_bins, limb_c)
        if use_default_switch
        else _finite_solver(bins, grid, caustic_bins, limb_c)
    )
    started = time.perf_counter_ns()
    try:
        info = solver.info(
            [x], t0=0.0, tE=1.0, u0=y, alpha=0.0,
            s=case.separation, q=case.mass_ratio, rho=case.source_radius,
        )
        elapsed = time.perf_counter_ns() - started
        connection.send({
            "value": float(info.magnifications[0]),
            "elapsed_ns": elapsed,
            "method": info.finite_source_method_names[0],
            "reported_error": float(info.finite_source_error_estimates[0]),
            "reported_converged": bool(info.finite_source_converged[0]),
            "refinement_level": int(info.finite_source_refinement_levels[0]),
            "point_magnification": float(info.point_source_magnifications[0]),
            "quadrupole_indicator": float(info.point_source_quadrupole_indicators[0]),
            "cusp_indicator": float(info.point_source_cusp_indicators[0]),
            "ghost_indicator": float(info.point_source_ghost_indicators[0]),
            "planetary_distance2": float(info.point_source_planetary_distances2[0]),
            "safety_flags": int(info.point_source_safety_flags[0]),
        })
    except Exception as exc:
        connection.send({"error": repr(exc), "elapsed_ns": time.perf_counter_ns() - started})
    finally:
        connection.close()


def _timed_lc(
    grid: str,
    bins: int,
    caustic_bins: int,
    case: LensCase,
    x: float,
    y: float,
    limb_c: float,
    use_default_switch: bool,
    timeout_seconds: float,
) -> dict[str, Any]:
    context = multiprocessing.get_context("fork")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_lc_worker,
        args=(
            child, grid, bins, caustic_bins, case, x, y, limb_c,
            use_default_switch,
        ),
        daemon=True,
    )
    process.start()
    child.close()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(2.0)
        if process.is_alive():
            process.kill()
            process.join()
        parent.close()
        return {"timeout": True, "timeout_seconds": timeout_seconds}
    try:
        return parent.recv() if parent.poll() else {"error": f"worker exited {process.exitcode}"}
    finally:
        parent.close()


def _vbm_worker(
    connection: Any,
    mode: str,
    case: LensCase,
    x: float,
    y: float,
    limb_c: float,
    tolerance: float,
) -> None:
    solver = VBMicrolensing()
    solver.Tol = tolerance
    solver.RelTol = 0.0
    solver.a1 = limb_c
    started = time.perf_counter_ns()
    try:
        if mode == "auto":
            value = float(solver.BinaryMag2(
                case.separation, case.mass_ratio, x, y, case.source_radius
            ))
        elif mode == "contour":
            value = float(solver.BinaryMag(
                case.separation, case.mass_ratio, x, y, case.source_radius, tolerance
            ))
        elif mode == "dark":
            value = float(solver.BinaryMagDark(
                case.separation, case.mass_ratio, x, y, case.source_radius, tolerance
            ))
        else:
            raise ValueError(f"unknown VBM mode: {mode}")
        connection.send({"value": value, "elapsed_ns": time.perf_counter_ns() - started})
    except Exception as exc:
        connection.send({"error": repr(exc), "elapsed_ns": time.perf_counter_ns() - started})
    finally:
        connection.close()


def _timed_vbm_call(
    mode: str,
    case: LensCase,
    x: float,
    y: float,
    limb_c: float,
    tolerance: float,
    timeout_seconds: float,
) -> dict[str, Any]:
    context = multiprocessing.get_context("fork")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_vbm_worker,
        args=(child, mode, case, x, y, limb_c, tolerance),
        daemon=True,
    )
    process.start()
    child.close()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(2.0)
        if process.is_alive():
            process.kill()
            process.join()
        parent.close()
        return {"timeout": True, "timeout_seconds": timeout_seconds}
    try:
        return parent.recv() if parent.poll() else {"error": f"worker exited {process.exitcode}"}
    finally:
        parent.close()


def _timed_vbm(
    case: LensCase,
    x: float,
    y: float,
    limb_c: float,
    tolerance: float,
    timeout_seconds: float,
) -> dict[str, Any]:
    if limb_c == 0.0:
        return {
            mode: _timed_vbm_call(
                mode, case, x, y, limb_c, tolerance, timeout_seconds
            )
            for mode in ("auto", "contour")
        }
    return {
        "dark": _timed_vbm_call(
            "dark", case, x, y, limb_c, tolerance, timeout_seconds
        )
    }


def evaluate_case(case: LensCase, args: argparse.Namespace) -> dict[str, Any]:
    geometry = caustic_geometry(case, args.caustic_bins)
    points = source_points(
        case, geometry, args.points_per_case, args.seed, args.include_exact_caustic
    )
    rows: list[dict[str, Any]] = []
    for point_id, (x, y, sampling, requested_factor) in enumerate(points):
        position = np.asarray([x, y], dtype=float)
        nearest = nearest_caustic_features(position, geometry)
        for limb_c in args.limb_coefficients:
            row: dict[str, Any] = {
                "point_id": point_id,
                "source_x": x,
                "source_y": y,
                "sampling": sampling,
                "requested_distance_factor": requested_factor,
                "limb_c": limb_c,
                **nearest,
            }
            row["caustic_distance_over_rho"] = nearest["caustic_distance"] / case.source_radius
            row["rho_over_caustic_diagonal"] = (
                case.source_radius / max(geometry["bbox_diagonal"], 1e-300)
            )
            row["rho_over_nearest_component_diagonal"] = (
                case.source_radius / max(nearest["nearest_component_diagonal"], 1e-300)
            )
            row["lc_auto"] = _timed_lc(
                "auto", 50, args.caustic_bins, case, x, y, limb_c, True,
                args.lc_timeout,
            )
            row["vbm"] = _timed_vbm(
                case, x, y, limb_c, args.vbm_tolerance, args.vbm_timeout
            )
            sequences: dict[str, list[dict[str, Any]]] = {}
            for grid in args.grids:
                values = []
                lower_timeout_bins: int | None = None
                for bins in args.bins:
                    if lower_timeout_bins is None:
                        result = _timed_lc(
                            grid, bins, args.caustic_bins, case, x, y, limb_c,
                            False, args.lc_timeout,
                        )
                        if result.get("timeout"):
                            lower_timeout_bins = bins
                    else:
                        # Cost grows with resolution.  Once a lower nbin has
                        # exceeded the wall-time budget, larger grids provide
                        # no useful auto-resolution candidate in this sweep.
                        # Retain explicit censoring rather than spending every
                        # timeout again; extreme cases are handled separately.
                        result = {
                            "skipped_after_timeout": True,
                            "lower_timeout_bins": lower_timeout_bins,
                        }
                    result["bins"] = bins
                    values.append(result)
                sequences[grid] = values
            row["lc_fixed_sequences"] = sequences
            rows.append(row)

    public_geometry = {
        key: geometry[key]
        for key in (
            "bbox_width", "bbox_height", "bbox_diagonal", "arc_length",
            "component_count", "largest_component_diagonal",
        )
    }
    return {
        "schema_version": 1,
        "case": asdict(case),
        "caustic": public_geometry,
        "configuration": {
            "bins": list(args.bins),
            "grids": list(args.grids),
            "limb_coefficients": list(args.limb_coefficients),
            "caustic_bins": args.caustic_bins,
            "vbm_tolerance": args.vbm_tolerance,
            "vbm_timeout": args.vbm_timeout,
            "lc_timeout": args.lc_timeout,
            "seed": args.seed,
            "include_exact_caustic": args.include_exact_caustic,
        },
        "rows": rows,
    }


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, allow_nan=True, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = make_lens_cases(args.lens_cases, args.seed)
    assigned = [case for case in cases if case.case_id % args.shard_count == args.shard_index]
    completed = 0
    started = time.monotonic()
    for case in assigned:
        destination = args.output_dir / f"case-{case.case_id:06d}.json"
        if destination.exists() and not args.overwrite:
            completed += 1
            continue
        case_started = time.monotonic()
        try:
            result = evaluate_case(case, args)
        except Exception as exc:
            result = {"schema_version": 1, "case": asdict(case), "error": repr(exc)}
        result["case_elapsed_seconds"] = time.monotonic() - case_started
        _atomic_json(destination, result)
        completed += 1
        if completed % args.progress_every == 0 or completed == len(assigned):
            elapsed = time.monotonic() - started
            print(
                f"shard={args.shard_index}/{args.shard_count} "
                f"cases={completed}/{len(assigned)} elapsed_s={elapsed:.1f}",
                flush=True,
            )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lens-cases", type=int, default=256)
    parser.add_argument("--points-per-case", type=int, default=12)
    parser.add_argument("--bins", type=lambda text: tuple(int(x) for x in text.split(",")), default=DEFAULT_BINS)
    parser.add_argument("--grids", type=lambda text: tuple(text.split(",")), default=("cartesian", "polar"))
    parser.add_argument("--limb-coefficients", type=lambda text: tuple(float(x) for x in text.split(",")), default=(0.0, 0.5, 0.8))
    parser.add_argument("--caustic-bins", type=int, default=1200)
    parser.add_argument("--vbm-tolerance", type=float, default=1e-5)
    parser.add_argument("--vbm-timeout", type=float, default=5.0)
    parser.add_argument("--lc-timeout", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-exact-caustic", action="store_true")
    args = parser.parse_args()
    if args.lens_cases <= 0 or args.points_per_case <= 0:
        parser.error("lens and point counts must be positive")
    if not args.bins or any(value <= 0 for value in args.bins):
        parser.error("bins must be positive")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        parser.error("invalid shard index/count")
    if any(grid not in {"cartesian", "polar"} for grid in args.grids):
        parser.error("grids must be cartesian and/or polar")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
