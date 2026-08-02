#!/usr/bin/env python3
"""Stratified accuracy, gradient, and dispatcher calibration for JAX kernels.

The sweep deliberately samples offsets from binary caustics and a far-field
control for each lens.  It records every backend separately so a dispatcher
mistake can be distinguished from a quadrature or support failure.  Native
Cartesian/polar agreement supplies the reference; no single external engine
is treated as an oracle.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import lcbinint  # noqa: E402
from lcbinint_jax import (  # noqa: E402
    binary_hexadecapole,
    binary_inverse_ray,
    binary_inverse_ray_polar,
    binary_magnification_auto,
    binary_magnification_calibrated,
)

PROFILE_COEFFICIENTS = {
    "uniform": (0.0, 0.0, "uniform"),
    "linear": (0.4, 0.0, "linear"),
    "square_root": (0.3, 0.2, "two_coefficient"),
}
CAUSTIC_DISTANCE_FACTORS = (0.5, 1.0, 2.0, 5.0, 20.0)
ANCHORS = (
    (0.55, 1.0, 3.0e-3),
    (0.75, 0.1, 1.0e-2),
    (0.90, 1.0e-3, 3.0e-3),
    (0.98, 1.0e-5, 3.0e-4),
    (1.00, 1.0e-3, 1.0e-4),
    (1.05, 1.0e-2, 5.0e-3),
    (1.20, 0.1, 2.0e-2),
    (1.50, 1.0, 1.0e-2),
    (1.80, 1.0e-3, 1.0e-3),
    (2.50, 1.0e-5, 1.0e-4),
)


@dataclass(frozen=True)
class LensCase:
    case_id: int
    separation: float
    mass_ratio: float
    source_radius: float


def make_cases(count: int, seed: int) -> list[LensCase]:
    rng = np.random.default_rng(seed)
    cases = [
        LensCase(index, *parameters) for index, parameters in enumerate(ANCHORS[:count])
    ]
    while len(cases) < count:
        index = len(cases)
        cases.append(
            LensCase(
                index,
                float(10.0 ** rng.uniform(math.log10(0.4), math.log10(3.0))),
                float(10.0 ** rng.uniform(-6.0, 0.0)),
                float(10.0 ** rng.uniform(-5.0, -1.0)),
            )
        )
    return cases


def caustic_branches(case: LensCase, bins: int) -> list[np.ndarray]:
    solver = lcbinint.LightCurve(
        options=lcbinint.Options(coordinates="center_of_mass", caustic_bins=bins)
    )
    geometry = solver.caustics(s=case.separation, q=case.mass_ratio, n_points=bins)
    return [
        np.column_stack((np.asarray(xs), np.asarray(ys)))
        for xs, ys in zip(geometry.x, geometry.y)
        if len(xs) >= 3
    ]


def source_points(
    case: LensCase,
    branches: list[np.ndarray],
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed + 7919 * (case.case_id + 1))
    all_points = np.concatenate(branches)
    samples: list[dict[str, Any]] = []
    near_count = max(1, count - 1)
    for point_id in range(near_count):
        branch = branches[point_id % len(branches)]
        index = int(rng.integers(0, len(branch)))
        previous = branch[(index - 1) % len(branch)]
        following = branch[(index + 1) % len(branch)]
        tangent = following - previous
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm == 0.0:
            normal = np.asarray((1.0, 0.0))
        else:
            normal = np.asarray((-tangent[1], tangent[0])) / tangent_norm
        factor = CAUSTIC_DISTANCE_FACTORS[point_id % len(CAUSTIC_DISTANCE_FACTORS)]
        side = -1.0 if point_id % 2 else 1.0
        point = branch[index] + side * factor * case.source_radius * normal
        samples.append(
            {
                "point_id": point_id,
                "source_x": float(point[0]),
                "source_y": float(point[1]),
                "sampling": "caustic_normal",
                "requested_distance_over_rho": factor,
            }
        )

    bbox_max = all_points.max(axis=0)
    pad = max(0.3, 30.0 * case.source_radius)
    far = bbox_max + np.asarray((pad, 0.7 * pad))
    samples.append(
        {
            "point_id": len(samples),
            "source_x": float(far[0]),
            "source_y": float(far[1]),
            "sampling": "field",
            "requested_distance_over_rho": math.nan,
        }
    )
    return samples


def _native_solver(
    case: LensCase,
    limb_c: float,
    limb_d: float,
    grid: str,
    bins: int,
) -> lcbinint.LightCurve:
    return lcbinint.LightCurve(
        options=lcbinint.Options(
            coordinates="center_of_mass",
            inverse_ray_grid=grid,
            source_bins=bins,
            polar_source_bins=bins,
            point_source_threshold=1.0e9,
            hexadecapole_threshold=1.0e9,
            adaptive_hex_threshold=0.0,
            max_source_bins=bins,
        ),
        limb_darkening=lcbinint.LimbDarkening(c=limb_c, d=limb_d),
    )


def native_value(
    case: LensCase,
    source_x: float,
    source_y: float,
    limb_c: float,
    limb_d: float,
    grid: str,
    bins: int,
) -> dict[str, Any]:
    solver = _native_solver(case, limb_c, limb_d, grid, bins)
    started = time.perf_counter()
    info = solver.info(
        [source_x],
        t0=0.0,
        tE=1.0,
        u0=source_y,
        alpha=0.0,
        s=case.separation,
        q=case.mass_ratio,
        rho=case.source_radius,
    )
    return {
        "value": float(info.magnifications[0]),
        "method": info.finite_source_method_names[0],
        "milliseconds": 1.0e3 * (time.perf_counter() - started),
    }


def reference_value(
    case: LensCase,
    source_x: float,
    source_y: float,
    limb_c: float,
    limb_d: float,
    coarse_bins: int,
    fine_bins: int,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    cart_coarse = native_value(
        case, source_x, source_y, limb_c, limb_d, "cartesian", coarse_bins
    )
    cart_fine = native_value(
        case, source_x, source_y, limb_c, limb_d, "cartesian", fine_bins
    )
    polar_fine = native_value(
        case, source_x, source_y, limb_c, limb_d, "polar", fine_bins
    )
    candidates = np.asarray((cart_fine["value"], polar_fine["value"]))
    value = float(np.median(candidates))
    budget = absolute_tolerance + relative_tolerance * max(abs(value), 1.0)
    cart_change = abs(cart_fine["value"] - cart_coarse["value"])
    grid_spread = float(np.ptp(candidates))
    trusted = (
        np.all(np.isfinite(candidates))
        and cart_change <= 2.0 * budget
        and grid_spread <= 2.0 * budget
    )
    return {
        "trusted": bool(trusted),
        "value": value,
        "budget": budget,
        "cartesian_change": cart_change,
        "grid_spread": grid_spread,
        "cartesian_coarse": cart_coarse,
        "cartesian_fine": cart_fine,
        "polar_fine": polar_fine,
    }


def timed_jax(function: Callable[[], Any], repeat: int) -> tuple[Any, float]:
    first = function()
    jax.block_until_ready(first)
    samples = []
    result = first
    for _ in range(max(1, repeat)):
        started = time.perf_counter()
        result = function()
        jax.block_until_ready(result)
        samples.append(1.0e3 * (time.perf_counter() - started))
    return result, float(np.median(samples))


def jax_backends(
    case: LensCase,
    source_x: float,
    source_y: float,
    limb_c: float,
    limb_d: float,
    moment_mode: str,
    absolute_tolerance: float,
    relative_tolerance: float,
    repeat: int,
) -> dict[str, Any]:
    parameters = (
        source_x,
        source_y,
        case.separation,
        case.mass_ratio,
        case.source_radius,
        limb_c,
        limb_d,
    )
    hex_result, hex_ms = timed_jax(lambda: binary_hexadecapole(*parameters), repeat)
    cart_coarse, cart_coarse_ms = timed_jax(
        lambda: binary_inverse_ray(
            *parameters,
            resolution=64,
            tile_size=16,
            tile_capacity=1024,
            limb_samples=16,
        ),
        repeat,
    )
    cart_fine, cart_fine_ms = timed_jax(
        lambda: binary_inverse_ray(
            *parameters,
            resolution=128,
            tile_size=16,
            tile_capacity=4096,
            limb_samples=32,
        ),
        repeat,
    )
    polar_coarse, polar_coarse_ms = timed_jax(
        lambda: binary_inverse_ray_polar(
            *parameters,
            resolution=64,
            angular_bins=4096,
            radial_capacity=256,
            band_capacity=4,
            limb_samples=32,
            angular_chunk_size=1024,
            moment_mode=moment_mode,
        ),
        repeat,
    )
    polar_fine, polar_fine_ms = timed_jax(
        lambda: binary_inverse_ray_polar(
            *parameters,
            resolution=128,
            angular_bins=8192,
            radial_capacity=512,
            band_capacity=4,
            limb_samples=64,
            angular_chunk_size=1024,
            moment_mode=moment_mode,
        ),
        repeat,
    )
    auto_result, auto_ms = timed_jax(
        lambda: binary_magnification_auto(
            *parameters,
            resolution=64,
            tile_capacity=1024,
            limb_samples=16,
            polar_resolution=64,
            polar_angular_bins=4096,
            polar_radial_capacity=256,
            polar_limb_samples=32,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            moment_mode=moment_mode,
        ),
        repeat,
    )
    calibrated_result, calibrated_ms = timed_jax(
        lambda: binary_magnification_calibrated(
            *parameters,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            maximum_source_bins=400,
            moment_mode=moment_mode,
        ),
        repeat,
    )

    def inverse_record(result: Any, milliseconds: float) -> dict[str, Any]:
        return {
            "value": float(result.magnification),
            "milliseconds": milliseconds,
            "support_valid": bool(result.support_valid),
            "overflow": bool(result.discovery_overflow),
            "root_failure": bool(result.root_failure),
            "support_count": int(result.tile_count),
        }

    return {
        "hexadecapole": {
            "value": float(hex_result.magnification),
            "point_magnification": float(hex_result.point_magnification),
            "quadrupole_correction": float(hex_result.quadrupole_correction),
            "hexadecapole_correction": float(hex_result.hexadecapole_correction),
            "estimated_error": float(hex_result.estimated_error),
            "topology_stable": bool(hex_result.topology_stable),
            "root_failure": bool(hex_result.root_failure),
            "milliseconds": hex_ms,
        },
        "cartesian_coarse": inverse_record(cart_coarse, cart_coarse_ms),
        "cartesian_fine": inverse_record(cart_fine, cart_fine_ms),
        "polar_coarse": inverse_record(polar_coarse, polar_coarse_ms),
        "polar_fine": inverse_record(polar_fine, polar_fine_ms),
        "auto": {
            "value": float(auto_result.magnification),
            "milliseconds": auto_ms,
            "method": int(auto_result.method),
            "estimated_error": float(auto_result.estimated_error),
            "support_valid": bool(auto_result.support_valid),
            "used_multipole": bool(auto_result.used_multipole),
            "used_polar": bool(auto_result.used_polar),
            "used_source_plane": bool(auto_result.used_source_plane),
            "used_expanded_cartesian": bool(auto_result.used_expanded_cartesian),
        },
        "calibrated": {
            "value": float(calibrated_result.magnification),
            "milliseconds": calibrated_ms,
            "method": int(calibrated_result.method),
            "estimated_error": float(calibrated_result.estimated_error),
            "support_valid": bool(calibrated_result.support_valid),
            "selected_source_bins": int(calibrated_result.selected_source_bins),
            "comparison_resolution": int(
                calibrated_result.comparison_resolution
            ),
            "executed_resolution": int(calibrated_result.executed_resolution),
            "tile_capacity": int(calibrated_result.tile_capacity),
            "caustic_distance": float(calibrated_result.caustic_distance),
            "prefer_polar": bool(calibrated_result.prefer_polar),
            "point_safe": bool(calibrated_result.point_safe),
            "chord_band": bool(calibrated_result.chord_band),
            "tangent_band": bool(calibrated_result.tangent_band),
            "grazing_ring_band": bool(calibrated_result.grazing_ring_band),
            "value_error": float(calibrated_result.value_error),
            "value_budget": float(calibrated_result.value_budget),
            "value_converged": bool(calibrated_result.value_converged),
            "used_multipole": bool(calibrated_result.used_multipole),
            "used_polar": bool(calibrated_result.used_polar),
            "used_source_plane": bool(calibrated_result.used_source_plane),
        },
    }


def gradient_check(
    case: LensCase,
    source_x: float,
    source_y: float,
    limb_c: float,
    limb_d: float,
    moment_mode: str,
    fine_bins: int,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    step = max(1.0e-5, 3.0e-2 * case.source_radius)

    def jax_function(active_x):
        return binary_magnification_auto(
            active_x,
            source_y,
            case.separation,
            case.mass_ratio,
            case.source_radius,
            limb_c,
            limb_d,
            resolution=64,
            tile_capacity=1024,
            limb_samples=16,
            polar_resolution=64,
            polar_angular_bins=4096,
            polar_radial_capacity=256,
            polar_limb_samples=32,
            moment_mode=moment_mode,
        ).magnification

    def calibrated_function(active_x):
        return binary_magnification_calibrated(
            active_x,
            source_y,
            case.separation,
            case.mass_ratio,
            case.source_radius,
            limb_c,
            limb_d,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            maximum_source_bins=400,
            moment_mode=moment_mode,
        ).magnification

    jax_auto_gradient = float(jax.grad(jax_function)(source_x))
    jax_calibrated_gradient = float(jax.grad(calibrated_function)(source_x))

    def native_gradient(bins):
        plus = native_value(
            case,
            source_x + step,
            source_y,
            limb_c,
            limb_d,
            "cartesian",
            bins,
        )["value"]
        minus = native_value(
            case,
            source_x - step,
            source_y,
            limb_c,
            limb_d,
            "cartesian",
            bins,
        )["value"]
        return (plus - minus) / (2.0 * step)

    native_gradients = np.asarray(
        (
            native_gradient(fine_bins // 2),
            native_gradient(fine_bins),
            native_gradient(2 * fine_bins),
        )
    )
    reference_gradient = float(np.median(native_gradients))
    budget = 1.0e-3 + 5.0e-3 * max(abs(reference_gradient), 1.0)
    reference_spread = float(np.ptp(native_gradients))
    reference_trusted = reference_spread <= 2.0 * budget
    auto_error = abs(jax_auto_gradient - reference_gradient)
    calibrated_error = abs(jax_calibrated_gradient - reference_gradient)
    return {
        "step": step,
        "jax": jax_auto_gradient,
        "jax_auto": jax_auto_gradient,
        "jax_calibrated": jax_calibrated_gradient,
        "reference": reference_gradient,
        "reference_sequence": native_gradients.tolist(),
        "reference_spread": reference_spread,
        "absolute_error": auto_error,
        "auto_absolute_error": auto_error,
        "calibrated_absolute_error": calibrated_error,
        "budget": budget,
        "passes": bool(
            reference_trusted and calibrated_error <= budget
        ),
        "auto_passes": bool(reference_trusted and auto_error <= budget),
        "calibrated_passes": bool(
            reference_trusted and calibrated_error <= budget
        ),
        "reference_trusted": bool(reference_trusted),
    }


def candidate_dispatch(
    row: dict[str, Any],
    multipole_safety_factor: float,
    polar_threshold: float,
    polar_max_rho: float,
    polar_min_mass_ratio: float,
) -> tuple[str, dict[str, Any]]:
    data = row["jax"]
    hex_result = data["hexadecapole"]
    reference = row["reference"]
    correction_scale = max(
        abs(hex_result["quadrupole_correction"]), reference["budget"]
    )
    ordered = hex_result["estimated_error"] <= 0.25 * correction_scale and abs(
        hex_result["quadrupole_correction"]
    ) <= 0.1 * max(abs(hex_result["value"]), 1.0)
    accept_hex = (
        hex_result["topology_stable"]
        and not hex_result["root_failure"]
        and ordered
        and multipole_safety_factor * hex_result["estimated_error"]
        <= reference["budget"]
    )
    if accept_hex:
        return "hexadecapole", hex_result
    use_polar = (
        min(
            row["case"]["mass_ratio"],
            1.0 / row["case"]["mass_ratio"],
        )
        >= polar_min_mass_ratio
        and hex_result["topology_stable"]
        and row["case"]["source_radius"] <= polar_max_rho
        and hex_result["point_magnification"] >= polar_threshold
    )
    cart = data["cartesian_coarse"]
    if use_polar:
        return "polar", data["polar_coarse"]
    return "cartesian", cart


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    trusted = [row for row in rows if row["reference"]["trusted"]]
    method_names = {
        0: "hexadecapole",
        1: "cartesian",
        2: "polar",
        3: "source_plane",
        4: "point_source",
    }
    default_failures = [
        row
        for row in trusted
        if (
            not row["jax"]["auto"]["support_valid"]
            or abs(row["jax"]["auto"]["value"] - row["reference"]["value"])
            > row["reference"]["budget"]
        )
    ]
    calibrated_failures = [
        row
        for row in trusted
        if (
            not row["jax"]["calibrated"]["support_valid"]
            or not row["jax"]["calibrated"]["value_converged"]
            or abs(
                row["jax"]["calibrated"]["value"] - row["reference"]["value"]
            )
            > row["reference"]["budget"]
        )
    ]
    calibration = []
    for safety_factor in (2.0, 4.0, 8.0, 16.0):
        for polar_threshold in (30.0, 50.0, 80.0, 120.0, 200.0):
            for polar_max_rho in (0.003, 0.01, 0.03):
                for polar_min_mass_ratio in (0.0, 1.0e-3, 5.0e-3, 1.0e-2):
                    invalid = 0
                    accuracy_failures = 0
                    times = []
                    methods: dict[str, int] = {}
                    for row in trusted:
                        method, result = candidate_dispatch(
                            row,
                            safety_factor,
                            polar_threshold,
                            polar_max_rho,
                            polar_min_mass_ratio,
                        )
                        methods[method] = methods.get(method, 0) + 1
                        times.append(result["milliseconds"])
                        result_invalid = not math.isfinite(result["value"]) or (
                            "support_valid" in result and not result["support_valid"]
                        )
                        invalid += result_invalid
                        accuracy_failures += (
                            not result_invalid
                            and abs(result["value"] - row["reference"]["value"])
                            > row["reference"]["budget"]
                        )
                    calibration.append(
                        {
                            "multipole_safety_factor": safety_factor,
                            "polar_threshold": polar_threshold,
                            "polar_max_rho": polar_max_rho,
                            "polar_min_mass_ratio": polar_min_mass_ratio,
                            "invalid": invalid,
                            "accuracy_failures": accuracy_failures,
                            "median_milliseconds": (
                                float(np.median(times)) if times else math.nan
                            ),
                            "method_counts": methods,
                        }
                    )
    calibration.sort(
        key=lambda item: (
            item["accuracy_failures"],
            item["invalid"],
            item["median_milliseconds"],
        )
    )
    gradients = [
        row["gradient"]
        for row in rows
        if row.get("gradient", {}).get("reference_trusted")
    ]
    return {
        "rows": len(rows),
        "trusted_rows": len(trusted),
        "default_failures": len(default_failures),
        "default_failure_rows": [
            {
                "case_id": row["case"]["case_id"],
                "point_id": row["point"]["point_id"],
                "profile": row["profile"],
                "method": method_names.get(row["jax"]["auto"]["method"], "unknown"),
                "absolute_error": abs(
                    row["jax"]["auto"]["value"] - row["reference"]["value"]
                ),
                "budget": row["reference"]["budget"],
            }
            for row in default_failures[:20]
        ],
        "calibrated_failures": len(calibrated_failures),
        "calibrated_failure_rows": [
            {
                "case_id": row["case"]["case_id"],
                "point_id": row["point"]["point_id"],
                "profile": row["profile"],
                "method": method_names.get(
                    row["jax"]["calibrated"]["method"], "unknown"
                ),
                "absolute_error": abs(
                    row["jax"]["calibrated"]["value"]
                    - row["reference"]["value"]
                ),
                "budget": row["reference"]["budget"],
                "support_valid": row["jax"]["calibrated"]["support_valid"],
                "value_converged": row["jax"]["calibrated"]["value_converged"],
                "selected_source_bins": row["jax"]["calibrated"][
                    "selected_source_bins"
                ],
            }
            for row in calibrated_failures[:20]
        ],
        "gradient_rows": len(gradients),
        "gradient_failures": sum(not item["passes"] for item in gradients),
        "auto_gradient_failures": sum(
            not item["auto_passes"] for item in gradients
        ),
        "calibrated_gradient_failures": sum(
            not item["calibrated_passes"] for item in gradients
        ),
        "best_candidates": calibration[:12],
    }


def run(args: argparse.Namespace) -> int:
    profiles = tuple(args.profiles.split(","))
    unknown = set(profiles) - set(PROFILE_COEFFICIENTS)
    if unknown:
        raise ValueError(f"unknown profiles: {sorted(unknown)}")
    cases = make_cases(args.lens_cases, args.seed)
    configuration = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in {"output", "resume"}
    }
    rows: list[dict[str, Any]] = []
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text())
        if previous.get("schema_version") != 2:
            raise ValueError("resume requires a schema_version=2 sweep")
        if previous.get("configuration") != configuration:
            raise ValueError("resume configuration does not match the existing sweep")
        rows = list(previous.get("rows", ()))
    completed = {
        (
            row["case"]["case_id"],
            row["point"]["point_id"],
            row["profile"],
        )
        for row in rows
    }

    def persist() -> None:
        output = {
            "schema_version": 2,
            "configuration": configuration,
            "rows": rows,
            "summary": summarize(rows),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.tmp")
        temporary.write_text(json.dumps(output, indent=2, allow_nan=True) + "\n")
        temporary.replace(args.output)

    for case in cases:
        branches = caustic_branches(case, args.caustic_bins)
        points = source_points(case, branches, args.points_per_case, args.seed)
        for point in points:
            for profile_index, profile in enumerate(profiles):
                row_key = (case.case_id, point["point_id"], profile)
                if row_key in completed:
                    continue
                limb_c, limb_d, moment_mode = PROFILE_COEFFICIENTS[profile]
                reference = reference_value(
                    case,
                    point["source_x"],
                    point["source_y"],
                    limb_c,
                    limb_d,
                    args.native_coarse_bins,
                    args.native_fine_bins,
                    args.absolute_tolerance,
                    args.relative_tolerance,
                )
                jax_results = jax_backends(
                    case,
                    point["source_x"],
                    point["source_y"],
                    limb_c,
                    limb_d,
                    moment_mode,
                    args.absolute_tolerance,
                    args.relative_tolerance,
                    args.repeat,
                )
                row = {
                    "case": asdict(case),
                    "point": point,
                    "profile": profile,
                    "limb_c": limb_c,
                    "limb_d": limb_d,
                    "reference": reference,
                    "jax": jax_results,
                }
                if (
                    args.gradient_stride > 0
                    and point["sampling"] == "field"
                    and case.case_id % args.gradient_stride == 0
                    and profile_index == case.case_id % len(profiles)
                ):
                    row["gradient"] = gradient_check(
                        case,
                        point["source_x"],
                        point["source_y"],
                        limb_c,
                        limb_d,
                        moment_mode,
                        max(4 * args.native_fine_bins, 512),
                        args.absolute_tolerance,
                        args.relative_tolerance,
                    )
                rows.append(row)
                completed.add(row_key)
                persist()
        print(
            f"processed case {case.case_id + 1}/{len(cases)} rows={len(rows)}",
            flush=True,
        )

    persist()
    output = json.loads(args.output.read_text())
    print(json.dumps(output["summary"], indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lens-cases", type=int, default=10)
    parser.add_argument("--points-per-case", type=int, default=6)
    parser.add_argument("--profiles", default="uniform,linear,square_root")
    parser.add_argument("--caustic-bins", type=int, default=512)
    parser.add_argument("--native-coarse-bins", type=int, default=128)
    parser.add_argument("--native-fine-bins", type=int, default=256)
    parser.add_argument("--absolute-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--relative-tolerance", type=float, default=1.0e-4)
    parser.add_argument(
        "--gradient-stride",
        type=int,
        default=1,
        help="check one rotating field profile every N lens cases; 0 disables",
    )
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.lens_cases <= 0 or args.points_per_case <= 1:
        parser.error("lens cases must be positive and points per case > 1")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
