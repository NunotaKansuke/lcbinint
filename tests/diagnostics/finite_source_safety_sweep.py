#!/usr/bin/env python3
"""Stratified validation of binary finite-source method selection.

The sweep samples the documented binary-lens range in log space, concentrates
source positions around caustics, and compares every result with a tighter
VBMicrolensing uniform-source calculation.  Its primary failure criterion is a
point-source or hexadecapole fast path whose absolute error exceeds the
requested tolerance.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import lcbinint
from VBMicrolensing import VBMicrolensing


FAST_METHODS = {"point_source", "hexadecapole"}
DISTANCE_FACTORS = np.asarray(
    [0.5, 0.8, 1.0, 1.3, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0],
    dtype=float,
)


@dataclass(frozen=True)
class LensCase:
    separation: float
    mass_ratio: float
    source_radius: float
    tolerance: float


@dataclass(frozen=True)
class SampleResult:
    error: float
    point_source_error: float
    method: str
    separation: float
    mass_ratio: float
    source_radius: float
    tolerance: float
    source_x: float
    source_y: float
    magnification: float
    reference: float
    safety_flags: int
    quadrupole_indicator: float
    cusp_indicator: float
    ghost_indicator: float
    planetary_distance2: float
    converged: bool


def _log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(10.0 ** rng.uniform(math.log10(low), math.log10(high)))


def lens_cases(count: int, rng: np.random.Generator) -> list[LensCase]:
    # Boundary and topology anchors are deliberately paired rather than taking
    # the full Cartesian product; random log-space cases fill the interior.
    separations = [0.1, 0.3, 0.67, 1.0, 1.5, 3.0, 4.0]
    mass_ratios = [1.0e-6, 1.0e-4, 1.0e-2, 0.1, 1.0]
    source_radii = [1.0e-4, 1.0e-3, 1.0e-2, 0.1]
    tolerances = [1.0e-4, 1.0e-3, 1.0e-2]
    anchors = [
        LensCase(
            separations[i % len(separations)],
            mass_ratios[(2 * i + i // len(mass_ratios)) % len(mass_ratios)],
            source_radii[(3 * i + 1) % len(source_radii)],
            tolerances[(2 * i) % len(tolerances)],
        )
        for i in range(min(count, 36))
    ]
    while len(anchors) < count:
        anchors.append(
            LensCase(
                _log_uniform(rng, 0.1, 4.0),
                _log_uniform(rng, 1.0e-6, 1.0),
                _log_uniform(rng, 1.0e-4, 0.1),
                _log_uniform(rng, 1.0e-4, 1.0e-2),
            )
        )
    return anchors


def source_samples(
    case: LensCase,
    points_per_case: int,
    rng: np.random.Generator,
) -> list[tuple[float, float]]:
    geometry = lcbinint.LightCurve(
        options=lcbinint.Options(coordinates="center_of_mass")
    )
    caustics = geometry.caustics(
        s=case.separation,
        q=case.mass_ratio,
        n_points=600,
    )
    points = np.asarray(
        [
            (x, y)
            for branch_x, branch_y in zip(caustics.x, caustics.y)
            for x, y in zip(branch_x, branch_y)
        ],
        dtype=float,
    )
    samples: list[tuple[float, float]] = []
    near_count = max(points_per_case - 2, 1)
    for i in range(near_count):
        caustic = points[int(rng.integers(len(points)))]
        factor = DISTANCE_FACTORS[i % len(DISTANCE_FACTORS)]
        angle = rng.uniform(0.0, 2.0 * math.pi)
        radius = factor * case.source_radius
        samples.append(
            (
                float(caustic[0] + radius * math.cos(angle)),
                float(abs(caustic[1] + radius * math.sin(angle))),
            )
        )

    x_pad = max(0.5, 50.0 * case.source_radius)
    for _ in range(points_per_case - near_count):
        samples.append(
            (
                float(rng.uniform(points[:, 0].min() - x_pad, points[:, 0].max() + x_pad)),
                float(rng.uniform(0.0, max(0.5, points[:, 1].max() + x_pad))),
            )
        )
    return samples


def make_solvers(case: LensCase):
    options = lcbinint.Options(
        coordinates="center_of_mass",
        caustic_bins=600,
        source_bins=40,
        adaptive_source_bins=1,
        max_source_bins=240,
        tol=case.tolerance,
        hex_tol=case.tolerance,
        inverse_ray_grid="cartesian",
    )
    light_curve = lcbinint.LightCurve(options=options)
    reference_solver = VBMicrolensing()
    reference_solver.a1 = 0.0
    reference_tolerance = min(1.0e-6, case.tolerance / 30.0)
    reference_solver.Tol = reference_tolerance
    return light_curve, reference_solver, reference_tolerance


def evaluate(
    case: LensCase,
    source_x: float,
    source_y: float,
    light_curve,
    reference_solver,
    reference_tolerance: float,
) -> SampleResult:
    info = light_curve.info(
        [source_x],
        t0=0.0,
        tE=1.0,
        u0=source_y,
        alpha=0.0,
        s=case.separation,
        q=case.mass_ratio,
        rho=case.source_radius,
    )
    reference = float(
        reference_solver.BinaryMagDark(
            case.separation,
            case.mass_ratio,
            source_x,
            source_y,
            case.source_radius,
            reference_tolerance,
        )
    )
    return SampleResult(
        error=abs(float(info.magnifications[0]) - reference),
        point_source_error=abs(float(info.point_source_magnifications[0]) - reference),
        method=info.finite_source_method_names[0],
        separation=case.separation,
        mass_ratio=case.mass_ratio,
        source_radius=case.source_radius,
        tolerance=case.tolerance,
        source_x=source_x,
        source_y=source_y,
        magnification=float(info.magnifications[0]),
        reference=reference,
        safety_flags=int(info.point_source_safety_flags[0]),
        quadrupole_indicator=float(info.point_source_quadrupole_indicators[0]),
        cusp_indicator=float(info.point_source_cusp_indicators[0]),
        ghost_indicator=float(info.point_source_ghost_indicators[0]),
        planetary_distance2=float(info.point_source_planetary_distances2[0]),
        converged=bool(info.finite_source_converged[0]),
    )


def coefficient_grid(results: list[SampleResult]) -> list[dict]:
    rows = []
    reference_safe = sum(item.point_source_error <= item.tolerance for item in results)
    for quadrupole_cusp_coefficient in [2.0, 3.0, 4.0, 6.0, 8.0]:
        for ghost_coefficient in [1.0, 2.0, 3.0, 4.0]:
            for planetary_coefficient in [1.0, 2.0, 3.0, 4.0]:
                accepted = []
                for item in results:
                    safety_radius = item.source_radius + 1.0e-3
                    local_safe = (
                        quadrupole_cusp_coefficient
                        * (item.quadrupole_indicator + item.cusp_indicator)
                        * safety_radius**2
                        < item.tolerance
                    )
                    ghost_safe = (
                        item.ghost_indicator == 0.0
                        or ghost_coefficient * safety_radius * item.ghost_indicator < 1.0
                    )
                    q = min(item.mass_ratio, 1.0 / item.mass_ratio)
                    planetary_safe = q >= 0.01 or item.planetary_distance2 > (
                        planetary_coefficient
                        * (item.source_radius**2 + 9.0 * q / item.separation**2)
                    )
                    if local_safe and ghost_safe and planetary_safe:
                        accepted.append(item)
                failures = [
                    item for item in accepted
                    if item.point_source_error > item.tolerance
                ]
                worst_failure = max(
                    failures,
                    key=lambda item: item.point_source_error / item.tolerance,
                    default=None,
                )
                true_accepts = len(accepted) - len(failures)
                rows.append({
                    "quadrupole_cusp_coefficient": quadrupole_cusp_coefficient,
                    "ghost_coefficient": ghost_coefficient,
                    "planetary_coefficient": planetary_coefficient,
                    "accepted": len(accepted),
                    "failures": len(failures),
                    "recall": true_accepts / reference_safe if reference_safe else 1.0,
                    "max_error_ratio": max(
                        (item.point_source_error / item.tolerance for item in accepted),
                        default=0.0,
                    ),
                    "worst_failure": (
                        {
                            key: asdict(worst_failure)[key]
                            for key in [
                                "separation", "mass_ratio", "source_radius",
                                "tolerance", "source_x", "source_y",
                                "point_source_error", "ghost_indicator",
                            ]
                        }
                        if worst_failure is not None else None
                    ),
                })
    return rows


def run(args: argparse.Namespace) -> dict:
    rng = np.random.default_rng(args.seed)
    methods: Counter[str] = Counter()
    safety_flags: Counter[int] = Counter()
    results: list[SampleResult] = []
    failures: list[dict] = []
    evaluation_errors: list[dict] = []

    cases = lens_cases(args.lens_cases, rng)
    for case_index, case in enumerate(cases, start=1):
        light_curve, reference_solver, reference_tolerance = make_solvers(case)
        for source_x, source_y in source_samples(case, args.points_per_case, rng):
            try:
                result = evaluate(
                    case,
                    source_x,
                    source_y,
                    light_curve,
                    reference_solver,
                    reference_tolerance,
                )
            except Exception as exc:  # diagnostic script: retain the full case
                evaluation_errors.append(
                    {**asdict(case), "source_x": source_x, "source_y": source_y,
                     "error": repr(exc)}
                )
                continue
            results.append(result)
            methods[result.method] += 1
            safety_flags[result.safety_flags] += 1
            if result.method in FAST_METHODS and result.error > result.tolerance:
                failures.append(asdict(result))
        if args.progress and (case_index % args.progress == 0 or case_index == len(cases)):
            print(f"cases {case_index}/{len(cases)} samples {len(results)} failures {len(failures)}")

    fast = [result for result in results if result.method in FAST_METHODS]
    point_selected = [result for result in results if result.method == "point_source"]
    point_safe_reference = [
        result for result in results if result.point_source_error <= result.tolerance
    ]
    point_safe_not_selected = [
        result for result in point_safe_reference if result.method != "point_source"
    ]
    top_fast_errors = sorted(
        fast, key=lambda item: item.error / item.tolerance, reverse=True
    )[:20]
    top_all_errors = sorted(
        results, key=lambda item: item.error / item.tolerance, reverse=True
    )[:20]
    method_stats = {}
    for method in sorted(methods):
        selected = [item for item in results if item.method == method]
        method_stats[method] = {
            "samples": len(selected),
            # A disagreement is not automatically an lcbinint failure: the
            # finite-source reference is independently unreliable in the most
            # extreme high-magnification cases.  Fast-path failures retain
            # their stricter meaning below, where the audited cases are in the
            # ordinary reference regime.
            "reference_disagreements": sum(
                item.error > item.tolerance for item in selected
            ),
            "over_half_tolerance": sum(
                item.error > 0.5 * item.tolerance for item in selected
            ),
            "unconverged": sum(not item.converged for item in selected),
            "max_error_ratio": max(
                (item.error / item.tolerance for item in selected), default=0.0
            ),
        }
    fast_method_stats = {}
    for method in sorted(FAST_METHODS):
        selected = [item for item in fast if item.method == method]
        fast_method_stats[method] = {
            "samples": len(selected),
            "failures": sum(item.error > item.tolerance for item in selected),
            "max_error_ratio": max(
                (item.error / item.tolerance for item in selected), default=0.0
            ),
        }
    summary = {
        "seed": args.seed,
        "lens_cases": len(cases),
        "requested_samples": len(cases) * args.points_per_case,
        "evaluated_samples": len(results),
        "method_counts": dict(sorted(methods.items())),
        "method_stats": method_stats,
        "safety_flag_counts": {str(key): value for key, value in sorted(safety_flags.items())},
        "all_method_reference_disagreements": sum(
            item.error > item.tolerance for item in results
        ),
        "all_method_over_half_tolerance": sum(
            item.error > 0.5 * item.tolerance for item in results
        ),
        "max_all_error_ratio": max(
            (item.error / item.tolerance for item in results), default=0.0
        ),
        "fast_path_samples": len(fast),
        "fast_method_stats": fast_method_stats,
        "fast_path_failures": len(failures),
        "fast_path_over_half_tolerance": sum(
            item.error > 0.5 * item.tolerance for item in fast
        ),
        "max_fast_error_ratio": max(
            (item.error / item.tolerance for item in fast), default=0.0
        ),
        "reference_point_source_safe_samples": len(point_safe_reference),
        "point_source_selected_samples": len(point_selected),
        "point_source_selection_failures": sum(
            item.point_source_error > item.tolerance for item in point_selected
        ),
        "point_source_safe_but_not_selected": len(point_safe_not_selected),
        "point_source_safe_but_not_selected_methods": dict(sorted(Counter(
            item.method for item in point_safe_not_selected
        ).items())),
        "point_source_safe_but_not_selected_flags": {
            str(key): value for key, value in sorted(Counter(
                item.safety_flags for item in point_safe_not_selected
            ).items())
        },
        "point_source_selection_recall": (
            sum(item.method == "point_source" for item in point_safe_reference) /
            len(point_safe_reference)
            if point_safe_reference else 1.0
        ),
        "unconverged_samples": sum(not item.converged for item in results),
        "unconverged_fast_path_samples": sum(not item.converged for item in fast),
        "evaluation_errors": evaluation_errors,
        "failures": failures,
        "top_fast_errors": [asdict(item) for item in top_fast_errors],
        "top_all_reference_differences": [asdict(item) for item in top_all_errors],
        "coefficient_grid": coefficient_grid(results),
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lens-cases", type=int, default=96)
    parser.add_argument("--points-per-case", type=int, default=12)
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--progress", type=int, default=12)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = run(args)
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 1 if summary["fast_path_failures"] or summary["evaluation_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
