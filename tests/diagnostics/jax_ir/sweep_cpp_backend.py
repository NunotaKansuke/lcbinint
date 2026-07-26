#!/usr/bin/env python3
"""Held-out value and derivative sweep for the JAX CPU FFI backend."""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import lcbinint  # noqa: E402
from lcbinint_jax import (  # noqa: E402
    binary_inverse_ray_fixed_support,
    binary_inverse_ray_fixed_support_ffi,
    discover_binary_macro_tiles,
    discover_binary_macro_tiles_ffi,
)


@dataclass(frozen=True)
class Case:
    separation: float
    mass_ratio: float
    source_radius: float


CASES = (
    Case(0.55, 1.0, 3.0e-3),
    Case(0.75, 0.1, 1.0e-2),
    Case(0.90, 1.0e-3, 3.0e-3),
    Case(1.00, 1.0e-3, 1.0e-3),
    Case(1.20, 0.1, 2.0e-2),
    Case(1.50, 1.0, 1.0e-2),
    Case(1.80, 1.0e-3, 1.0e-3),
    Case(2.50, 1.0e-5, 1.0e-4),
)
PROFILES = {
    "uniform": (0.0, 0.0, "uniform", 3),
    "linear": (0.4, 0.0, "linear", 3),
    "square_root": (0.3, 0.2, "two_coefficient", 4),
}
DISTANCE_FACTORS = (0.5, 1.0, 2.0, 5.0)


def caustic_branches(case, bins):
    solver = lcbinint.LightCurve(
        options=lcbinint.Options(
            coordinates="center_of_mass",
            caustic_bins=bins,
        )
    )
    geometry = solver.caustics(
        s=case.separation,
        q=case.mass_ratio,
        n_points=bins,
    )
    return [
        np.column_stack((np.asarray(x), np.asarray(y)))
        for x, y in zip(geometry.x, geometry.y)
        if len(x) >= 3
    ]


def source_points(case, branches):
    points = []
    for point_index, factor in enumerate(DISTANCE_FACTORS):
        branch = branches[point_index % len(branches)]
        index = (37 + 97 * point_index) % len(branch)
        previous = branch[(index - 1) % len(branch)]
        following = branch[(index + 1) % len(branch)]
        tangent = following - previous
        tangent_norm = np.linalg.norm(tangent)
        normal = (
            np.asarray((1.0, 0.0))
            if tangent_norm == 0.0
            else np.asarray((-tangent[1], tangent[0])) / tangent_norm
        )
        side = -1.0 if point_index % 2 else 1.0
        point = branch[index] + side * factor * case.source_radius * normal
        points.append((f"caustic_{factor:g}rho", point))
    all_points = np.concatenate(branches)
    pad = max(0.3, 30.0 * case.source_radius)
    points.append(
        (
            "field",
            all_points.max(axis=0) + np.asarray((pad, 0.7 * pad)),
        )
    )
    return points


def compiled_evaluators(moment_mode, boundary_subdivision):
    def result(function, origins, mask, cell_size, parameters):
        return function(
            origins,
            mask,
            cell_size,
            *parameters,
            tile_size=16,
            moment_mode=moment_mode,
            boundary_subdivision=boundary_subdivision,
        )

    def observables(function, origins, mask, cell_size, parameters):
        evaluated = result(function, origins, mask, cell_size, parameters)
        return jnp.concatenate(
            (jnp.reshape(evaluated.magnification, (1,)), evaluated.moments)
        )

    def make_bundle(function):
        value = jax.jit(
            lambda origins, mask, cell_size, parameters: observables(
                function, origins, mask, cell_size, parameters
            )
        )
        directional = jax.jit(
            lambda origins, mask, cell_size, parameters, tangent: jax.jvp(
                lambda active: observables(function, origins, mask, cell_size, active),
                (parameters,),
                (tangent,),
            )[1]
        )
        gradient = jax.jit(
            jax.grad(
                lambda parameters, origins, mask, cell_size: result(
                    function, origins, mask, cell_size, parameters
                ).magnification
            )
        )
        return value, directional, gradient

    return (
        make_bundle(binary_inverse_ray_fixed_support),
        make_bundle(binary_inverse_ray_fixed_support_ffi),
    )


def within_budget(actual, expected, absolute, relative):
    budget = absolute + relative * np.maximum(np.abs(expected), 1.0)
    error = np.abs(actual - expected)
    return bool(np.all(error <= budget)), float(np.max(error / budget))


def run(args):
    cases = CASES[: args.lens_cases]
    evaluators = {
        profile: compiled_evaluators(moment_mode, subdivision)
        for profile, (_, _, moment_mode, subdivision) in PROFILES.items()
    }
    rows = []
    for case_index, case in enumerate(cases):
        branches = caustic_branches(case, args.caustic_bins)
        for point_name, point in source_points(case, branches):
            cell_size = case.source_radius / args.resolution
            discovery = discover_binary_macro_tiles(
                point[0],
                point[1],
                case.separation,
                case.mass_ratio,
                case.source_radius,
                cell_size,
                tile_size=16,
                tile_capacity=args.tile_capacity,
                limb_samples=args.limb_samples,
            )
            ffi_discovery = discover_binary_macro_tiles_ffi(
                point[0],
                point[1],
                case.separation,
                case.mass_ratio,
                case.source_radius,
                cell_size,
                tile_size=16,
                tile_capacity=args.tile_capacity,
                limb_samples=args.limb_samples,
            )
            jax.block_until_ready(discovery.tile_origins)
            jax.block_until_ready(ffi_discovery.tile_origins)
            discovery_matches = all(
                np.array_equal(
                    np.asarray(getattr(discovery, field)),
                    np.asarray(getattr(ffi_discovery, field)),
                )
                for field in discovery._fields
            )
            support_valid = not (
                bool(discovery.overflow) or bool(discovery.root_failure)
            )
            for profile, (limb_c, limb_d, _, _) in PROFILES.items():
                row = {
                    "case": case_index,
                    "separation": case.separation,
                    "mass_ratio": case.mass_ratio,
                    "source_radius": case.source_radius,
                    "point": point_name,
                    "source_x": float(point[0]),
                    "source_y": float(point[1]),
                    "profile": profile,
                    "support_valid": support_valid,
                    "tile_count": int(discovery.visited_count),
                    "discovery_matches": discovery_matches,
                }
                if not support_valid:
                    rows.append(row)
                    continue

                parameters = jnp.asarray(
                    (
                        point[0],
                        point[1],
                        case.separation,
                        case.mass_ratio,
                        case.source_radius,
                        limb_c,
                        limb_d,
                    )
                )
                tangent = jnp.asarray(
                    (
                        0.2 * case.source_radius,
                        -0.1 * case.source_radius,
                        0.03 * case.separation,
                        0.02 * max(case.mass_ratio, 1.0e-4),
                        -0.05 * case.source_radius,
                        0.1,
                        -0.05,
                    )
                )
                origins = discovery.tile_origins
                mask = discovery.tile_mask
                pure, ffi = evaluators[profile]
                pure_value = np.asarray(pure[0](origins, mask, cell_size, parameters))
                ffi_value = np.asarray(ffi[0](origins, mask, cell_size, parameters))
                pure_jvp = np.asarray(
                    pure[1](origins, mask, cell_size, parameters, tangent)
                )
                ffi_jvp = np.asarray(
                    ffi[1](origins, mask, cell_size, parameters, tangent)
                )
                pure_gradient = np.asarray(
                    pure[2](parameters, origins, mask, cell_size)
                )
                ffi_gradient = np.asarray(ffi[2](parameters, origins, mask, cell_size))
                value_passes, value_ratio = within_budget(
                    ffi_value, pure_value, 2.0e-10, 2.0e-10
                )
                jvp_passes, jvp_ratio = within_budget(ffi_jvp, pure_jvp, 2.0e-8, 2.0e-8)
                gradient_passes, gradient_ratio = within_budget(
                    # The sequential C++ and XLA reductions differ at roughly
                    # 1e-7 in extreme q=1e-5 derivative cancellation cases.
                    ffi_gradient,
                    pure_gradient,
                    2.0e-7,
                    2.0e-7,
                )
                row.update(
                    {
                        "value_passes": value_passes,
                        "jvp_passes": jvp_passes,
                        "gradient_passes": gradient_passes,
                        "value_max_budget_ratio": value_ratio,
                        "jvp_max_budget_ratio": jvp_ratio,
                        "gradient_max_budget_ratio": gradient_ratio,
                        "pure_gradient": pure_gradient.tolist(),
                        "ffi_gradient": ffi_gradient.tolist(),
                        "gradient_absolute_error": np.abs(
                            ffi_gradient - pure_gradient
                        ).tolist(),
                        "passes": (value_passes and jvp_passes and gradient_passes),
                    }
                )
                rows.append(row)
        print(f"processed case {case_index + 1}/{len(cases)}", flush=True)

    evaluated = [row for row in rows if row["support_valid"]]
    summary = {
        "rows": len(rows),
        "support_valid_rows": len(evaluated),
        "unsupported_rows": len(rows) - len(evaluated),
        "failures": sum(not row["passes"] for row in evaluated),
        "discovery_failures": sum(not row["discovery_matches"] for row in rows),
        "max_value_budget_ratio": max(
            (row["value_max_budget_ratio"] for row in evaluated),
            default=math.nan,
        ),
        "max_jvp_budget_ratio": max(
            (row["jvp_max_budget_ratio"] for row in evaluated),
            default=math.nan,
        ),
        "max_gradient_budget_ratio": max(
            (row["gradient_max_budget_ratio"] for row in evaluated),
            default=math.nan,
        ),
        "failure_rows": [row for row in evaluated if not row["passes"]][:20],
    }
    output = {
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "summary": summary,
        "rows": rows,
    }
    rendered = json.dumps(output, indent=2)
    print(json.dumps(summary, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    return int(summary["failures"] != 0 or summary["discovery_failures"] != 0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lens-cases", type=int, default=len(CASES))
    parser.add_argument("--caustic-bins", type=int, default=512)
    parser.add_argument("--resolution", type=int, default=96)
    parser.add_argument("--tile-capacity", type=int, default=4096)
    parser.add_argument("--limb-samples", type=int, default=24)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.lens_cases < 1 or args.lens_cases > len(CASES):
        parser.error(f"lens-cases must be between 1 and {len(CASES)}")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
