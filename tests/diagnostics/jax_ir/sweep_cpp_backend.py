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
    binary_hexadecapole,
    binary_inverse_ray_cartesian_ffi,
    binary_inverse_ray_fixed_support,
    binary_inverse_ray_fixed_support_ffi,
    binary_point_source_magnification,
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


def compiled_fused_evaluator(
    moment_mode,
    boundary_subdivision,
    resolution,
    tile_capacity,
    limb_samples,
):
    def observables(parameters):
        result = binary_inverse_ray_cartesian_ffi(
            *parameters,
            cell_size=jax.lax.stop_gradient(parameters[4] / resolution),
            tile_size=16,
            tile_capacity=tile_capacity,
            limb_samples=limb_samples,
            moment_mode=moment_mode,
            boundary_subdivision=boundary_subdivision,
        )
        return jnp.concatenate(
            (jnp.reshape(result.magnification, (1,)), result.moments)
        )

    return (
        jax.jit(observables),
        jax.jit(
            lambda parameters, tangent: jax.jvp(
                observables,
                (parameters,),
                (tangent,),
            )[1]
        ),
        jax.jit(jax.grad(lambda parameters: observables(parameters)[0])),
    )


def compiled_point_evaluator(root_backend):
    def evaluate(parameters):
        result = binary_point_source_magnification(
            *parameters,
            root_backend=root_backend,
        )
        return result.magnification, (result.image_count, result.root_failure)

    return jax.jit(jax.value_and_grad(evaluate, has_aux=True))


def compiled_hex_evaluator(root_backend):
    def evaluate(parameters):
        result = binary_hexadecapole(
            *parameters,
            root_backend=root_backend,
        )
        diagnostics = (
            result.point_magnification,
            result.quadrupole_correction,
            result.hexadecapole_correction,
            result.estimated_error,
            result.topology_stable,
            result.root_failure,
        )
        return result.magnification, diagnostics

    return jax.jit(jax.value_and_grad(evaluate, has_aux=True))


def within_budget(actual, expected, absolute, relative):
    budget = absolute + relative * np.maximum(np.abs(expected), 1.0)
    error = np.abs(actual - expected)
    return bool(np.all(error <= budget)), float(np.max(error / budget))


def discovery_signature(discovery):
    indices = np.asarray(discovery.tile_indices)
    mask = np.asarray(discovery.tile_mask)
    active = np.asarray(discovery.active_mask)
    visited_tiles = {(int(index[0]), int(index[1])) for index in indices[mask]}
    active_tiles = {(int(index[0]), int(index[1])) for index in indices[mask & active]}
    return (
        visited_tiles,
        active_tiles,
        bool(discovery.overflow),
        int(discovery.visited_count),
        int(discovery.active_count),
        int(discovery.seed_count),
        bool(discovery.root_failure),
    )


def discoveries_equivalent(pure, ffi):
    failure_flags_match = bool(pure.overflow) == bool(ffi.overflow) and bool(
        pure.root_failure
    ) == bool(ffi.root_failure)
    if not failure_flags_match:
        return False
    if bool(pure.overflow) or bool(pure.root_failure):
        return True
    return discovery_signature(pure) == discovery_signature(ffi)


def run(args):
    cases = CASES[: args.lens_cases]
    evaluators = {
        profile: compiled_evaluators(moment_mode, subdivision)
        for profile, (_, _, moment_mode, subdivision) in PROFILES.items()
    }
    fused_evaluators = {
        profile: compiled_fused_evaluator(
            moment_mode,
            subdivision,
            args.resolution,
            args.tile_capacity,
            args.limb_samples,
        )
        for profile, (_, _, moment_mode, subdivision) in PROFILES.items()
    }
    point_evaluators = (
        compiled_point_evaluator("jax"),
        compiled_point_evaluator("ffi"),
    )
    hex_evaluators = (
        compiled_hex_evaluator("jax"),
        compiled_hex_evaluator("ffi"),
    )
    rows = []
    for case_index, case in enumerate(cases):
        branches = caustic_branches(case, args.caustic_bins)
        for point_name, point in source_points(case, branches):
            point_parameters = jnp.asarray(
                (point[0], point[1], case.separation, case.mass_ratio)
            )
            pure_point, pure_point_gradient = point_evaluators[0](point_parameters)
            ffi_point, ffi_point_gradient = point_evaluators[1](point_parameters)
            point_value_passes, point_value_ratio = within_budget(
                np.asarray(ffi_point[0]),
                np.asarray(pure_point[0]),
                2.0e-10,
                2.0e-10,
            )
            point_gradient_passes, point_gradient_ratio = within_budget(
                np.asarray(ffi_point_gradient),
                np.asarray(pure_point_gradient),
                2.0e-7,
                2.0e-7,
            )
            point_diagnostics_match = all(
                np.array_equal(np.asarray(ffi), np.asarray(pure))
                for ffi, pure in zip(ffi_point[1], pure_point[1])
            )
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
                root_backend="jax",
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
                root_backend="ffi",
            )
            jax.block_until_ready(discovery.tile_origins)
            jax.block_until_ready(ffi_discovery.tile_origins)
            discovery_matches = discoveries_equivalent(
                discovery,
                ffi_discovery,
            )
            pure_support_valid = not (
                bool(discovery.overflow) or bool(discovery.root_failure)
            )
            ffi_support_valid = not (
                bool(ffi_discovery.overflow) or bool(ffi_discovery.root_failure)
            )
            support_valid = pure_support_valid and ffi_support_valid
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
                    "point_value_passes": point_value_passes,
                    "point_gradient_passes": point_gradient_passes,
                    "point_diagnostics_match": point_diagnostics_match,
                    "point_value_max_budget_ratio": point_value_ratio,
                    "point_gradient_max_budget_ratio": point_gradient_ratio,
                }
                hex_parameters = jnp.asarray(
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
                pure_hex, pure_hex_gradient = hex_evaluators[0](hex_parameters)
                ffi_hex, ffi_hex_gradient = hex_evaluators[1](hex_parameters)
                hex_value_passes, hex_value_ratio = within_budget(
                    np.asarray(ffi_hex[0]),
                    np.asarray(pure_hex[0]),
                    2.0e-9,
                    2.0e-9,
                )
                hex_gradient_passes, hex_gradient_ratio = within_budget(
                    np.asarray(ffi_hex_gradient),
                    np.asarray(pure_hex_gradient),
                    2.0e-6,
                    2.0e-6,
                )
                hex_diagnostics_match = all(
                    np.allclose(
                        np.asarray(ffi),
                        np.asarray(pure),
                        rtol=2.0e-9,
                        atol=2.0e-9,
                        equal_nan=True,
                    )
                    for ffi, pure in zip(ffi_hex[1], pure_hex[1])
                )
                row.update(
                    {
                        "hex_value_passes": hex_value_passes,
                        "hex_gradient_passes": hex_gradient_passes,
                        "hex_diagnostics_match": hex_diagnostics_match,
                        "hex_value_max_budget_ratio": hex_value_ratio,
                        "hex_gradient_max_budget_ratio": hex_gradient_ratio,
                    }
                )
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
                pure, ffi = evaluators[profile]
                pure_value = np.asarray(
                    pure[0](
                        discovery.tile_origins,
                        discovery.tile_mask,
                        cell_size,
                        parameters,
                    )
                )
                ffi_value = np.asarray(
                    ffi[0](
                        ffi_discovery.tile_origins,
                        ffi_discovery.tile_mask,
                        cell_size,
                        parameters,
                    )
                )
                pure_jvp = np.asarray(
                    pure[1](
                        discovery.tile_origins,
                        discovery.tile_mask,
                        cell_size,
                        parameters,
                        tangent,
                    )
                )
                ffi_jvp = np.asarray(
                    ffi[1](
                        ffi_discovery.tile_origins,
                        ffi_discovery.tile_mask,
                        cell_size,
                        parameters,
                        tangent,
                    )
                )
                pure_gradient = np.asarray(
                    pure[2](
                        parameters,
                        discovery.tile_origins,
                        discovery.tile_mask,
                        cell_size,
                    )
                )
                ffi_gradient = np.asarray(
                    ffi[2](
                        parameters,
                        ffi_discovery.tile_origins,
                        ffi_discovery.tile_mask,
                        cell_size,
                    )
                )
                fused = fused_evaluators[profile]
                fused_value = np.asarray(fused[0](parameters))
                fused_jvp = np.asarray(fused[1](parameters, tangent))
                fused_gradient = np.asarray(fused[2](parameters))
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
                fused_value_passes, fused_value_ratio = within_budget(
                    fused_value, ffi_value, 2.0e-10, 2.0e-10
                )
                fused_jvp_passes, fused_jvp_ratio = within_budget(
                    # Internal root seeding can permute the same support queue;
                    # square-root moments then accumulate at slightly different
                    # floating-point order.
                    fused_jvp,
                    ffi_jvp,
                    2.0e-8,
                    2.0e-8,
                )
                fused_gradient_passes, fused_gradient_ratio = within_budget(
                    fused_gradient, ffi_gradient, 2.0e-7, 2.0e-7
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
                        "fused_value_passes": fused_value_passes,
                        "fused_jvp_passes": fused_jvp_passes,
                        "fused_gradient_passes": fused_gradient_passes,
                        "fused_value_max_budget_ratio": fused_value_ratio,
                        "fused_jvp_max_budget_ratio": fused_jvp_ratio,
                        "fused_gradient_max_budget_ratio": fused_gradient_ratio,
                        "passes": (value_passes and jvp_passes and gradient_passes),
                        "fused_passes": (
                            fused_value_passes
                            and fused_jvp_passes
                            and fused_gradient_passes
                        ),
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
        "fused_failures": sum(not row["fused_passes"] for row in evaluated),
        "discovery_failures": sum(not row["discovery_matches"] for row in rows),
        "point_failures": sum(
            not (
                row["point_value_passes"]
                and row["point_gradient_passes"]
                and row["point_diagnostics_match"]
            )
            for row in rows
        ),
        "hex_failures": sum(
            not (
                row["hex_value_passes"]
                and row["hex_gradient_passes"]
                and row["hex_diagnostics_match"]
            )
            for row in rows
        ),
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
        "max_fused_value_budget_ratio": max(
            (row["fused_value_max_budget_ratio"] for row in evaluated),
            default=math.nan,
        ),
        "max_fused_jvp_budget_ratio": max(
            (row["fused_jvp_max_budget_ratio"] for row in evaluated),
            default=math.nan,
        ),
        "max_fused_gradient_budget_ratio": max(
            (row["fused_gradient_max_budget_ratio"] for row in evaluated),
            default=math.nan,
        ),
        "failure_rows": [row for row in evaluated if not row["passes"]][:20],
        "fused_failure_rows": [row for row in evaluated if not row["fused_passes"]][
            :20
        ],
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
    return int(
        summary["failures"] != 0
        or summary["fused_failures"] != 0
        or summary["discovery_failures"] != 0
        or summary["point_failures"] != 0
        or summary["hex_failures"] != 0
    )


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
