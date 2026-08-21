#!/usr/bin/env python3
"""Run corrected JAX-only rows, either a retry subset or a full lane."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np


def _load_benchmark(path):
    spec = importlib.util.spec_from_file_location("bench_jax_microlux", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--forward-timeout", type=float, default=60.0)
    parser.add_argument("--derivative-timeout", type=float, default=60.0)
    parser.add_argument(
        "--adaptive-polar",
        action="store_true",
        help=(
            "use the generic angular convergence ladder for selected retry "
            "rows (kept as a compatibility flag)"
        ),
    )
    parser.add_argument(
        "--polar-angular-bins",
        type=int,
        help=(
            "use this as the first grid in the generic polar angular "
            "convergence ladder; zero selects the C++ geometry-aware ladder"
        ),
    )
    parser.add_argument(
        "--all-rows",
        action="store_true",
        help="rerun every selected input row instead of source timeouts",
    )
    parser.add_argument(
        "--failed-accuracy",
        action="store_true",
        help=(
            "rerun completed source rows whose JAX max relative error "
            "exceeded the requested target"
        ),
    )
    parser.add_argument(
        "--retry-dadt-timeouts",
        action="store_true",
        help=(
            "rerun completed source rows whose JAX derivative timing was "
            "not available"
        ),
    )
    parser.add_argument(
        "--native-certified-only",
        action="store_true",
        help=(
            "with --failed-accuracy, exclude rows whose saved native warm-up "
            "already missed the requested VBM target"
        ),
    )
    parser.add_argument(
        "--profile",
        choices=("uniform", "linear"),
        help="restrict an all-rows run to one source-profile lane",
    )
    parser.add_argument(
        "--target",
        type=float,
        help="restrict an all-rows run to one target lane",
    )
    parser.add_argument("--case-id-min", type=int)
    parser.add_argument("--case-id-max", type=int)
    parser.add_argument(
        "--case-id",
        dest="case_ids",
        type=int,
        action="append",
        help="restrict an all-rows run to one or more exact case IDs",
    )
    parser.add_argument(
        "--d-over-rho",
        type=float,
        help="restrict an all-rows run to one source-size lane",
    )
    args = parser.parse_args()
    if args.polar_angular_bins is not None and args.polar_angular_bins < 0:
        raise SystemExit("--polar-angular-bins must be non-negative")

    bench = _load_benchmark(args.benchmark)
    _, rows = bench._load_rows(args.input)
    if args.all_rows and args.failed_accuracy:
        raise SystemExit("--failed-accuracy cannot be combined with --all-rows")
    if args.all_rows and args.retry_dadt_timeouts:
        raise SystemExit(
            "--retry-dadt-timeouts cannot be combined with --all-rows"
        )
    if args.failed_accuracy and args.retry_dadt_timeouts:
        raise SystemExit(
            "--failed-accuracy cannot be combined with "
            "--retry-dadt-timeouts"
        )
    previous = None if args.all_rows else json.loads(args.source_result.read_text())
    # The JAX retry result is more specific than the native input status: a
    # few rows completed in the native run but timed out in JAX.  Coordinates
    # identify those rows, while preserving duplicate case/profile keys.
    def key(row):
        return int(row["case_id"]), str(row["profile"]), float(row["target"])

    def coordinate(row):
        return f"{float(row['x']):.17g}", f"{float(row['y']):.17g}"

    def native_plan_certified(row):
        values = row.get("chosen_vbm_errors")
        if not isinstance(values, (list, tuple)):
            return False
        finite = [float(value) for value in values if np.isfinite(value)]
        return bool(finite) and all(
            value <= float(row["target"]) for value in finite
        )

    def in_scope(item):
        if args.profile is not None and str(item.get("profile")) != args.profile:
            return False
        if args.target is not None and float(item.get("target")) != float(args.target):
            return False
        case_id = int(item.get("case_id"))
        if args.case_id_min is not None and case_id < args.case_id_min:
            return False
        if args.case_id_max is not None and case_id >= args.case_id_max:
            return False
        if args.case_ids is not None and case_id not in set(args.case_ids):
            return False
        if args.d_over_rho is not None and not np.isclose(
            float(item.get("d_over_rho", np.nan)),
            float(args.d_over_rho),
            rtol=0.0,
            atol=1.0e-12,
        ):
            return False
        return True

    if args.all_rows:
        selected = bench._select_rows(
            rows,
            None if args.profile is None else (args.profile,),
            None if args.target is None else (args.target,),
            args.case_id_min,
            args.case_id_max,
            None,
        )
        if args.d_over_rho is not None:
            selected = [
                row
                for row in selected
                if np.isclose(
                    float(row.get("d_over_rho", np.nan)),
                    float(args.d_over_rho),
                    rtol=0.0,
                    atol=1.0e-12,
                )
            ]
        if args.case_ids is not None:
            case_ids = set(args.case_ids)
            selected = [row for row in selected if int(row["case_id"]) in case_ids]
        retry_scope = (
            "all selected input rows under corrected native-routed JAX; "
            "no microLUX calls"
        )
        print(
            f"rerunning all {len(selected)} selected JAX rows; "
            "microLUX is not called",
            flush=True,
        )
    else:
        failed_keys = {
            (int(item["case_id"]), str(item["profile"]), float(item["target"]))
            for item in previous["results"]
            if item.get("status") == "timeout" and in_scope(item)
        }
        failed_accuracy_coordinates = {
            (key(item), coordinate(item))
            for item in previous["results"]
            if item.get("status") == "completed"
            and in_scope(item)
            and item.get("jax_max_relative_error") is not None
            and float(item["jax_max_relative_error"]) > float(item["target"])
            and item.get("x") is not None
            and item.get("y") is not None
        }
        derivative_timeout_coordinates = {
            (key(item), coordinate(item))
            for item in previous["results"]
            if args.retry_dadt_timeouts
            and item.get("status") == "completed"
            and in_scope(item)
            and item.get("jax_dA_dt_timeout")
            and item.get("x") is not None
            and item.get("y") is not None
        }
        if args.native_certified_only and args.failed_accuracy:
            certified_coordinates = {
                (key(item), coordinate(item))
                for item in rows
                if native_plan_certified(item)
            }
            failed_accuracy_coordinates &= certified_coordinates
        completed_coordinates = {
            (key(item), coordinate(item))
            for item in previous["results"]
            if item.get("status") == "completed"
            and item.get("x") is not None
            and item.get("y") is not None
        }
        selected = []
        for row in rows:
            row_key = key(row)
            row_coordinate = coordinate(row)
            timeout_match = (
                row_key in failed_keys
                and (row_key, row_coordinate) not in completed_coordinates
            )
            accuracy_match = (
                args.failed_accuracy
                and (row_key, row_coordinate) in failed_accuracy_coordinates
            )
            derivative_timeout_match = (
                args.retry_dadt_timeouts
                and (row_key, row_coordinate) in derivative_timeout_coordinates
            )
            if timeout_match or accuracy_match or derivative_timeout_match:
                selected.append(row)
        expected = sum(
            item.get("status") == "timeout" and in_scope(item)
            for item in previous["results"]
        )
        if args.failed_accuracy:
            expected += len(failed_accuracy_coordinates)
        if args.retry_dadt_timeouts:
            expected += len(derivative_timeout_coordinates)
        if len(selected) != expected:
            raise SystemExit(
                f"expected {expected} failed rows, selected {len(selected)}"
            )
        retry_scope = (
            (
                "only input rows whose source JAX status was timeout or whose "
                "completed JAX result exceeded its target; no successful JAX "
                "rows and no microLUX calls"
            )
            if args.failed_accuracy
            else (
                "only input rows whose completed source JAX derivative timing "
                "was unavailable; no microLUX calls"
            )
            if args.retry_dadt_timeouts
            else (
                "only input rows whose source status was timeout and whose "
                "key was timeout in source result; no successful JAX rows and "
                "no microLUX calls"
            )
        )
        print(
            f"retrying exactly {len(selected)} failed JAX rows; "
            "microLUX is not called",
            flush=True,
        )

    executor = bench.JaxBatchExecutor(
        max_source_bins=400,
        polar_angular_bins=(
            bench.DEFAULT_POLAR_ANGULAR_BINS
            if args.polar_angular_bins is None else args.polar_angular_bins
        ),
        polar_angular_ratio=64.0,
    )
    refined_executor = bench.JaxBatchExecutor(
        max_source_bins=400,
        polar_angular_bins=(
            bench.DEFAULT_POLAR_ANGULAR_BINS
            if args.polar_angular_bins is None else args.polar_angular_bins
        ),
        polar_angular_ratio=64.0,
        boundary_subdivision=bench.CARTESIAN_REFINED_BOUNDARY_SUBDIVISION,
    )
    source_by_key = {}
    if args.failed_accuracy:
        source_by_key = {
            (key(item), coordinate(item)): item
            for item in previous["results"]
            if item.get("status") == "completed"
            and item.get("x") is not None
            and item.get("y") is not None
        }
    results = []
    started = time.perf_counter()
    for index, row in enumerate(selected, 1):
        profile = str(row["profile"])
        target = float(row["target"])
        profile_c = float(row.get("limb_darkening_c", 0.0))
        times = bench._times(row)
        references, _, reference_source = bench._reference_bundle(row)
        print(
            f"[{index}/{len(selected)}] case={row['case_id']} "
            f"profile={profile} target={target:g} "
            f"d/rho={float(row.get('d_over_rho', np.nan)):g}",
            flush=True,
        )
        try:
            source_item = source_by_key.get((key(row), coordinate(row)))
            # The input corpus carries the native warm-up's converged
            # per-epoch grid/nbin plan.  That is the comparison contract for
            # this benchmark, so let select() discover and use it directly.
            # Only rows without a usable native plan may reuse the previous
            # JAX plan as a starting point; reusing it for every failed row
            # silently changes the numerical policy and can preserve a bad
            # route from an earlier implementation.
            native_plan = bench.JaxBatchExecutor._saved_route_plan(
                row, len(times)
            )
            saved_route_plan = (
                None if source_item is None
                else source_item.get("jax_route_plan")
            )
            saved_resolution_plan = (
                None if source_item is None
                else source_item.get("jax_resolution_plan")
            )

            def select_for(current_executor):
                if native_plan is not None:
                    return current_executor.select(
                        profile,
                        profile_c,
                        row,
                        times,
                        references,
                        target,
                        fallback_resolution=128,
                        forward_timeout=args.forward_timeout,
                    )
                if (
                    args.failed_accuracy
                    and isinstance(saved_route_plan, (list, tuple))
                    and isinstance(saved_resolution_plan, (list, tuple))
                    and len(saved_route_plan) == len(times)
                    and len(saved_resolution_plan) == len(times)
                ):
                    return current_executor.select(
                        profile,
                        profile_c,
                        row,
                        times,
                        references,
                        target,
                        fallback_resolution=128,
                        forward_timeout=args.forward_timeout,
                        initial_plan=(
                            tuple(str(value) for value in saved_route_plan),
                            tuple(
                                max(2, int(value))
                                for value in saved_resolution_plan
                            ),
                        ),
                    )
                return current_executor.select(
                    profile,
                    profile_c,
                    row,
                    times,
                    references,
                    target,
                    fallback_resolution=128,
                    forward_timeout=args.forward_timeout,
                )

            selection = select_for(executor)
            timing_executor = executor
            if (
                not selection["target_pass"]
                and native_plan_certified(row)
                and any(
                    route == "cartesian"
                    for route in selection["route_plan"]
                )
            ):
                refined_selection = select_for(refined_executor)
                if (
                    refined_selection["target_pass"]
                    or refined_selection["selected_max_relative_error"]
                    < selection["selected_max_relative_error"]
                ):
                    selection = refined_selection
                    timing_executor = refined_executor
            derivative_compile_seconds = timing_executor.prepare_derivative(
                profile,
                profile_c,
                target,
                selection["route_plan"],
                selection["resolution_plan"],
                row,
                times,
                args.derivative_timeout,
            )
            selection["warmup_seconds"] += derivative_compile_seconds
            timing = timing_executor.timed(
                profile,
                profile_c,
                target,
                selection["route_plan"],
                selection["resolution_plan"],
                row,
                times,
                args.repeats,
                args.forward_timeout,
                args.derivative_timeout,
                False,
            )
            values = timing["forward_values"]
            derivatives = timing["dA_dt_values"]
            errors = [
                bench._relative_error(value, reference)
                for value, reference in zip(values, references)
            ]
            dadt_block = timing["dA_dt_block_seconds"]
            result = {
                "status": "completed",
                "case_id": int(row["case_id"]),
                "input_status": row.get("status"),
                "profile": profile,
                "target": target,
                "limb_darkening_c": profile_c,
                "s": float(row["s"]),
                "q": float(row["q"]),
                "rho": float(row["rho"]),
                "x": float(row["x"]),
                "y": float(row["y"]),
                "d_over_rho": float(row.get("d_over_rho", np.nan)),
                "times": times.tolist(),
                "batch_epochs": len(times),
                "reference": references,
                "reference_source": reference_source,
                "jax_warmup_mode": selection["warmup_mode"],
                "jax_convergence_certified": selection[
                    "convergence_certified"
                ],
                "jax_target_pass": selection["target_pass"],
                "jax_support_valid": selection["support_valid"],
                "jax_certified_pass": selection["certified_pass"],
                "jax_discovery_overflow": selection[
                    "discovery_overflow"
                ],
                "jax_root_failure": selection["root_failure"],
                "jax_boundary_subdivision": selection[
                    "boundary_subdivision"
                ],
                "jax_route_mode": "convergence_selected_static_ffi",
                "jax_selected_resolution": selection["resolution"],
                "jax_max_source_bins": selection["max_source_bins"],
                "jax_polar_angular_bins": timing_executor.polar_angular_bins,
                "jax_polar_angular_ratio": timing_executor.polar_angular_ratio,
                "jax_route_plan": selection["route_plan"],
                "jax_resolution_plan": selection["resolution_plan"],
                "jax_route_plan_source": selection["route_plan_source"],
                "jax_native_plan_start": selection["native_plan_start"],
                "jax_native_plan_start_mode": selection[
                    "native_plan_start_mode"
                ],
                "jax_warmup_seconds": selection["warmup_seconds"],
                "jax_derivative_compile_seconds": derivative_compile_seconds,
                "jax_forward_values": values.tolist(),
                "jax_dA_dt_values": (
                    None if derivatives is None else derivatives.tolist()
                ),
                "jax_relative_errors": errors,
                "jax_max_relative_error": float(
                    np.nanmax(np.asarray(errors, dtype=float))
                ),
                "jax_forward_block_seconds": timing[
                    "forward_block_seconds"
                ],
                "jax_dA_dt_block_seconds": dadt_block,
                "jax_forward_seconds_per_epoch": timing[
                    "forward_block_seconds"
                ] / len(times),
                "jax_dA_dt_seconds_per_epoch": (
                    None if dadt_block is None else dadt_block / len(times)
                ),
                "jax_forward_samples_seconds": timing[
                    "forward_samples_seconds"
                ],
                "jax_dA_dt_samples_seconds": timing[
                    "dA_dt_samples_seconds"
                ],
                "jax_dA_dt_timeout": timing["dA_dt_timeout"],
                "jax_dA_dt_timeout_message": timing[
                    "dA_dt_timeout_message"
                ],
                "derivative_skipped_hard_case": False,
            }
            print(
                f"  completed resolution={selection['resolution']} "
                f"forward={timing['forward_block_seconds']:.6g}s "
                f"dA_dt={'skipped' if dadt_block is None else f'{dadt_block:.6g}s'} "
                f"max_relerr={result['jax_max_relative_error']:.3g}",
                flush=True,
            )
        except Exception as error:  # noqa: BLE001
            result = {
                "status": "error",
                "case_id": int(row["case_id"]),
                "profile": profile,
                "target": target,
                "d_over_rho": row.get("d_over_rho"),
                "error": f"{type(error).__name__}: {error}",
            }
            print(f"  error {result['error']}", flush=True)
        results.append(result)

    payload = {
        "source_result": str(args.source_result),
        "input": str(args.input),
        "retry_scope": retry_scope,
        "configuration": {
            "rows": len(selected),
            "batch_epochs": len(bench.REFERENCE_INDICES),
            "repeats": args.repeats,
            "forward_timeout_seconds": args.forward_timeout,
            "derivative_timeout_seconds": args.derivative_timeout,
            "omp_num_threads": "from environment",
            "jax_backend": "native_plan_direct_ffi",
            "polar_angular_bins": executor.polar_angular_bins,
            "polar_angular_policy": (
                f"base {args.polar_angular_bins} with generic power-of-two "
                "angular convergence for selected accuracy rows"
                if args.polar_angular_bins is not None
                else (
                    f"base {bench.DEFAULT_POLAR_ANGULAR_BINS} with generic "
                    "power-of-two angular convergence"
                )
            ),
            "jax_angular_policy": (
                "global angular bins 65536,131072,262144,524288,1048576,2097152 "
                "with adjacent-grid convergence; stable misses are recorded "
                "as unresolved and stop the ladder; "
                "no q/d-over-rho branches"
            ),
            "jax_cartesian_boundary_policy": (
                "compiled 4x4 boundary quadrature by default; native-certified "
                "rows whose 4x4 result misses target are rechecked with the "
                "global 8x8 quadrature rule"
            ),
            "jax_cartesian_boundary_subdivision": (
                bench.CARTESIAN_BOUNDARY_SUBDIVISION
            ),
            "jax_cartesian_refined_boundary_subdivision": (
                bench.CARTESIAN_REFINED_BOUNDARY_SUBDIVISION
            ),
            "cartesian_boundary_subdivision": (
                bench.CARTESIAN_BOUNDARY_SUBDIVISION
            ),
            "jax_geometry_skip_policy": "none",
            "lcbinint_build_root": str(bench.RUNTIME_BUILD_ROOT),
            "lcbinint_extension": str(bench.RUNTIME_EXTENSION),
            "jax_ffi_capabilities": bench.JAX_FFI_CAPABILITIES,
            "microLUX_called": False,
            "native_certified_only": args.native_certified_only,
        },
        "results": results,
        "compile_records": [
            *executor.compile_records.values(),
            *refined_executor.compile_records.values(),
        ],
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
