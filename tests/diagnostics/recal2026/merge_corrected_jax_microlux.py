#!/usr/bin/env python3
"""Merge corrected JAX lanes with the original microLUX measurements.

The original combined run wrote microLUX and JAX into the same row.  Its
JAX rows are not reusable because all routes were forced to Cartesian.  This
script replaces every JAX row with the corrected native-routed result while
leaving the original microLUX fields untouched.
"""

from __future__ import annotations

import copy
import argparse
import json
import math
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RESULT_ROOT = ROOT / "tests" / "diagnostics" / "results" / "recal2026"
# This is the previous combined artifact: it already contains the original
# microLUX measurements and the corrected schema/metadata from the earlier
# partial JAX retry.  Use it as the merge base so microLUX is never rerun or
# accidentally dropped while replacing all JAX rows.
SOURCE_RESULT = RESULT_ROOT / "jax_microlux_12800_corrected" / "results.json"
INPUT_REPORT = (
    RESULT_ROOT
    / "pure_kernel_balanced_loguniform_20260812"
    / "report_v2"
    / "results.json"
)
OUTPUT_RESULT = (
    RESULT_ROOT / "jax_microlux_12800_final_adaptive_20260818" / "results.json"
)

LANES = (
    (
        "uniform",
        0.001,
        RESULT_ROOT / "jax_microlux_12800_full_uniform_1e3_adaptive_20260818" / "results.json",
    ),
    (
        "linear",
        0.001,
        RESULT_ROOT / "jax_microlux_12800_full_linear_1e3_adaptive_20260818" / "results.json",
    ),
    (
        "uniform",
        0.0001,
        RESULT_ROOT / "jax_microlux_12800_full_uniform_1e4_adaptive_20260818" / "results.json",
    ),
    (
        "linear",
        0.0001,
        RESULT_ROOT / "jax_microlux_12800_full_linear_1e4_adaptive_20260818" / "results.json",
    ),
)

TARGETED_CORRECTIONS = (
    RESULT_ROOT
    / "jax_microlux_12800_uniform_1e4_dadt_retry120_adaptive_20260818"
    / "results.json",
)


def _float_token(value):
    if value is None:
        return None
    return f"{float(value):.17g}"


def row_key(row):
    """Use coordinates because timeout placeholders lack their coordinates."""

    if row.get("x") is None or row.get("y") is None:
        return None
    return (
        int(row["case_id"]),
        str(row["profile"]),
        _float_token(row["target"]),
        _float_token(row["x"]),
        _float_token(row["y"]),
    )


def lane_key(row):
    return str(row["profile"]), float(row["target"])


def finite_values(values):
    if not isinstance(values, (list, tuple)):
        return []
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def scalar_values(rows, field):
    values = []
    for row in rows:
        value = row.get(field)
        if value is not None:
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
    return values


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def describe(values):
    values = list(values)
    if not values:
        return {"count": 0, "median": None, "p90": None, "maximum": None}
    return {
        "count": len(values),
        "median": percentile(values, 0.5),
        "p90": percentile(values, 0.9),
        "maximum": max(values),
    }


def compare_fields(rows, left_field, right_field, left_name, right_name):
    counts = Counter()
    ratios = []
    for row in rows:
        left = row.get(left_field)
        right = row.get(right_field)
        try:
            left = float(left)
            right = float(right)
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(value) and value > 0.0 for value in (left, right)):
            continue
        ratios.append(left / right)
        if left < right:
            counts[f"{left_name}_wins"] += 1
        elif right < left:
            counts[f"{right_name}_wins"] += 1
        else:
            counts["ties"] += 1
    matched = sum(counts.values())
    return {
        "matched_rows": matched,
        "left_field": left_field,
        "right_field": right_field,
        "counts": dict(counts),
        "win_rates_percent": {
            name: (100.0 * counts.get(f"{name}_wins", 0) / matched if matched else None)
            for name in (left_name, right_name)
        },
        "median_left_over_right": percentile(ratios, 0.5),
        "p10_left_over_right": percentile(ratios, 0.1),
        "p90_left_over_right": percentile(ratios, 0.9),
    }


def copy_jax_result(base, jax_row, native_row):
    result = copy.deepcopy(base) if base is not None else {}
    if base is not None:
        result["legacy_status"] = base.get("status")
        if base.get("timeout_stage") is not None:
            result["legacy_timeout_stage"] = base["timeout_stage"]

    # Remove every JAX field from the invalid combined run before inserting
    # the corrected schema.  In particular, old ``jax_values`` and
    # ``jax_dA_dt`` aliases must not survive beside the new measurements.
    for field in list(result):
        if field.startswith("jax_"):
            del result[field]

    # Keep the row's physical inputs and target-specific reference in sync
    # with the corrected JAX run.  Do not copy any microLUX fields.
    for field in (
        "case_id",
        "profile",
        "target",
        "limb_darkening_c",
        "s",
        "q",
        "rho",
        "x",
        "y",
        "d_over_rho",
        "times",
        "batch_epochs",
        "input_status",
        "reference",
        "reference_source",
    ):
        if field in jax_row:
            result[field] = copy.deepcopy(jax_row[field])

    for field, value in jax_row.items():
        if field.startswith("jax_") or field == "derivative_skipped_hard_case":
            result[field] = copy.deepcopy(value)
    result["jax_values"] = copy.deepcopy(jax_row.get("jax_forward_values"))
    result["jax_dA_dt"] = copy.deepcopy(jax_row.get("jax_dA_dt_values"))

    result["status"] = jax_row.get("status")
    result["jax_status"] = jax_row.get("status")
    result["jax_input_status"] = jax_row.get("input_status")
    result["jax_corrected_all_rows"] = True
    # The corrected retry executes both forward and derivative paths for
    # every selected row.  Do not retain the old combined-run skip marker.
    result["jax_forward_skipped_hard_case"] = False

    errors = finite_values(native_row.get("chosen_vbm_errors") if native_row else None)
    target = float(jax_row["target"])
    if errors:
        native_max = max(errors)
        native_passes = native_max <= target
        native_status = "passes_target" if native_passes else "fails_target"
    else:
        native_max = None
        native_passes = None
        native_status = "missing"
    result["native_plan_max_relative_error"] = native_max
    result["native_plan_passes_target"] = native_passes
    result["native_plan_status"] = native_status
    if native_row:
        result["native_warmup_chosen_grid"] = copy.deepcopy(native_row.get("chosen_grid"))
        result["native_warmup_chosen_nbin"] = copy.deepcopy(native_row.get("chosen_nbin"))
        result["native_warmup_chosen_seconds"] = copy.deepcopy(native_row.get("chosen_seconds"))
        result["native_vbm_over_lcbinint_ratios"] = copy.deepcopy(
            native_row.get("ratios_vbm_over_lcbinint")
        )

    micro_available = finite_values(result.get("microlux_default_values"))
    result["microlux_status"] = "available" if micro_available else "missing_from_original_source"
    if result["jax_status"] == "completed" and micro_available:
        result["combined_status"] = "completed"
    elif result["jax_status"] == "completed":
        result["combined_status"] = "jax_completed_microlux_missing"
    else:
        result["combined_status"] = "jax_timeout"
    return result


def lane_summary(rows):
    profile = str(rows[0]["profile"])
    target = float(rows[0]["target"])
    jax_errors = scalar_values(rows, "jax_max_relative_error")
    native_errors = scalar_values(rows, "native_plan_max_relative_error")
    jax_completed = [row for row in rows if row.get("jax_status") == "completed"]
    micro_rows = [row for row in rows if row.get("microlux_status") == "available"]
    both_forward = []
    both_derivative = []
    for row in micro_rows:
        jax_forward = row.get("jax_forward_block_seconds")
        micro_forward = row.get("microlux_default_forward_block_seconds")
        jax_derivative = row.get("jax_dA_dt_block_seconds")
        micro_derivative = row.get("microlux_default_dA_dt_block_seconds")
        if all(value is not None for value in (jax_forward, micro_forward)) and float(jax_forward) > 0:
            both_forward.append(float(micro_forward) / float(jax_forward))
        if all(value is not None for value in (jax_derivative, micro_derivative)) and float(jax_derivative) > 0:
            both_derivative.append(float(micro_derivative) / float(jax_derivative))

    jax_pass = sum(
        row.get("jax_max_relative_error") is not None
        and float(row["jax_max_relative_error"]) <= target
        for row in rows
    )
    native_pass = sum(row.get("native_plan_status") == "passes_target" for row in rows)
    native_fail = sum(row.get("native_plan_status") == "fails_target" for row in rows)
    native_missing = sum(row.get("native_plan_status") == "missing" for row in rows)
    for row in rows:
        native_seconds = finite_values(row.get("native_warmup_chosen_seconds"))
        if native_seconds and len(native_seconds) == int(row.get("batch_epochs", 0)):
            row["native_warmup_chosen_block_seconds"] = sum(native_seconds)
        else:
            row["native_warmup_chosen_block_seconds"] = None
    native_vbm_ratios = [
        ratio
        for row in rows
        for ratio in finite_values(row.get("native_vbm_over_lcbinint_ratios"))
        if ratio > 0.0
    ]
    native_vbm_counts = Counter(
        "VBM_wins" if ratio < 1.0 else "native_wins" if ratio > 1.0 else "ties"
        for ratio in native_vbm_ratios
    )
    jax_fail_native_pass = sum(
        row.get("jax_max_relative_error") is not None
        and float(row["jax_max_relative_error"]) > target
        and row.get("native_plan_status") == "passes_target"
        for row in rows
    )
    jax_fail_native_missing = sum(
        row.get("jax_max_relative_error") is not None
        and float(row["jax_max_relative_error"]) > target
        and row.get("native_plan_status") == "missing"
        for row in rows
    )
    polar_epoch_count = sum(
        route == "polar"
        for row in rows
        for route in (row.get("jax_route_plan") or [])
    )
    cartesian_epoch_count = sum(
        route == "cartesian"
        for row in rows
        for route in (row.get("jax_route_plan") or [])
    )

    return {
        "profile": profile,
        "target": target,
        "rows": len(rows),
        "jax_status_counts": dict(Counter(row.get("jax_status") for row in rows)),
        "jax_dA_dt_timeout_count": sum(bool(row.get("jax_dA_dt_timeout")) for row in rows),
        "jax_forward_skipped_hard_case_count": sum(
            bool(row.get("jax_forward_skipped_hard_case")) for row in rows
        ),
        "derivative_skipped_hard_case_count": sum(
            bool(row.get("derivative_skipped_hard_case")) for row in rows
        ),
        "jax_target_pass_count": jax_pass,
        "jax_target_fail_count": len(rows) - jax_pass,
        "jax_max_relative_error": describe(jax_errors),
        "native_plan_status_counts": {
            "passes_target": native_pass,
            "fails_target": native_fail,
            "missing": native_missing,
        },
        "native_plan_max_relative_error": describe(native_errors),
        "jax_target_fail_with_native_pass_count": jax_fail_native_pass,
        "jax_target_fail_with_native_missing_count": jax_fail_native_missing,
        "route_summary": {
            "rows_with_any_polar_epoch": sum(
                any(route == "polar" for route in (row.get("jax_route_plan") or []))
                for row in rows
            ),
            "all_cartesian_rows": sum(
                bool(row.get("jax_route_plan"))
                and all(route == "cartesian" for route in row["jax_route_plan"])
                for row in rows
            ),
            "polar_epoch_count": polar_epoch_count,
            "cartesian_epoch_count": cartesian_epoch_count,
            "route_plan_source_counts": dict(
                Counter(row.get("jax_route_plan_source") for row in rows)
            ),
        },
        "microLUX_available_count": len(micro_rows),
        "microLUX_dA_dt_timeout_count": sum(
            bool(row.get("microlux_default_dA_dt_timeout")) for row in rows
        ),
        "compiled_steady_state_block_seconds": {
            "jax_forward": describe(scalar_values(rows, "jax_forward_block_seconds")),
            "jax_dA_dt": describe(scalar_values(rows, "jax_dA_dt_block_seconds")),
            "microLUX_forward": describe(
                scalar_values(micro_rows, "microlux_default_forward_block_seconds")
            ),
            "microLUX_dA_dt": describe(
                scalar_values(micro_rows, "microlux_default_dA_dt_block_seconds")
            ),
        },
        "microLUX_over_jax_block_median": {
            "forward": percentile(both_forward, 0.5),
            "dA_dt": percentile(both_derivative, 0.5),
        },
        "win_rates_and_median_ratios": {
            "microLUX_vs_jax_forward": compare_fields(
                rows,
                "microlux_default_forward_block_seconds",
                "jax_forward_block_seconds",
                "microLUX",
                "JAX",
            ),
            "microLUX_vs_jax_dA_dt": compare_fields(
                rows,
                "microlux_default_dA_dt_block_seconds",
                "jax_dA_dt_block_seconds",
                "microLUX",
                "JAX",
            ),
            "native_vs_jax_forward": compare_fields(
                rows,
                "native_warmup_chosen_block_seconds",
                "jax_forward_block_seconds",
                "native",
                "JAX",
            ),
            "VBM_vs_native_forward": {
                "matched_epochs": len(native_vbm_ratios),
                "counts": dict(native_vbm_counts),
                "win_rates_percent": {
                    "VBM": (
                        100.0 * native_vbm_counts.get("VBM_wins", 0) / len(native_vbm_ratios)
                        if native_vbm_ratios
                        else None
                    ),
                    "native": (
                        100.0 * native_vbm_counts.get("native_wins", 0) / len(native_vbm_ratios)
                        if native_vbm_ratios
                        else None
                    ),
                },
                "median_VBM_over_native": percentile(native_vbm_ratios, 0.5),
                "p10_VBM_over_native": percentile(native_vbm_ratios, 0.1),
                "p90_VBM_over_native": percentile(native_vbm_ratios, 0.9),
            },
        },
        "jax_completed_rows_with_native_failure": sum(
            row.get("jax_status") == "completed"
            and row.get("native_plan_status") == "fails_target"
            for row in rows
        ),
    }


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Merge full corrected JAX lanes with original microLUX rows."
    )
    parser.add_argument("--source-result", type=Path, default=SOURCE_RESULT)
    parser.add_argument("--input-report", type=Path, default=INPUT_REPORT)
    parser.add_argument("--output-result", type=Path, default=OUTPUT_RESULT)
    parser.add_argument(
        "--lane",
        action="append",
        nargs=3,
        metavar=("PROFILE", "TARGET", "RESULT_JSON"),
        help="override/add a lane as profile target result.json; repeat four times",
    )
    parser.add_argument(
        "--targeted-correction",
        action="append",
        type=Path,
        help="optional corrected JAX result JSON that overrides a lane row",
    )
    parser.add_argument(
        "--no-targeted-corrections",
        action="store_true",
        help="do not apply legacy single-row correction artifacts over full-lane runs",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    source_result = args.source_result
    input_report = args.input_report
    output_result = args.output_result
    lane_specs = (
        tuple((str(profile), float(target), Path(path)) for profile, target, path in args.lane)
        if args.lane
        else LANES
    )
    if args.no_targeted_corrections:
        targeted_paths = ()
    else:
        targeted_paths = (
            tuple(args.targeted_correction)
            if args.targeted_correction is not None
            else TARGETED_CORRECTIONS
        )

    source = json.loads(source_result.read_text())
    input_rows = json.loads(input_report.read_text())["results"]
    native_by_key = {row_key(row): row for row in input_rows if row_key(row) is not None}

    jax_by_key = {}
    lane_metadata = []
    for profile, target, path in lane_specs:
        payload = json.loads(path.read_text())
        lane_rows = payload["results"]
        if len(lane_rows) != 800:
            raise SystemExit(f"expected 800 rows in {path}, got {len(lane_rows)}")
        lane_metadata.append(
            {
                "profile": profile,
                "target": target,
                "path": str(path),
                "elapsed_seconds": payload.get("elapsed_seconds"),
            }
        )
        for row in lane_rows:
            key = row_key(row)
            if key is None or key in jax_by_key:
                raise SystemExit(f"duplicate or coordinate-less corrected JAX row: {row}")
            jax_by_key[key] = row

    targeted_metadata = []
    for path in targeted_paths:
        payload = json.loads(path.read_text())
        targeted_metadata.append(
            {"path": str(path), "rows": len(payload.get("results", [])), "elapsed_seconds": payload.get("elapsed_seconds")}
        )
        for row in payload.get("results", []):
            key = row_key(row)
            if key is None:
                raise SystemExit(f"coordinate-less targeted JAX row: {row}")
            jax_by_key[key] = row

    old_by_key = {row_key(row): row for row in source["results"] if row_key(row) is not None}
    results = []
    for key, jax_row in jax_by_key.items():
        result = copy_jax_result(old_by_key.get(key), jax_row, native_by_key.get(key))
        results.append(result)

    expected_rows = len(lane_specs) * 800
    if len(jax_by_key) != expected_rows or len(results) != expected_rows:
        raise SystemExit(
            f"expected {expected_rows} merged JAX rows, got {len(results)}"
        )
    if len({row_key(row) for row in results}) != expected_rows:
        raise SystemExit("merged JAX rows are not unique")

    # Match the original lane/case ordering for easy row-by-row inspection.
    order = {
        (profile, target): lane_index
        for lane_index, (profile, target, _) in enumerate(lane_specs)
    }
    results.sort(
        key=lambda row: (
            order[(str(row["profile"]), float(row["target"]))],
            float(row["d_over_rho"]),
            int(row["case_id"]),
        )
    )

    grouped = {}
    for row in results:
        grouped.setdefault(lane_key(row), []).append(row)
    summary = {
        f"{profile}:target={target}": lane_summary(grouped[(profile, target)])
        for profile, target, _ in lane_specs
    }

    output = copy.deepcopy(source)
    output["results"] = results
    output["summary"] = summary
    output["legacy_summary"] = source.get("summary")
    output["legacy_elapsed_seconds"] = source.get("elapsed_seconds")
    output["elapsed_seconds"] = sum(
        item["elapsed_seconds"] or 0.0 for item in lane_metadata
    ) + sum(item["elapsed_seconds"] or 0.0 for item in targeted_metadata)
    output["compile_records"] = []
    for item in [*lane_metadata, *targeted_metadata]:
        payload = json.loads(Path(item["path"]).read_text())
        for record in payload.get("compile_records", []):
            record = copy.deepcopy(record)
            record["corrected_lane_source"] = item["path"]
            output["compile_records"].append(record)
    output["correction"] = {
        "jax_rows_rerun": len(results),
        "jax_rows_from_full_corrected_lanes": 3200,
        "jax_rows_targeted_replaced_after_policy_fix": sum(
            item["rows"] for item in targeted_metadata
        ),
        "microLUX_rerun": False,
        "microLUX_rows_reused_from_original": sum(
            row.get("microlux_status") == "available" for row in results
        ),
        "microLUX_rows_missing_from_original": sum(
            row.get("microlux_status") != "available" for row in results
        ),
        "microLUX_default_n_annuli_for_linear": 10,
        "route_policy": (
            "use saved native chosen_grid/chosen_nbin directly; calibrate only "
            "when a native plan is missing; retain polar when native accuracy "
            "fails; allow Cartesian fallback only for supported/accurate plans"
        ),
        "linear_high_resolution_policy": (
            "do not promote a valid high-resolution linear polar plan to Cartesian"
        ),
        "mixed_resolution_retry_policy": (
            "expand polar epochs only; hold Cartesian epochs fixed in linear mixed plans"
        ),
        "source_result": str(source_result),
        "input_report": str(input_report),
        "full_lane_runs": lane_metadata,
        "targeted_policy_fix_runs": targeted_metadata,
    }
    output["configuration"] = copy.deepcopy(source.get("configuration", {}))
    output["configuration"].update(
        {
            "jax_result_status": "corrected_native_routed_all_rows",
            "jax_all_rows_rerun": True,
            "jax_microLUX_comparison_policy": "JAX rerun; original microLUX fields reused",
            "microLUX_rerun": False,
            "microLUX_default_n_annuli_for_linear": 10,
            "jax_cartesian_fallback_policy": output["correction"]["route_policy"],
            "jax_linear_high_resolution_policy": output["correction"]["linear_high_resolution_policy"],
            "jax_mixed_resolution_retry_policy": output["correction"]["mixed_resolution_retry_policy"],
        }
    )

    output_result.parent.mkdir(parents=True, exist_ok=True)
    output_result.write_text(json.dumps(output, indent=2) + "\n")
    print(f"saved {output_result}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
