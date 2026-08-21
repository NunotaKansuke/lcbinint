#!/usr/bin/env python3
"""Attach a fixed-annulus microLUX rerun to the saved JAX comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


def _finite(value):
    try:
        return value is not None and float(value) == float(value)
    except (TypeError, ValueError):
        return False


def _key(row):
    return (
        int(row["case_id"]),
        str(row["profile"]),
        f"{float(row['target']):.17g}",
        f"{float(row['x']):.17g}",
        f"{float(row['y']):.17g}",
    )


def _values(rows, name):
    return [
        float(row[name])
        for row in rows
        if _finite(row.get(name)) and float(row[name]) > 0.0
    ]


def _stats(rows, name):
    values = _values(rows, name)
    if not values:
        return {"count": 0}
    values.sort()
    return {
        "count": len(values),
        "median_seconds": float(statistics.median(values)),
        "p10_seconds": float(
            values[max(0, int(0.1 * (len(values) - 1)))]
        ),
        "p90_seconds": float(
            values[min(len(values) - 1, int(0.9 * (len(values) - 1)))]
        ),
        "minimum_seconds": float(min(values)),
    }


def _ratio_stats(rows, numerator, denominator):
    values = []
    for row in rows:
        left = row.get(numerator)
        right = row.get(denominator)
        if _finite(left) and _finite(right) and float(right) != 0.0:
            values.append(float(left) / float(right))
    if not values:
        return {"count": 0}
    values.sort()
    return {
        "count": len(values),
        "median_ratio": float(statistics.median(values)),
        "p10_ratio": float(values[max(0, int(0.1 * (len(values) - 1)))]),
        "p90_ratio": float(
            values[min(len(values) - 1, int(0.9 * (len(values) - 1)))]
        ),
    }


def _lane_summary(rows, prefix):
    completed = [
        row
        for row in rows
        if row.get(f"{prefix}status") == "completed"
    ]
    fixed_forward = f"{prefix}forward_block_seconds"
    fixed_dadt = f"{prefix}dA_dt_block_seconds"
    default_forward = "microlux_default_forward_block_seconds"
    default_dadt = "microlux_default_dA_dt_block_seconds"
    jax_forward = "jax_forward_block_seconds"
    jax_dadt = "jax_dA_dt_block_seconds"
    native_forward = "native_warmup_chosen_block_seconds"

    def _wins(left, right):
        eligible = [
            row
            for row in completed
            if _finite(row.get(left)) and _finite(row.get(right))
        ]
        return {
            "eligible": len(eligible),
            "left_faster_count": sum(
                float(row[left]) < float(row[right]) for row in eligible
            ),
            "left_faster_fraction": (
                None
                if not eligible
                else sum(float(row[left]) < float(row[right]) for row in eligible)
                / len(eligible)
            ),
        }

    return {
        "jobs": len(rows),
        "fixed_status_counts": {
            status: sum(row.get(f"{prefix}status") == status for row in rows)
            for status in sorted(
                {row.get(f"{prefix}status") for row in rows},
                key=lambda value: "" if value is None else str(value),
            )
        },
        "fixed_completed": len(completed),
        "fixed_accuracy_certified_count": sum(
            bool(row.get(f"{prefix}reference_certified_for_target"))
            for row in completed
        ),
        "fixed_accuracy_pass_count": sum(
            bool(row.get(f"{prefix}passes_reference")) for row in completed
        ),
        "fixed_accuracy_pass_no_budget_warning_count": sum(
            bool(row.get(f"{prefix}passes_reference"))
            and not bool(row.get(f"{prefix}budget_exhausted"))
            for row in completed
        ),
        "fixed_accuracy_pass_with_budget_warning_count": sum(
            bool(row.get(f"{prefix}passes_reference"))
            and bool(row.get(f"{prefix}budget_exhausted"))
            for row in completed
        ),
        "fixed_accuracy_fail_count": sum(
            row.get(f"{prefix}reference_certified_for_target") is True
            and row.get(f"{prefix}passes_reference") is False
            for row in completed
        ),
        "fixed_dA_dt_timeout_count": sum(
            bool(row.get(f"{prefix}dA_dt_timeout")) for row in completed
        ),
        "fixed_budget_exhausted_count": sum(
            bool(row.get(f"{prefix}budget_exhausted")) for row in completed
        ),
        "fixed_forward": _stats(completed, fixed_forward),
        "fixed_dA_dt": _stats(completed, fixed_dadt),
        "fixed_max_relative_error": _stats(
            completed, f"{prefix}max_relative_error"
        ),
        "fixed_dA_dt_max_relative_error": _stats(
            completed, f"{prefix}dA_dt_max_relative_error"
        ),
        "default_forward": _stats(completed, default_forward),
        "default_dA_dt": _stats(completed, default_dadt),
        "jax_forward": _stats(completed, jax_forward),
        "jax_dA_dt": _stats(completed, jax_dadt),
        "native_forward": _stats(completed, native_forward),
        "fixed_over_default_forward": _ratio_stats(
            completed, fixed_forward, default_forward
        ),
        "fixed_over_default_dA_dt": _ratio_stats(
            completed, fixed_dadt, default_dadt
        ),
        "fixed_over_jax_forward": _ratio_stats(
            completed, fixed_forward, jax_forward
        ),
        "fixed_over_jax_dA_dt": _ratio_stats(
            completed, fixed_dadt, jax_dadt
        ),
        "fixed_over_native_forward": _ratio_stats(
            completed, fixed_forward, native_forward
        ),
        "fixed_forward_wins_vs_default": _wins(fixed_forward, default_forward),
        "fixed_dA_dt_wins_vs_default": _wins(fixed_dadt, default_dadt),
        "fixed_forward_wins_vs_jax": _wins(fixed_forward, jax_forward),
        "fixed_dA_dt_wins_vs_jax": _wins(fixed_dadt, jax_dadt),
        "fixed_forward_wins_vs_native": _wins(fixed_forward, native_forward),
    }


def _format_stat(stat, key):
    value = stat.get(key, {}).get("median_seconds")
    return "n/a" if value is None else f"{value:.6g}"


def _report(payload, prefix, fixed_input):
    lines = [
        "# microLUX fixed-annulus rerun",
        "",
        f"Input fixed-annulus result: `{fixed_input}`",
        "",
        "This report keeps the saved native/JAX measurements unchanged and "
        "replaces only the separately measured microLUX lane for the fixed "
        f"linear `n_annuli={payload['configuration']['microlux_fixed_n_annuli']}` "
        "audit. Uniform rows do not use annuli.",
        "",
        "The count is not a literal VBMicrolensing parameter: VBM does not "
        "use the same annular quadrature. It is the fixed microLUX setting "
        "selected from the earlier convergence check.",
        "",
        "| lane | fixed forward p50 (s) | fixed dA/dt p50 (s) | "
        "numeric pass | no-warning pass | budget warnings | "
        "fixed dA/dt timeout | fixed/JAX forward p50 | "
        "fixed/default forward p50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for lane, summary in payload["fixed_annuli_summary"].items():
        lines.append(
            f"| `{lane}` | "
            f"{_format_stat(summary, 'fixed_forward')} | "
            f"{_format_stat(summary, 'fixed_dA_dt')} | "
            f"{summary['fixed_accuracy_pass_count']}/"
            f"{summary['fixed_accuracy_certified_count']} | "
            f"{summary['fixed_accuracy_pass_no_budget_warning_count']}/"
            f"{summary['fixed_accuracy_certified_count']} | "
            f"{summary['fixed_budget_exhausted_count']} | "
            f"{summary['fixed_dA_dt_timeout_count']} | "
            f"{summary['fixed_over_jax_forward'].get('median_ratio', 'n/a')} | "
            f"{summary['fixed_over_default_forward'].get('median_ratio', 'n/a')} |"
        )
    lines.extend(
        [
            "",
            "A pass means all four stored reference epochs are within the "
            "target and the reference uncertainty is sufficiently below the "
            "target. `no-warning pass` additionally excludes microLUX "
            "sampler-budget warnings. The fixed-annulus run is therefore an "
            "accuracy audit, not an assumption that equal annulus counts imply "
            "equal error.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = json.loads(args.base.read_text())
    fixed = json.loads(args.fixed.read_text())
    fixed_rows = {_key(row): row for row in fixed.get("results", ())}
    fixed_n_annuli = int(
        fixed.get("configuration", {}).get("microlux_fixed_n_annuli", 80)
    )
    prefix = f"microlux_fixed{fixed_n_annuli}_"
    merged_rows = []
    missing = []
    for source in base.get("results", ()):
        row = dict(source)
        match = fixed_rows.get(_key(source))
        if match is None:
            missing.append(_key(source))
            row[f"{prefix}status"] = "missing"
        else:
            row[f"{prefix}status"] = match.get("status")
            for key, value in match.items():
                if key.startswith("microlux_"):
                    row[f"{prefix}{key[len('microlux_'):]}"] = value
                elif key in (
                    "reference_available",
                    "reference_certified_for_target",
                    "reference_source",
                ):
                    row[f"{prefix}{key}"] = value
            if match.get("status") != "completed":
                for key in ("error", "timeout_stage", "error_stage"):
                    if key in match:
                        row[f"{prefix}{key}"] = match[key]
        merged_rows.append(row)

    grouped = {}
    for row in merged_rows:
        key = f"{row['profile']}:target={float(row['target']):g}"
        grouped.setdefault(key, []).append(row)
    summary = {
        key: _lane_summary(rows, prefix) for key, rows in sorted(grouped.items())
    }

    merged = dict(base)
    merged["timing_mode"] = (
        "saved_native_jax_plus_microLUX_fixed_annuli_rerun"
    )
    merged["configuration"] = dict(base.get("configuration", {}))
    merged["configuration"].update(
        {
            "microlux_fixed_n_annuli": fixed_n_annuli,
            "microlux_fixed_result": str(args.fixed),
            "microlux_fixed_policy": (
                "fixed requested n_annuli for every linear row; uniform rows "
                "do not use annuli"
            ),
        }
    )
    merged["results"] = merged_rows
    merged["fixed_annuli_summary"] = summary
    merged["fixed_annuli_merge"] = {
        "fixed_input": str(args.fixed),
        "fixed_rows": len(fixed_rows),
        "base_rows": len(merged_rows),
        "missing_rows": len(missing),
        "missing_keys": missing,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2) + "\n")
    report_path = args.output.with_name(
        f"REPORT_microlux_fixed{fixed_n_annuli}_{args.output.stem}.md"
    )
    report_path.write_text(_report(merged, prefix, args.fixed) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"saved {args.output}", flush=True)
    print(f"saved {report_path}", flush=True)


if __name__ == "__main__":
    main()
