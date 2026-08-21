#!/usr/bin/env python3
"""Attach the VBM-event-annulus microLUX rerun to the saved comparison."""

from __future__ import annotations

import argparse
from collections import Counter
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


def _numeric(values):
    values = sorted(float(value) for value in values if _finite(value))
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "median": float(statistics.median(values)),
        "p10": float(values[max(0, int(0.1 * (len(values) - 1)))]),
        "p90": float(values[min(len(values) - 1, int(0.9 * (len(values) - 1)))]),
        "minimum": float(values[0]),
        "maximum": float(values[-1]),
    }


def _stats(rows, field):
    return _numeric(row.get(field) for row in rows)


def _ratio_stats(rows, left, right):
    return _numeric(
        float(row[left]) / float(row[right])
        for row in rows
        if _finite(row.get(left))
        and _finite(row.get(right))
        and float(row[right]) != 0.0
    )


def _wins(rows, left, right):
    eligible = [
        row
        for row in rows
        if _finite(row.get(left)) and _finite(row.get(right))
    ]
    wins = sum(float(row[left]) < float(row[right]) for row in eligible)
    return {
        "eligible": len(eligible),
        "left_faster_count": wins,
        "left_faster_fraction": None if not eligible else wins / len(eligible),
    }


def _counts(values):
    counts = Counter(str(int(value)) for value in values if _finite(value))
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def _lane_summary(rows, prefix):
    status_field = f"{prefix}status"
    completed = [row for row in rows if row.get(status_field) == "completed"]
    event_forward = f"{prefix}forward_block_seconds"
    event_dadt = f"{prefix}dA_dt_block_seconds"
    event_n = "vbm_event_n_annuli"
    event_epoch_n = "vbm_nannuli_per_epoch"
    default_forward = "microlux_default_forward_block_seconds"
    default_dadt = "microlux_default_dA_dt_block_seconds"
    jax_forward = "jax_forward_block_seconds"
    jax_dadt = "jax_dA_dt_block_seconds"
    native_forward = "native_warmup_chosen_block_seconds"
    epoch_counts = _counts(
        value
        for row in completed
        for value in (row.get(event_epoch_n) or ())
    )
    return {
        "jobs": len(rows),
        "event_status_counts": {
            status: sum(row.get(status_field) == status for row in rows)
            for status in sorted(
                {row.get(status_field) for row in rows},
                key=lambda value: "" if value is None else str(value),
            )
        },
        "event_completed": len(completed),
        "vbm_event_n_annuli_counts": _counts(
            row.get(event_n) for row in completed
        ),
        "vbm_epoch_nannuli_counts": epoch_counts,
        "vbm_event_n_annuli": _stats(completed, event_n),
        "event_accuracy_certified_count": sum(
            bool(row.get(f"{prefix}reference_certified_for_target"))
            for row in completed
        ),
        "event_accuracy_pass_count": sum(
            bool(row.get(f"{prefix}passes_reference")) for row in completed
        ),
        "event_accuracy_pass_no_warning_count": sum(
            bool(row.get(f"{prefix}passes_reference"))
            and not bool(row.get(f"{prefix}budget_exhausted"))
            for row in completed
        ),
        "event_accuracy_fail_count": sum(
            row.get(f"{prefix}reference_certified_for_target") is True
            and row.get(f"{prefix}passes_reference") is False
            for row in completed
        ),
        "event_dA_dt_timeout_count": sum(
            bool(row.get(f"{prefix}dA_dt_timeout")) for row in completed
        ),
        "event_budget_exhausted_count": sum(
            bool(row.get(f"{prefix}budget_exhausted")) for row in completed
        ),
        "event_forward": _stats(completed, event_forward),
        "event_dA_dt": _stats(completed, event_dadt),
        "event_max_relative_error": _stats(
            completed, f"{prefix}max_relative_error"
        ),
        "default_forward": _stats(completed, default_forward),
        "default_dA_dt": _stats(completed, default_dadt),
        "jax_forward": _stats(completed, jax_forward),
        "jax_dA_dt": _stats(completed, jax_dadt),
        "native_forward": _stats(completed, native_forward),
        "event_over_default_forward": _ratio_stats(
            completed, event_forward, default_forward
        ),
        "event_over_default_dA_dt": _ratio_stats(
            completed, event_dadt, default_dadt
        ),
        "event_over_jax_forward": _ratio_stats(
            completed, event_forward, jax_forward
        ),
        "event_over_jax_dA_dt": _ratio_stats(
            completed, event_dadt, jax_dadt
        ),
        "event_over_native_forward": _ratio_stats(
            completed, event_forward, native_forward
        ),
        "event_forward_wins_vs_default": _wins(
            completed, event_forward, default_forward
        ),
        "event_dA_dt_wins_vs_default": _wins(
            completed, event_dadt, default_dadt
        ),
        "event_forward_wins_vs_jax": _wins(
            completed, event_forward, jax_forward
        ),
        "event_dA_dt_wins_vs_jax": _wins(
            completed, event_dadt, jax_dadt
        ),
        "event_forward_wins_vs_native": _wins(
            completed, event_forward, native_forward
        ),
    }


def _fmt(stat, key="median"):
    value = stat.get(key)
    return "n/a" if value is None else f"{value:.6g}"


def _fmt_ratio(stat):
    value = stat.get("median")
    return "n/a" if value is None else f"{value:.4g}x"


def _fmt_fraction(wins):
    if not wins["eligible"]:
        return "n/a"
    return f"{wins['left_faster_count']}/{wins['eligible']} ({wins['left_faster_fraction']:.1%})"


def _report(payload, prefix, event_input):
    lines = [
        "# microLUX VBM event-annulus rerun",
        "",
        f"Input event-annulus result: `{event_input}`",
        "",
        "For each linear row, VBMicrolensing was called at the four batched "
        "epochs with the same `Tol=1e-12` and `RelTol=target`.  The maximum "
        "of the four final VBM `nannuli` values was passed to microLUX as the "
        "single static `n_annuli` for that event. Uniform rows do not use "
        "annuli.",
        "",
        "All reported microLUX times are after compilation/warm-up. Compilation "
        "was isolated one static annulus value per child process to prevent "
        "XLA/LLVM cache growth; it is not included in the timing samples.",
        "",
        "| lane | VBM event n p10/median/p90 | event forward p50 (s) | "
        "event dA/dt p50 (s) | event/default forward | event/JAX forward | "
        "event/native forward | accuracy pass/certified | dA/dt timeout |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for lane, summary in payload["vbm_event_annuli_summary"].items():
        annuli = summary["vbm_event_n_annuli"]
        lines.append(
            f"| `{lane}` | "
            f"{_fmt(annuli, 'p10')}/"
            f"{_fmt(annuli)}/"
            f"{_fmt(annuli, 'p90')} | "
            f"{_fmt(summary['event_forward'])} | "
            f"{_fmt(summary['event_dA_dt'])} | "
            f"{_fmt_ratio(summary['event_over_default_forward'])} | "
            f"{_fmt_ratio(summary['event_over_jax_forward'])} | "
            f"{_fmt_ratio(summary['event_over_native_forward'])} | "
            f"{summary['event_accuracy_pass_count']}/"
            f"{summary['event_accuracy_certified_count']} | "
            f"{summary['event_dA_dt_timeout_count']} |"
        )
    lines.extend(
        [
            "",
            "| lane | event faster than default forward | event faster than "
            "JAX forward | event faster than native forward | event faster "
            "than default dA/dt |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for lane, summary in payload["vbm_event_annuli_summary"].items():
        lines.append(
            f"| `{lane}` | "
            f"{_fmt_fraction(summary['event_forward_wins_vs_default'])} | "
            f"{_fmt_fraction(summary['event_forward_wins_vs_jax'])} | "
            f"{_fmt_fraction(summary['event_forward_wins_vs_native'])} | "
            f"{_fmt_fraction(summary['event_dA_dt_wins_vs_default'])} |"
        )
    lines.extend(
        [
            "",
            "The accuracy pass is the existing four-epoch VBM-reference test; "
            "dA/dt accuracy is intentionally not reported because the corpus "
            "does not contain an independent derivative reference. Budget "
            "warnings are retained in the JSON for audit rather than hidden "
            "by a fallback.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--event", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = json.loads(args.base.read_text())
    event = json.loads(args.event.read_text())
    event_rows = {_key(row): row for row in event.get("results", ())}
    prefix = "microlux_vbm_event_"
    merged_rows = []
    missing = []
    for source in base.get("results", ()):
        row = dict(source)
        match = event_rows.get(_key(source))
        if match is None:
            missing.append(_key(source))
            row[f"{prefix}status"] = "missing"
        else:
            row[f"{prefix}status"] = match.get("status")
            for key, value in match.items():
                if key.startswith("microlux_"):
                    row[f"{prefix}{key[len('microlux_'):]}" ] = value
                elif key.startswith("vbm_"):
                    row[key] = value
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
        key: _lane_summary(rows, prefix)
        for key, rows in sorted(grouped.items())
    }
    merged = dict(base)
    merged["timing_mode"] = (
        "saved_native_jax_plus_microLUX_vbm_event_annuli_rerun"
    )
    merged["configuration"] = dict(base.get("configuration", {}))
    merged["configuration"].update(
        {
            "microlux_vbm_event_result": str(args.event),
            "microlux_vbm_event_annuli_policy": (
                "per linear event, use max VBM final nannuli across the four "
                "batched epochs as one static microLUX n_annuli"
            ),
        }
    )
    merged["results"] = merged_rows
    merged["vbm_event_annuli_summary"] = summary
    merged["vbm_event_annuli_merge"] = {
        "event_input": str(args.event),
        "event_rows": len(event_rows),
        "base_rows": len(merged_rows),
        "missing_rows": len(missing),
        "missing_keys": missing,
        "event_group_count": len(event.get("event_groups", ())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2) + "\n")
    report_path = args.output.with_name(
        f"REPORT_microlux_vbm_event_annuli_{args.output.stem}.md"
    )
    report_path.write_text(_report(merged, prefix, args.event) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"saved {args.output}", flush=True)
    print(f"saved {report_path}", flush=True)


if __name__ == "__main__":
    main()
