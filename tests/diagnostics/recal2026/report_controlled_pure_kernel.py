#!/usr/bin/env python3
"""Merge and report the balanced pure-kernel benchmark parts.

The timing harness is run once per measured ``d/rho`` stratum so that each
stratum can be pinned to its own CPU.  This script joins those parts back to
the corpus, records the achieved (rather than requested) ``d/rho``, and
produces the paper-facing summary.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


PROFILES = ("uniform", "linear")
TARGETS = (1.0e-3, 1.0e-4)
D_BINS = (
    (0.0, 0.4),
    (0.4, 0.8),
    (0.8, 1.2),
    (1.2, 1.6),
    (1.6, 2.0),
)
D_FACTORS = (0.2, 0.6, 1.0, 1.4, 1.8)


def _finite(value):
    return value is not None and math.isfinite(float(value))


def _key(case_id, profile, x, y):
    return (int(case_id), str(profile), f"{float(x):.17g}", f"{float(y):.17g}")


def _factor_key(value):
    return round(float(value), 12)


def _join_corpus(results, corpus_rows):
    lookup = {
        _key(row["case_id"], row["profile"], row["x"], row["y"]): row
        for row in corpus_rows
    }
    factor_lookup = {
        (int(row["case_id"]), str(row["profile"]),
         _factor_key(row["requested_factor"])): row
        for row in corpus_rows
    }
    bin_lookup = {
        (int(row["case_id"]), str(row["profile"]), int(row["d_bin_index"])): row
        for row in corpus_rows
    }
    factor_to_bin = {
        _factor_key(factor): index
        for index, factor in enumerate(D_FACTORS)
    }
    joined = []
    missing = []
    for result in results:
        has_coordinates = result.get("x") is not None and result.get("y") is not None
        key = None
        source = None
        if has_coordinates:
            key = _key(
                result.get("case_id"), result.get("profile"),
                result["x"], result["y"],
            )
            source = lookup.get(key)
        if source is None:
            factor = result.get("d_over_rho", result.get("requested_factor"))
            if factor is not None:
                case_profile = (int(result["case_id"]), str(result["profile"]))
                source = bin_lookup.get(case_profile + (
                    factor_to_bin.get(_factor_key(factor), -1),
                ))
                if source is None:
                    source = factor_lookup.get(case_profile + (
                        _factor_key(factor),
                    ))
        if source is None:
            missing.append(key or (
                result.get("case_id"), result.get("profile"),
                result.get("d_over_rho"),
            ))
            joined.append(result)
            continue
        joined_row = {
            **result,
            "x": result.get("x", source.get("x")),
            "y": result.get("y", source.get("y")),
            "actual_d_over_rho": source.get("actual_d_over_rho"),
            "d_bin_index": source.get("d_bin_index"),
            "d_bin_low": source.get("d_bin_low"),
            "d_bin_high": source.get("d_bin_high"),
            "requested_factor": source.get("requested_factor"),
        }
        joined_row.pop("reference_floor", None)
        joined.append(joined_row)
    if missing:
        raise RuntimeError(f"could not join {len(missing)} timing rows to corpus")
    return joined


def _ratio_values(rows):
    return [
        float(value)
        for row in rows
        for value in row.get("ratios_vbm_over_lcbinint", ())
        if _finite(value) and float(value) > 0.0
    ]


def _stats(values):
    values = np.asarray(values, dtype=float)
    if not values.size:
        return {"count": 0}
    return {
        "count": int(values.size),
        "win_rate": float(np.mean(values > 1.0)),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }


def _summary(rows):
    ratios = _ratio_values(rows)
    statuses = [
        status for row in rows for status in row.get("ratio_status", ())
    ]
    reference_count = max(
        [
            len(row.get("ratio_status", ()))
            for row in rows
            if row.get("ratio_status")
        ]
        or [0]
    )
    point_count = reference_count * len(rows)
    timeout_jobs = sum(row.get("status") != "completed" for row in rows)
    counts = Counter(statuses)
    nbin = [
        int(value)
        for row in rows
        for value in row.get("chosen_nbin", ())
        if value is not None
    ]
    mismatch_flags = [
        bool(value)
        for row in rows
        for value in row.get("vbm_mismatch", ())
        if value is not None
    ]
    self_statuses = Counter()
    for row in rows:
        for grid in row.get("grid", {}).values():
            for sample in grid.get("samples", {}).values():
                status = sample.get("status")
                if status is not None:
                    self_statuses[status] += 1
    return {
        "jobs": len(rows),
        "points": point_count,
        "measured": len(ratios),
        "grid_wins": int(sum(value > 1.0 for value in ratios)),
        "vbm_wins": int(sum(value <= 1.0 for value in ratios)),
        "unresolved": point_count - len(ratios),
        "timeout_jobs": int(timeout_jobs),
        "ratio": _stats(ratios),
        "nbin": _stats(nbin),
        "status_counts": dict(counts),
        "vbm_mismatch": int(sum(mismatch_flags)),
        "vbm_mismatch_points": len(mismatch_flags),
        "vbm_mismatch_rate": (
            float(np.mean(mismatch_flags)) if mismatch_flags else float("nan")
        ),
        "self_status_counts": dict(self_statuses),
    }


def _select(rows, profile, target, d_bin_index=None):
    selected = [
        row for row in rows
        if row.get("profile") == profile
        and abs(float(row.get("target")) - target) < 1.0e-15
    ]
    if d_bin_index is not None:
        selected = [
            row for row in selected
            if int(row.get("d_bin_index", -1)) == int(d_bin_index)
        ]
    return selected


def _merge(parts, corpus, output):
    corpus_payload = json.loads((corpus / "rows.json").read_text())
    corpus_rows = corpus_payload["rows"]
    payloads = [json.loads(path.read_text()) for path in parts]
    results = []
    for payload in payloads:
        results.extend(payload.get("results", ()))
    results = _join_corpus(results, corpus_rows)
    if len(results) != len(set(
        (r.get("case_id"), r.get("profile"), r.get("target"),
         r.get("x"), r.get("y")) for r in results
    )):
        raise RuntimeError("duplicate timing rows in part files")
    first = payloads[0]
    max_source_bins_by_factor = {}
    for payload in payloads:
        factor_values = payload.get("factors", ())
        if not factor_values:
            continue
        factor = f"{float(factor_values[0]):g}"
        # Parts made before the CLI field was added used the historical cap.
        max_source_bins_by_factor[factor] = int(
            payload.get("max_source_bins") or 400
        )
    merged = {
        "corpus": str(corpus),
        "parts": [str(path) for path in parts],
        "input": first.get("input"),
        "case_count": first.get("case_count"),
        "factors": [
            float(payload.get("factors", [float("nan")])[0])
            for payload in payloads
        ],
        "profiles": list(first.get("profiles", PROFILES)),
        "targets": list(first.get("targets", TARGETS)),
        "repeats": first.get("repeats"),
        "search_missing": first.get("search_missing"),
        "max_source_bins": max(max_source_bins_by_factor.values()),
        "max_source_bins_by_factor": max_source_bins_by_factor,
        "point_timeout": first.get("point_timeout"),
        "search_point_timeout": first.get("search_point_timeout"),
        "job_timeout": first.get("job_timeout"),
        "route_filter": first.get("route_filter"),
        "timing_mode": first.get("timing_mode"),
        "coordinate_mode": first.get("coordinate_mode"),
        "point_source_hint_mode": first.get("point_source_hint_mode"),
        "resolution_mode": first.get("resolution_mode"),
        "self_confirmation_points": first.get("self_confirmation_points"),
        "reference_mode": first.get("reference_mode"),
        "build_extension": first.get("build_extension"),
        "reference_indices": first.get("reference_indices"),
        "results": results,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(merged, indent=2))
    return merged, corpus_payload


def _make_report(merged, corpus_payload):
    rows = merged["results"]
    parameter_sampling = corpus_payload["manifest"] if "manifest" in corpus_payload else {}
    manifest = corpus_payload.get("_manifest", {})
    configs = manifest.get("configurations", ())
    all_summaries = {
        profile: {
            str(target): _summary(_select(rows, profile, target))
            for target in TARGETS
        }
        for profile in PROFILES
    }
    bin_summaries = {
        profile: {
            str(target): [
                _summary(_select(rows, profile, target, index))
                for index in range(len(D_BINS))
            ]
            for target in TARGETS
        }
        for profile in PROFILES
    }
    lines = [
        "# Controlled pure-kernel speed comparison",
        "",
        "This report uses the balanced corpus with independent log-uniform lens",
        "parameters and equal-width bins in the measured caustic distance",
        "`d/rho`. The timing is the cache-warm pure finite-source kernel only.",
        "",
        "## Benchmark design",
        "",
        "- 160 independent binary-lens configurations; `s`, `q`, and `rho` are",
        "  independently log-uniform over `[0.2, 4]`, `[1e-4, 1]`, and",
        "  `[3e-5, 1]`, respectively.",
        "- For every configuration, one source position is accepted in each",
        "  measured `d/rho` bin: `[0,0.4)`, `[0.4,0.8)`, `[0.8,1.2)`,",
        "  `[1.2,1.6)`, and `[1.6,2]`. This gives 800 positions and 1600",
        "  profile rows.",
        "- Each position is evaluated for a uniform source and linear limb",
        "  darkening with `c=0.5`.",
        "- The requested relative tolerances are `1e-3` and `1e-4`.",
        "- lcbinint increases Nbin independently of VBM and selects the first",
        "  Nbin in a run of three increasing grid values whose relative spread",
        "  is within the requested tolerance. The native `support_proven`",
        "  certificate is intentionally not used in this pure-kernel diagnostic.",
        "- The maximum source-grid resolution is `Nbin=400`; a point that does",
        "  not self-converge by that cap is recorded as unresolved.",
        "- Each candidate self-search point has a separate timeout; timeout",
        "  points remain in the result with an explicit unresolved status.",
        "- VBMicrolensing is evaluated once per reference epoch at the requested",
        "  `RelTol`. Its value is not used to choose Nbin. The selected lcbinint",
        "  value is compared to that VBM value afterward, and disagreement is",
        "  retained as `vbm_mismatch` rather than filtered out.",
        "- Search time is excluded from the kernel timing. Cartesian and polar",
        "  candidates are both kept and the faster self-converged one is selected",
        "  per epoch.",
        "- The harness uses `route-filter=all`: this is a direct integrator",
        "  comparison, not a production-route win-rate measurement.",
        "- Each timing process was pinned to one physical CPU core with",
        "  `OMP_NUM_THREADS=1`; repeated samples use the cache-warm native kernel",
        "  timing protocol.",
        "",
        "## Reference handling",
        "",
        f"The corpus contains {len(corpus_payload['rows'])} rows. No stored",
        "high-precision VBM reference or reference-floor filter is used in this",
        "run. Each target has its own single-call VBM reference and its own",
        "lcbinint self-convergence search.",
        "",
        "## Overall timing summary",
        "",
        "`R = t_VBM / t_lcbinint`; `R > 1` means lcbinint is faster.",
        "",
        "| profile | target | jobs | measured points | lcbinint wins | VBM wins | unresolved points | timeout jobs | VBM mismatch | win rate | median R | p10 | p90 | median Nbin |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for profile in PROFILES:
        for target in TARGETS:
            item = all_summaries[profile][str(target)]
            ratio = item["ratio"]
            nbin = item["nbin"]
            lines.append(
                f"| {profile} | `{target:g}` | {item['jobs']} | {item['measured']} | "
                f"{item['grid_wins']} | {item['vbm_wins']} | {item['unresolved']} | "
                f"{item['timeout_jobs']} | {item['vbm_mismatch']}/{item['vbm_mismatch_points']} | "
                f"{ratio.get('win_rate', float('nan')):.1%} | "
                f"{ratio.get('median', float('nan')):.3f} | "
                f"{ratio.get('p10', float('nan')):.3f} | "
                f"{ratio.get('p90', float('nan')):.3f} | "
                f"{nbin.get('median', float('nan')):.0f} |"
            )
    lines += [
        "",
        "## Equal-width d/rho strata",
        "",
        "| profile | target | d/rho bin | jobs | measured | lcbinint wins | VBM wins | unresolved points | timeout jobs | VBM mismatch | win rate | median R |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for profile in PROFILES:
        for target in TARGETS:
            for index, (low, high) in enumerate(D_BINS):
                item = bin_summaries[profile][str(target)][index]
                ratio = item["ratio"]
                high_bracket = "]" if index == len(D_BINS) - 1 else ")"
                label = f"[{low:g}, {high:g}{high_bracket}"
                lines.append(
                    f"| {profile} | `{target:g}` | `{label}` | {item['jobs']} | "
                    f"{item['measured']} | {item['grid_wins']} | {item['vbm_wins']} | "
                    f"{item['unresolved']} | {item['timeout_jobs']} | "
                    f"{item['vbm_mismatch']}/{item['vbm_mismatch_points']} | "
                    f"{ratio.get('win_rate', float('nan')):.1%} | "
                    f"{ratio.get('median', float('nan')):.3f} |"
                )
    lines += [
        "",
        "## Self-convergence search diagnostics",
        "",
        "Counts below are candidate epoch evaluations across both Cartesian and",
        "polar searches. A self-unresolved candidate is retained in the raw",
        "results; it is not silently treated as a VBM mismatch.",
        "",
        "| profile | target | self-converged | self-unresolved | self-timeout |",
        "|---|---:|---:|---:|---:|",
    ]
    for profile in PROFILES:
        for target in TARGETS:
            item = all_summaries[profile][str(target)]
            counts = item["self_status_counts"]
            lines.append(
                f"| {profile} | `{target:g}` | "
                f"{counts.get('self_converged', 0)} | "
                f"{counts.get('self_unresolved', 0)} | "
                f"{counts.get('self_timeout', 0)} |"
            )
    return "\n".join(lines) + "\n", all_summaries, bin_summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--parts", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((args.corpus / "manifest.json").read_text())
    corpus_rows = json.loads((args.corpus / "rows.json").read_text())
    corpus_payload = {"_manifest": manifest, **corpus_rows}
    merged, _ = _merge(args.parts, args.corpus, args.output)
    report, summaries, bin_summaries = _make_report(merged, corpus_payload)
    (args.output / "REPORT_controlled_pure_kernel.md").write_text(report)
    (args.output / "summary.json").write_text(json.dumps({
        "overall": summaries,
        "by_d_over_rho_bin": bin_summaries,
    }, indent=2))
    print(report)


if __name__ == "__main__":
    main()
