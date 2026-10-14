#!/usr/bin/env python3
"""Summarize the D14 root-work A/B trajectory benchmark.

The input TSV is emitted by bench_d14_positive_trajectory.cpp.  This script
keeps the timing boundary explicit: ``cold_ms`` and ``warm_ms`` include the
complete epoch path, while the ``*_stage_ms`` fields are reported separately.
It deliberately does not turn a successful run into an accuracy claim; the
reference column and status/stop counts are retained for independent review.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, TextIO


LANES = ("cold", "warm", "radial")
PERCENTILES = (50, 90, 95, 99, 100)
STAGE_FIELDS = ("topology", "adaptive", "setup", "physical", "estimator", "scheduler")


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def load_rows(path: Path) -> list[dict[str, str]]:
    with open_text(path) as stream:
        lines = (line for line in stream if line.strip() and not line.startswith("#"))
        return list(csv.DictReader(lines, delimiter=" "))


def percentile(values: Iterable[float], p: float) -> float | None:
    data = sorted(v for v in values if math.isfinite(v))
    if not data:
        return None
    x = (len(data) - 1) * p / 100.0
    lo = int(x)
    if lo + 1 >= len(data):
        return data[-1]
    return data[lo] + (data[lo + 1] - data[lo]) * (x - lo)


def finite_float(row: dict[str, str], key: str) -> float | None:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def timing_summary(rows: list[dict[str, str]], field: str) -> dict[str, float | None]:
    values = [value for row in rows if (value := finite_float(row, field)) is not None]
    return {f"p{p}": percentile(values, p) for p in PERCENTILES}


def lane_summary(rows: list[dict[str, str]], lane: str) -> dict[str, object]:
    value_key = f"{lane}_value_converged"
    status_key = f"{lane}_status"
    stop_key = f"{lane}_stop"
    mu_key = f"{lane}_mu"
    error_key = f"{lane}_error"

    relative_errors: list[float] = []
    for row in rows:
        mu = finite_float(row, mu_key)
        ref = finite_float(row, "reference")
        if mu is not None and ref is not None:
            relative_errors.append(abs(mu - ref) / max(abs(ref), 1e-300))

    stage = {
        f"{name}_ms": timing_summary(rows, f"{lane}_{name}_ms")
        for name in STAGE_FIELDS
        if f"{lane}_{name}_ms" in rows[0]
    }
    return {
        "rows": len(rows),
        "whole_ms": timing_summary(rows, f"{lane}_ms"),
        "value_converged": sum(row.get(value_key) == "1" for row in rows),
        "status_counts": dict(Counter(row.get(status_key, "") for row in rows)),
        "stop_counts": dict(Counter(row.get(stop_key, "") for row in rows)),
        "relative_error_to_input_reference": {
            "n": len(relative_errors),
            **timing_summary(
                [{"x": str(value)} for value in relative_errors], "x"
            ),
        },
        "stages": stage,
    }


def summarize_variant(path: Path) -> dict[str, object]:
    rows = load_rows(path)
    by_target: dict[str, object] = {}
    slow_rows: dict[str, object] = {}
    for target, label in (("0.001", "1e-3"), ("0.0001", "1e-4")):
        selected = [row for row in rows if row.get("target") == target]
        by_target[label] = {
            "target": float(target),
            "rows": len(selected),
            "lanes": {lane: lane_summary(selected, lane) for lane in LANES},
        }
        slow_rows[label] = {}
        for lane in LANES:
            ordered = sorted(
                selected,
                key=lambda row: finite_float(row, f"{lane}_ms") or -math.inf,
                reverse=True,
            )
            slow_rows[label][lane] = [
                {
                    key: row.get(key)
                    for key in (
                        "case_id", "configuration_id", "profile", "d_bin_index",
                        "epoch_index", "trajectory_id", "trajectory_pos", "target",
                        f"{lane}_ms", f"{lane}_topology_ms", f"{lane}_adaptive_ms",
                        f"{lane}_status", f"{lane}_stop", f"{lane}_nodes",
                        f"{lane}_evaluations", "topology_events", "topology_cells",
                        "warm_l2", "warm_l3", "warm_seed_used",
                    )
                }
                for row in ordered[:10]
            ]
    return {
        "path": str(path),
        "rows": len(rows),
        "by_target": by_target,
        "slow_rows": slow_rows,
    }


def compare_variants(
    baseline: dict[str, object], candidate: dict[str, object]
) -> dict[str, object]:
    result: dict[str, object] = {}
    for target in ("1e-3", "1e-4"):
        result[target] = {}
        base_target = baseline["by_target"][target]
        cand_target = candidate["by_target"][target]
        for lane in LANES:
            base_timing = base_target["lanes"][lane]["whole_ms"]
            cand_timing = cand_target["lanes"][lane]["whole_ms"]
            speedup: dict[str, float | None] = {}
            for key in ("p50", "p90", "p95", "p99", "p100"):
                old = base_timing.get(key)
                new = cand_timing.get(key)
                speedup[key] = old / new if old is not None and new else None
            result[target][lane] = {
                "baseline_ms": base_timing,
                "candidate_ms": cand_timing,
                "baseline_over_candidate": speedup,
                "candidate_value_converged_delta": (
                    cand_target["lanes"][lane]["value_converged"]
                    - base_target["lanes"][lane]["value_converged"]
                ),
            }
    return result


def pairwise_output_check(baseline_path: Path, candidate_path: Path) -> dict[str, object]:
    baseline_rows = load_rows(baseline_path)
    candidate_rows = load_rows(candidate_path)
    key_fields = (
        "case_id", "configuration_id", "profile", "d_bin_index",
        "epoch_index", "target",
    )
    baseline_by_key = {tuple(row.get(key, "") for key in key_fields): row
                       for row in baseline_rows}
    candidate_by_key = {tuple(row.get(key, "") for key in key_fields): row
                        for row in candidate_rows}
    common_keys = sorted(baseline_by_key.keys() & candidate_by_key.keys())
    output: dict[str, object] = {
        "matched_rows": len(common_keys),
        "baseline_rows": len(baseline_rows),
        "candidate_rows": len(candidate_rows),
        "by_target_lane": {},
    }
    for target, label in (("0.001", "1e-3"), ("0.0001", "1e-4")):
        output["by_target_lane"][label] = {}
        for lane in LANES:
            rel_diffs: list[float] = []
            abs_diffs: list[float] = []
            convergence_mismatches = 0
            status_mismatches = 0
            both_converged = 0
            for key in common_keys:
                if key[-1] != target:
                    continue
                old = baseline_by_key[key]
                new = candidate_by_key[key]
                convergence_mismatches += (
                    old.get(f"{lane}_value_converged")
                    != new.get(f"{lane}_value_converged")
                )
                status_mismatches += old.get(f"{lane}_status") != new.get(f"{lane}_status")
                if old.get(f"{lane}_value_converged") != "1" or \
                        new.get(f"{lane}_value_converged") != "1":
                    continue
                old_mu = finite_float(old, f"{lane}_mu")
                new_mu = finite_float(new, f"{lane}_mu")
                if old_mu is None or new_mu is None:
                    continue
                both_converged += 1
                abs_diffs.append(abs(old_mu - new_mu))
                rel_diffs.append(
                    abs(old_mu - new_mu) / max(abs(old_mu), abs(new_mu), 1e-300)
                )
            output["by_target_lane"][label][lane] = {
                "both_converged_rows": both_converged,
                "value_convergence_mismatches": convergence_mismatches,
                "status_mismatches": status_mismatches,
                "absolute_mu_difference": {
                    "p50": percentile(abs_diffs, 50),
                    "p99": percentile(abs_diffs, 99),
                    "max": percentile(abs_diffs, 100),
                },
                "relative_mu_difference": {
                    "p50": percentile(rel_diffs, 50),
                    "p90": percentile(rel_diffs, 90),
                    "p99": percentile(rel_diffs, 99),
                    "max": percentile(rel_diffs, 100),
                },
            }
    return output


PROFILE_RE = re.compile(
    r"^PROFILE (?P<name>\S+) cases=(?P<cases>\d+) wall_ms "
    r"p50=(?P<p50>\S+) p90=(?P<p90>\S+) p95=(?P<p95>\S+) "
    r"p99=(?P<p99>\S+) max=(?P<max>\S+) status_ok=(?P<ok>\d+) "
    r"status_bad=(?P<bad>\d+)"
)


def parse_profile(path: Path) -> dict[str, object]:
    blocks: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            match = PROFILE_RE.match(line.rstrip())
            if match:
                current = {
                    "name": match.group("name"),
                    "cases": int(match.group("cases")),
                    "whole_ms": {
                        key: float(match.group(key))
                        for key in ("p50", "p90", "p95", "p99", "max")
                    },
                    "status_ok": int(match.group("ok")),
                    "status_bad": int(match.group("bad")),
                    "raw": [line.rstrip()],
                }
                blocks.append(current)
            elif current is not None and line.strip():
                current["raw"].append(line.rstrip())
    return {"path": str(path), "blocks": blocks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--baseline", default="baseline_current.tsv")
    args = parser.parse_args()

    root = args.root
    paths = sorted(
        path
        for pattern in ("*.tsv", "*.tsv.gz")
        for path in root.glob(pattern)
        if path.name != "input_snapshot.tsv"
    )
    variants = {
        path.stem.removesuffix(".tsv"): summarize_variant(path)
        for path in paths
    }
    baseline_name = Path(args.baseline).name
    baseline_name = baseline_name.removesuffix(".gz").removesuffix(".tsv")
    comparisons: dict[str, object] = {}
    pairwise: dict[str, object] = {}
    if baseline_name in variants:
        for name, variant in variants.items():
            if name != baseline_name:
                comparisons[name] = compare_variants(variants[baseline_name], variant)
                pairwise[name] = pairwise_output_check(
                    Path(variants[baseline_name]["path"]), Path(variant["path"])
                )

    profiles_dir = root / "profiles"
    profiles = []
    if profiles_dir.is_dir():
        profiles = [parse_profile(path) for path in sorted(profiles_dir.glob("*.log"))]

    print(json.dumps({
        "schema": 1,
        "metadata": {
            "input_format": "space-delimited trajectory TSV",
            "percentiles": list(PERCENTILES),
            "timing_scope": "cold_ms/warm_ms/radial_ms include the complete measured lane",
            "accuracy_reference": "input reference column; not an independent acceptance oracle",
        },
        "variants": variants,
        "comparisons_to_baseline": {
            "baseline": baseline_name,
            "variants": comparisons,
            "pairwise_output_checks": pairwise,
        },
        "profiles": profiles,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
