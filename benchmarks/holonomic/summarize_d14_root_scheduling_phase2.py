#!/usr/bin/env python3
"""Summarize Phase 2 D14 root-scheduling evidence without mixing scopes."""

from __future__ import annotations

import argparse
import collections
import gzip
import itertools
import json
import math
import statistics
from pathlib import Path

from summarize_d14_root_optimization import (
    compare_variants,
    pairwise_output_check,
    parse_profile,
    summarize_variant,
)


def read_ws(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        header = stream.readline().split()
        for line in stream:
            values = line.split()
            if len(values) == len(header):
                yield dict(zip(header, values))


def numbers(rows, field):
    out = []
    for row in rows:
        try:
            value = float(row[field])
        except (KeyError, ValueError):
            continue
        if math.isfinite(value):
            out.append(value)
    return out


def stats(values):
    values = sorted(value for value in values if math.isfinite(value))
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "p50": statistics.median(values),
        "p90": values[int(0.90 * (len(values) - 1))],
        "p99": values[int(0.99 * (len(values) - 1))],
        "max": values[-1],
    }


def minimum_assignment(cost):
    """Hungarian minimum-cost bijection for the fixed 14-root set."""
    n = len(cost)
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [math.inf] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = math.inf
            j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = [-1] * n
    for j in range(1, n + 1):
        if p[j] > 0:
            assignment[p[j] - 1] = j - 1
    return assignment


def root_set_parity(base_path: Path, scheduled_path: Path):
    group_fields = ("case_id", "configuration_id", "profile", "d_bin_index",
                    "epoch_index", "rep", "lane")
    distances = []
    rows = groups = key_mismatches = incomplete_groups = 0
    maximum_displacement = None
    open_rows = lambda path: gzip.open(path, "rt", encoding="utf-8") \
        if path.suffix == ".gz" else path.open(encoding="utf-8")
    with open_rows(base_path) as base, open_rows(scheduled_path) as scheduled:
        base_header = base.readline().split()
        candidate_header = scheduled.readline().split()
        if base_header != candidate_header:
            return {"error": "root record headers differ"}
        base_i = {name: i for i, name in enumerate(base_header)}
        group_key = None
        left = []
        right = []
        left_root_indices = []
        right_root_indices = []

        def finish_group():
            nonlocal groups, incomplete_groups, maximum_displacement
            if not left:
                return
            groups += 1
            if len(left) != 14 or len(right) != 14:
                incomplete_groups += 1
                return
            cost = [[math.hypot(a[0] - b[0], a[1] - b[1])
                     for b in right] for a in left]
            assignment = minimum_assignment(cost)
            for i, j in enumerate(assignment):
                d = cost[i][j]
                an = math.hypot(*left[i])
                bn = math.hypot(*right[j])
                normalized = d / max(1.0, an, bn)
                distances.append(normalized)
                if maximum_displacement is None or normalized > \
                        maximum_displacement["normalized_displacement"]:
                    maximum_displacement = {
                        "group": dict(zip(group_fields, group_key)),
                        "baseline_root_index": left_root_indices[i],
                        "scheduled_root_index": right_root_indices[j],
                        "baseline_v": list(left[i]),
                        "scheduled_v": list(right[j]),
                        "absolute_displacement": d,
                        "normalized_displacement": normalized,
                    }

        for line_a, line_b in itertools.zip_longest(base, scheduled):
            if line_a is None or line_b is None:
                key_mismatches += 1
                break
            a = line_a.split()
            b = line_b.split()
            rows += 1
            if len(a) != len(base_header) or len(b) != len(candidate_header):
                key_mismatches += 1
                continue
            key_a = tuple(a[base_i[name]] for name in group_fields)
            key_b = tuple(b[base_i[name]] for name in group_fields)
            if key_a != key_b:
                key_mismatches += 1
            if group_key is not None and key_a != group_key:
                finish_group()
                left = []
                right = []
                left_root_indices = []
                right_root_indices = []
            group_key = key_a
            left.append((float(a[base_i["final_v_re"]]),
                         float(a[base_i["final_v_im"]])))
            right.append((float(b[base_i["final_v_re"]]),
                          float(b[base_i["final_v_im"]])))
            left_root_indices.append(a[base_i["root_index"]])
            right_root_indices.append(b[base_i["root_index"]])
        finish_group()
    return {
        "baseline_path": str(base_path),
        "scheduled_path": str(scheduled_path),
        "root_rows": rows,
        "root_sets": groups,
        "key_mismatches": key_mismatches,
        "incomplete_root_sets": incomplete_groups,
        "minimum_bijection_normalized_displacement": stats(distances),
        "maximum_displacement_record": maximum_displacement,
    }


def root_diagnostics(path: Path):
    rows = list(read_ws(path))
    roles = {"0": "other_warm_completeness", "1": "positive_real_candidate",
             "2": "complex_soft_cut"}
    by_role = {}
    for role_id, role_name in roles.items():
        subset = [row for row in rows if row.get("role") == role_id]
        by_role[role_name] = {
            "roots": len(subset),
            "d14real_updates": stats(numbers(subset, "d14real_updates")),
            "d14real_relative_newton_correction": stats(
                numbers(subset, "d14real_rel_correction")),
            "d14real_position_error": stats(
                numbers(subset, "d14real_position_error")),
            "qf_escalated_roots": sum(row.get("qf_escalated") == "1"
                                       for row in subset),
            "qf_source_match_valid": sum(
                row.get("qf_source_match_valid") == "1" for row in subset),
        }
    cluster_hist = collections.Counter(row.get("cluster_size", "") for row in rows)
    update_hist = collections.Counter(row.get("d14real_updates", "") for row in rows)
    return {
        "path": str(path),
        "root_records": len(rows),
        "role_counts": {roles[key]: sum(row.get("role") == key for row in rows)
                        for key in roles},
        "by_role": by_role,
        "d14real_updates": stats(numbers(rows, "d14real_updates")),
        "freeze_count_total": sum(int(row.get("freeze_count", "0"))
                                   for row in rows),
        "reactivation_count_total": sum(
            int(row.get("reactivation_count", "0")) for row in rows),
        "cluster_size_histogram": dict(sorted(cluster_hist.items())),
        "d14real_update_histogram": dict(sorted(update_hist.items(),
                                                  key=lambda item: int(item[0]))),
        "qf_escalated_roots": sum(row.get("qf_escalated") == "1" for row in rows),
        "qf_source_match_valid_roots": sum(
            row.get("qf_source_match_valid") == "1" for row in rows),
        "qf_source_match_invalid_roots": sum(
            row.get("qf_source_match_valid") == "0" for row in rows),
        "double_seed_valid_roots": sum(
            row.get("double_seed_valid") == "1" for row in rows),
        "expanded_seed_match_valid_roots": sum(
            row.get("expanded_seed_match_valid") == "1" for row in rows),
        "expanded_seed_match_invalid_roots": sum(
            row.get("double_seed_valid") == "1" and
            row.get("expanded_seed_match_valid") == "0" for row in rows),
        "qf_expanded_relative_residual": stats(
            numbers(rows, "qf_expanded_relative_residual")),
        "qf_newton_correction": stats(numbers(rows, "qf_newton")),
        "qf_displacement": stats(numbers(
            [row for row in rows if row.get("qf_displacement_valid") == "1"],
            "qf_displacement")),
    }


def stage_diagnostics(path: Path):
    rows = list(read_ws(path))
    out = {"path": str(path), "rows": len(rows), "lanes": {}}
    fields = ("whole_classify_ms", "radial_events_ms", "d14_solve_ms",
              "presearch_ms", "d14real_ms", "qf_polish_ms",
              "residual_eval_ms", "completeness_check_ms",
              "physical_classify_ms", "soft_event_ms")
    for lane in sorted({row.get("lane", "") for row in rows}):
        selected = [row for row in rows if row.get("lane") == lane]
        out["lanes"][lane] = {
            "rows": len(selected),
            "fields": {field: stats(numbers(selected, field)) for field in fields},
            "qf_warm_calls": sum(int(row.get("qf_warm_calls", "0"))
                                  for row in selected),
            "qf_cold_calls": sum(int(row.get("qf_cold_calls", "0"))
                                 for row in selected),
        }
    return out


def case92_diagnostics(root: Path):
    out = {}
    for variant in ("safe", "scheduled"):
        stage_path = root / f"case92_final_{variant}_stages.tsv"
        roots_path = root / f"case92_final_{variant}_roots.tsv"
        stage_rows = list(read_ws(stage_path))
        root_rows = list(read_ws(roots_path))
        out[variant] = {"stage_path": str(stage_path), "root_path": str(roots_path),
                        "lanes": {}}
        for lane in ("cold", "warm"):
            lane_stages = [row for row in stage_rows if row.get("lane") == lane]
            epochs = {}
            for epoch in sorted({row.get("epoch_index", "") for row in lane_stages},
                                key=lambda value: int(value)):
                selected = [row for row in lane_stages
                            if row.get("epoch_index") == epoch]
                epoch_summary = {field: stats(numbers(selected, field)) for field in (
                    "whole_classify_ms", "d14_solve_ms", "presearch_ms",
                    "d14real_ms", "qf_polish_ms", "residual_eval_ms",
                    "completeness_check_ms", "physical_classify_ms",
                    "soft_event_ms", "presearch_sweeps", "dd_sweeps",
                    "qf_warm_sweeps", "qf_cold_sweeps", "d14real_calls",
                    "d14real_finite_calls", "d14real_nonconverged",
                    "completeness_fails", "root_count_bad", "conjugacy_bad",
                    "vieta_bad")}
                epoch_roots = [row for row in root_rows
                               if row.get("lane") == lane and
                               row.get("epoch_index") == epoch]
                epoch_summary["root_records"] = len(epoch_roots)
                epoch_summary["expanded_root_match_valid"] = sum(
                    row.get("expanded_seed_match_valid") == "1"
                    for row in epoch_roots)
                epoch_summary["expanded_relative_residual"] = stats(
                    numbers(epoch_roots, "qf_expanded_relative_residual"))
                epoch_summary["qf_newton_correction"] = stats(
                    numbers(epoch_roots, "qf_newton"))
                epoch_summary["qf_position_error"] = stats(
                    numbers(epoch_roots, "qf_position_error"))
                epochs[epoch] = epoch_summary
            out[variant]["lanes"][lane] = epochs
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root
    names = ("baseline_safe", "scheduled_abs1e-10_rel1e-8",
             "scheduled_after_diag_elide")
    paths = {
        name: (root / f"{name}.tsv.gz") if (root / f"{name}.tsv.gz").exists()
        else root / f"{name}.tsv"
        for name in names
    }
    variants = {name: summarize_variant(paths[name]) for name in names}
    baseline = variants["baseline_safe"]
    comparisons = {}
    for name, candidate in variants.items():
        if name == "baseline_safe":
            continue
        comparisons[name] = {
            "whole_epoch": compare_variants(baseline, candidate),
            "paired_outputs": pairwise_output_check(
                paths["baseline_safe"], paths[name]),
        }
    profile_dir = root / "profiles"
    profiles = [parse_profile(path) for path in sorted(profile_dir.glob("*.log"))]
    result = {
        "schema": 1,
        "metadata": {
            "timing_scope": "whole trajectory lanes; 1,804 trajectories, 14,432 rows, 3 reps",
            "targets": [1e-3, 1e-4],
            "accuracy_pairing": "same input rows and same V2 call path; status and value compared pairwise",
            "schedule_enabled_only_by_env": True,
            "qf_residual_and_completeness_gates_changed": False,
        },
        "variants": variants,
        "comparisons_to_baseline": comparisons,
        "root_diagnostics": {
            "baseline_safe": root_diagnostics(
                root / "root_work_safe.tsv.gz" if
                (root / "root_work_safe.tsv.gz").exists() else
                root / "root_work_safe.tsv"),
            "scheduled_abs1e-10_rel1e-8": root_diagnostics(
                root / "root_work_scheduled.tsv.gz" if
                (root / "root_work_scheduled.tsv.gz").exists() else
                root / "root_work_scheduled.tsv"),
            "stages_safe": stage_diagnostics(
                root / "stages_safe.tsv.gz" if (root / "stages_safe.tsv.gz").exists()
                else root / "stages_safe.tsv"),
            "stages_scheduled": stage_diagnostics(
                root / "stages_scheduled.tsv.gz" if
                (root / "stages_scheduled.tsv.gz").exists() else
                root / "stages_scheduled.tsv"),
        },
        "root_set_parity": root_set_parity(
            root / "root_work_safe.tsv.gz" if
            (root / "root_work_safe.tsv.gz").exists() else
            root / "root_work_safe.tsv",
            root / "root_work_scheduled.tsv.gz" if
            (root / "root_work_scheduled.tsv.gz").exists() else
            root / "root_work_scheduled.tsv"),
        "case92_tail": case92_diagnostics(root),
        "profiles": profiles,
    }
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
