#!/usr/bin/env python3
"""Summarize qf-warm pair-polish diagnostics and matched epoch A/B runs."""

from __future__ import annotations

import csv
import gzip
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CASES = (149, 0, 64, 9)
ROW_KEYS = ("case_id", "configuration_id", "profile", "d_bin_index",
            "epoch_index", "rep", "lane")
EPOCH_KEYS = ("case_id", "d_bin_index", "epoch_index", "target")


def read_tsv(path: Path, delimiter: str = " ") -> list[dict[str, str]]:
    with path.open() as stream:
        lines = [line for line in stream
                 if line.strip() and not line.startswith("#")]
    return list(csv.DictReader(lines, delimiter=delimiter, skipinitialspace=True))


def open_text(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open()


def keyed_rows(path: Path, keys: tuple[str, ...], delimiter: str = " ") -> dict:
    return {tuple(row[k] for k in keys): row for row in read_tsv(path, delimiter)}


def trace_fields(line: str) -> tuple[str, dict[str, str]]:
    parts = line.rstrip("\n").split("\t")
    out: dict[str, str] = {}
    for part in parts[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            out[key] = value
    return parts[0], out


def pair_attempts() -> list[dict[str, str]]:
    attempts: list[dict[str, str]] = []
    for case in CASES:
        path = ROOT / f"case{case}_pair_v2_trace.log.gz"
        if not path.exists():
            path = ROOT / f"case{case}_pair_v2_trace.log"
        row: dict[str, str] = {}
        with open_text(path) as stream:
            lines = list(stream)
        for line in lines:
            if line.startswith("D14TRACE_ROW\t"):
                _, row = trace_fields(line)
            elif line.startswith("D14QF_PAIRPOLISH\t"):
                _, fields = trace_fields(line)
                attempts.append({**row, **fields})
    return attempts


def rootset_match(base: list[dict[str, str]],
                  candidate: list[dict[str, str]]) -> dict[str, float | int | bool]:
    if len(base) != 14 or len(candidate) != 14:
        return {"count_match": False, "max_abs_root_delta": math.nan,
                "max_scaled_root_delta": math.nan, "role_mismatches": -1}
    b = sorted(base, key=lambda r: int(r["root_index"]))
    c = sorted(candidate, key=lambda r: int(r["root_index"]))
    bz = [(float(r["final_v_re"]), float(r["final_v_im"])) for r in b]
    cz = [(float(r["final_v_re"]), float(r["final_v_im"])) for r in c]
    costs: list[list[float]] = []
    distances: list[list[float]] = []
    for ar, ai in bz:
        row_cost: list[float] = []
        row_dist: list[float] = []
        scale = max(1.0, math.hypot(ar, ai))
        for br, bi in cz:
            dist = math.hypot(ar - br, ai - bi)
            row_dist.append(dist)
            row_cost.append(dist / scale)
        costs.append(row_cost)
        distances.append(row_dist)

    # Small exact assignment DP; D14 has only fourteen roots.
    dp: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for i in range(14):
        nxt: dict[int, tuple[float, tuple[int, ...]]] = {}
        for mask, (total, assignment) in dp.items():
            for j in range(14):
                if mask & (1 << j):
                    continue
                new_mask = mask | (1 << j)
                trial = (total + costs[i][j], assignment + (j,))
                previous = nxt.get(new_mask)
                if previous is None or trial[0] < previous[0]:
                    nxt[new_mask] = trial
        dp = nxt
    assignment = dp[(1 << 14) - 1][1]
    matched_distances = [distances[i][j] for i, j in enumerate(assignment)]
    role_mismatches = sum(b[i]["role"] != c[j]["role"]
                          for i, j in enumerate(assignment))
    return {
        "count_match": True,
        "max_abs_root_delta": max(matched_distances),
        "max_scaled_root_delta": max(
            matched_distances[i] / max(1.0, math.hypot(*bz[i]))
            for i in range(14)),
        "role_mismatches": role_mismatches,
    }


def rootwork_ab() -> tuple[list[dict], list[dict]]:
    rows_out: list[dict] = []
    accepted_root_checks: list[dict] = []
    for case in CASES:
        base_timing = keyed_rows(ROOT / f"case{case}_timing.tsv", ROW_KEYS)
        cand_timing = keyed_rows(ROOT / f"case{case}_pair_v2_timing.tsv", ROW_KEYS)
        base_roots = defaultdict(list)
        cand_roots = defaultdict(list)
        for root in read_tsv(ROOT / f"case{case}_roots.tsv"):
            base_roots[tuple(root[k] for k in ROW_KEYS)].append(root)
        for root in read_tsv(ROOT / f"case{case}_pair_v2_roots.tsv"):
            cand_roots[tuple(root[k] for k in ROW_KEYS)].append(root)
        if base_timing.keys() != cand_timing.keys():
            raise RuntimeError(f"rootwork row mismatch for case {case}")
        for key in sorted(base_timing):
            old, new = base_timing[key], cand_timing[key]
            item = {k: old[k] for k in ROW_KEYS}
            item.update({
                "base_whole_classify_ms": float(old["whole_classify_ms"]),
                "candidate_whole_classify_ms": float(new["whole_classify_ms"]),
                "base_qf_warm_calls": int(old["qf_warm_calls"]),
                "candidate_qf_warm_calls": int(new["qf_warm_calls"]),
                "base_qf_warm_sweeps": int(old["qf_warm_sweeps"]),
                "candidate_qf_warm_sweeps": int(new["qf_warm_sweeps"]),
                "base_qf_cold_calls": int(old["qf_cold_calls"]),
                "candidate_qf_cold_calls": int(new["qf_cold_calls"]),
                "base_qf_cold_sweeps": int(old["qf_cold_sweeps"]),
                "candidate_qf_cold_sweeps": int(new["qf_cold_sweeps"]),
                "topology_status_equal": old["topology_status"] == new["topology_status"],
                "cells_equal": old["cells"] == new["cells"],
                "events_equal": old["events"] == new["events"],
            })
            rows_out.append(item)
            if int(old["qf_cold_calls"]) != int(new["qf_cold_calls"]):
                roots = rootset_match(base_roots[key], cand_roots[key])
                accepted_root_checks.append({
                    **{k: item[k] for k in ROW_KEYS},
                    **roots,
                    "base_completeness_fails": int(old["completeness_fails"]),
                    "candidate_completeness_fails": int(new["completeness_fails"]),
                    "base_root_count_bad": int(old["root_count_bad"]),
                    "candidate_root_count_bad": int(new["root_count_bad"]),
                    "base_conjugacy_bad": int(old["conjugacy_bad"]),
                    "candidate_conjugacy_bad": int(new["conjugacy_bad"]),
                    "base_vieta_bad": int(old["vieta_bad"]),
                    "candidate_vieta_bad": int(new["vieta_bad"]),
                })
    return rows_out, accepted_root_checks


def percentile(values: list[float], quantile: float) -> float:
    xs = sorted(values)
    if not xs:
        return math.nan
    position = (len(xs) - 1) * quantile
    lo = int(position)
    hi = min(lo + 1, len(xs) - 1)
    weight = position - lo
    return xs[lo] * (1.0 - weight) + xs[hi] * weight


def merge_whole_epoch_runs() -> tuple[dict, list[dict]]:
    run_paths = {
        "base": [ROOT / "pair_whole_epoch_base.tsv",
                 ROOT / "pair_whole_epoch_base_repeat.tsv"],
        "candidate": [ROOT / "pair_whole_epoch_candidate_v2.tsv",
                      ROOT / "pair_whole_epoch_candidate_v2_repeat.tsv"],
    }
    loaded = {name: [keyed_rows(path, EPOCH_KEYS) for path in paths]
              for name, paths in run_paths.items()}
    keyset = set(loaded["base"][0])
    if len(keyset) != 16 or any(set(rows) != keyset
                                for runs in loaded.values() for rows in runs):
        raise RuntimeError("whole-epoch A/B row mismatch")

    merged: list[dict] = []
    lane_summary: dict[str, dict] = {}
    parity: dict[str, dict] = {}
    for lane in ("cold", "warm", "radial"):
        before: list[float] = []
        after: list[float] = []
        max_mu_delta = 0.0
        max_error_delta = 0.0
        field_mismatches = 0
        for key in sorted(keyset):
            old_runs = [rows[key] for rows in loaded["base"]]
            new_runs = [rows[key] for rows in loaded["candidate"]]
            old_time = statistics.median(float(r[f"{lane}_ms"]) for r in old_runs)
            new_time = statistics.median(float(r[f"{lane}_ms"]) for r in new_runs)
            old, new = old_runs[0], new_runs[0]
            before.append(old_time)
            after.append(new_time)
            mu0 = float(old[f"{lane}_mu"])
            mu1 = float(new[f"{lane}_mu"])
            max_mu_delta = max(max_mu_delta, abs(mu1 - mu0) / max(1.0, abs(mu0)))
            max_error_delta = max(max_error_delta,
                                  abs(float(new[f"{lane}_error"]) -
                                      float(old[f"{lane}_error"])))
            checked = ("topology_status", "topology_cells", "topology_events",
                       f"{lane}_value_converged", f"{lane}_stop", f"{lane}_status",
                       f"{lane}_nodes", f"{lane}_evaluations", f"{lane}_panels",
                       f"{lane}_splits")
            field_mismatches += sum(old[f] != new[f] for f in checked)
            merged.append({
                **dict(zip(EPOCH_KEYS, key)),
                "lane": lane,
                "base_ms": old_time,
                "candidate_ms": new_time,
                "delta_ms": new_time - old_time,
                "base_mu": mu0,
                "candidate_mu": mu1,
                "scaled_mu_delta": abs(mu1 - mu0) / max(1.0, abs(mu0)),
                "base_value_converged": old[f"{lane}_value_converged"],
                "candidate_value_converged": new[f"{lane}_value_converged"],
                "base_status": old[f"{lane}_status"],
                "candidate_status": new[f"{lane}_status"],
            })
        qs = (0.50, 0.90, 0.95, 0.99, 1.0)
        lane_summary[lane] = {
            "rows": len(before),
            "base_ms_percentiles": {str(q): percentile(before, q) for q in qs},
            "candidate_ms_percentiles": {str(q): percentile(after, q) for q in qs},
            "base_sum_ms": sum(before),
            "candidate_sum_ms": sum(after),
            "sum_delta_ms": sum(after) - sum(before),
        }
        parity[lane] = {
            "checked_field_mismatches": field_mismatches,
            "max_scaled_mu_delta": max_mu_delta,
            "max_abs_estimated_error_delta": max_error_delta,
        }
    return {"lanes": lane_summary, "parity": parity,
            "input_trajectories": 2, "epochs_per_trajectory": 4,
            "targets": [1e-3, 1e-4], "rows_per_lane": 16,
            "timing_note": "Each process used 3 repeats/row; summary takes median of the two process medians. This is a deliberately tail-targeted 8-epoch subset, not a population benchmark."}, merged


def write_tsv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]),
                                delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    attempts = pair_attempts()
    rootwork, accepted_roots = rootwork_ab()
    whole, whole_rows = merge_whole_epoch_runs()
    pair_summary = {
        "schema": 1,
        "attempts": len(attempts),
        "accepted": sum(a.get("accepted") == "1" for a in attempts),
        "local_pair_converged": sum(a.get("pair_converged") == "1" for a in attempts),
        "global_verification_sweeps": sum(int(a.get("verify_iters", "0"))
                                           for a in attempts),
        "qf_cold_avoided_rows": [
            {k: row[k] for k in ROW_KEYS} for row in rootwork
            if row["base_qf_cold_calls"] > row["candidate_qf_cold_calls"]
        ],
        "accepted_root_set_checks": accepted_roots,
        "whole_epoch": whole,
        "scope": "compile-time research candidate; production route unchanged",
    }
    (ROOT / "pair_polish_summary.json").write_text(
        json.dumps(pair_summary, indent=2, allow_nan=False) + "\n")
    write_tsv(ROOT / "pair_polish_attempts.tsv", attempts)
    write_tsv(ROOT / "pair_polish_rootwork_ab.tsv", rootwork)
    write_tsv(ROOT / "pair_polish_accepted_root_sets.tsv", accepted_roots)
    write_tsv(ROOT / "pair_whole_epoch_ab.tsv", whole_rows)
    print(f"pair attempts={len(attempts)} accepted={pair_summary['accepted']}; "
          f"qf-cold avoided rows={len(pair_summary['qf_cold_avoided_rows'])}; "
          f"whole-epoch rows/lane={whole['rows_per_lane']}")


if __name__ == "__main__":
    main()
