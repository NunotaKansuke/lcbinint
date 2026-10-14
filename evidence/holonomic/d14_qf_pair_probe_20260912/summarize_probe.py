#!/usr/bin/env python3
"""Summarize the qf-warm pair one-step probe A/B evidence."""

from __future__ import annotations

import gzip
import json
import math
import statistics
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "d14_qf_warm_rootwise_20260912"
CASES = (149, 0, 64, 9)
TRACE_KEYS = ("case", "config", "profile", "d_bin", "epoch", "rep", "lane")
ROW_KEYS = ("case_id", "configuration_id", "profile", "d_bin_index",
            "epoch_index", "rep", "lane")
EPOCH_KEYS = ("case_id", "d_bin_index", "epoch_index", "target")
LANES = ("cold", "warm", "radial")


def rows(path: Path) -> list[dict[str, str]]:
    lines = [line.split() for line in path.read_text().splitlines()
             if line.strip() and not line.startswith("#")]
    if not lines:
        return []
    header = lines[0]
    return [dict(zip(header, line)) for line in lines[1:]]


def trace_rows(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    out: list[dict[str, str]] = []
    context: dict[str, str] = {}
    with opener(path, "rt") as stream:
        for line in stream:
            parts = line.rstrip("\n").split("\t")
            fields = dict(part.split("=", 1) for part in parts[1:] if "=" in part)
            if parts[0] == "D14TRACE_ROW":
                context = fields
            elif parts[0] == "D14QF_PAIRPOLISH":
                out.append({**context, **fields})
    return out


def percentile(values: list[float], q: float) -> float:
    xs = sorted(values)
    if not xs:
        return math.nan
    position = (len(xs) - 1) * q
    lo = int(position)
    hi = min(lo + 1, len(xs) - 1)
    frac = position - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def write_tsv(path: Path, data: list[dict]) -> None:
    if not data:
        return
    fields = list(data[0])
    with path.open("w") as stream:
        stream.write("\t".join(fields) + "\n")
        for item in data:
            stream.write("\t".join(str(item.get(field, "")) for field in fields)
                         + "\n")


def assignment_match(old: list[dict[str, str]],
                     new: list[dict[str, str]]) -> dict:
    if len(old) != 14 or len(new) != 14:
        return {"count_match": False, "max_abs_root_delta": None,
                "max_scaled_root_delta": None, "role_mismatches": -1}
    old = sorted(old, key=lambda r: int(r["root_index"]))
    new = sorted(new, key=lambda r: int(r["root_index"]))
    z0 = [(float(r["final_v_re"]), float(r["final_v_im"])) for r in old]
    z1 = [(float(r["final_v_re"]), float(r["final_v_im"])) for r in new]
    cost: list[list[float]] = []
    dist: list[list[float]] = []
    for ar, ai in z0:
        rowc, rowd = [], []
        for br, bi in z1:
            d = math.hypot(ar - br, ai - bi)
            rowd.append(d)
            rowc.append(d / max(1.0, math.hypot(ar, ai)))
        cost.append(rowc)
        dist.append(rowd)
    dp: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for i in range(14):
        nxt: dict[int, tuple[float, tuple[int, ...]]] = {}
        for mask, (total, path) in dp.items():
            for j in range(14):
                if mask & (1 << j):
                    continue
                key = mask | (1 << j)
                trial = (total + cost[i][j], path + (j,))
                if key not in nxt or trial[0] < nxt[key][0]:
                    nxt[key] = trial
        dp = nxt
    match = dp[(1 << 14) - 1][1]
    deltas = [dist[i][match[i]] for i in range(14)]
    roles = sum(old[i]["role"] != new[match[i]]["role"] for i in range(14))
    return {
        "count_match": True,
        "max_abs_root_delta": max(deltas),
        "max_scaled_root_delta": max(
            deltas[i] / max(1.0, math.hypot(*z0[i])) for i in range(14)),
        "role_mismatches": roles,
    }


def context_key(r: dict[str, str]) -> tuple[str, ...]:
    return (r["case"], r["d_bin"], r["epoch"], r["lane"])


def root_key(r: dict[str, str]) -> tuple[str, ...]:
    return (r["case_id"], r["configuration_id"], r["profile"],
            r["d_bin_index"], r["epoch_index"], r["rep"], r["lane"])


def attempt_reason(r: dict[str, str]) -> str:
    if r["probe_viable"] == "1":
        return "passed"
    if r["probe_available"] != "1":
        return "no-finite-first-step"
    if int(r["probe_backtracks"]) < 0:
        if r["probe_finite"] != "1":
            return "no-finite-descending-step-within-3-backtracks"
        return "no-residual-decrease-within-3-backtracks"
    if r["probe_finite"] != "1":
        return "accepted-trial-not-finite"
    if float(r["probe_accepted_ratio"]) > 0.5:
        return "residual-not-halved"
    if r["probe_driver_stable"] == "1":
        return "pair-external-correction-above-1e-20"
    return "post-probe-global-correction-above-1e-20"


def pair_attempts() -> tuple[list[dict], dict]:
    old_attempts = rows(OLD / "pair_polish_attempts.tsv")
    old_by_key = {(r["case"], r["d_bin"], r["epoch"],
                   r["lane"]): r for r in old_attempts}
    probes: list[dict[str, str]] = []
    for case in CASES:
        probes.extend(trace_rows(ROOT / f"case{case}_probe_trace.log.gz"))
    probe_by_key = {context_key(r): r for r in probes}
    if set(probe_by_key) != set(old_by_key):
        raise RuntimeError("v2/probe D14 pair attempt key mismatch")
    merged = []
    for key in sorted(probe_by_key):
        p, o = probe_by_key[key], old_by_key[key]
        if p["pair"] != o["pair"]:
            raise RuntimeError(f"driver pair changed before probe at {key}")
        merged.append({
            "case_id": p["case"], "configuration_id": p["config"],
            "profile": p["profile"], "d_bin_index": p["d_bin"],
            "epoch_index": p["epoch"], "rep": p["rep"], "lane": p["lane"],
            "pair": p["pair"], "v2_accepted": o["accepted"],
            "probe_accepted": p["accepted"], "probe_viable": p["probe_viable"],
            "probe_finite": p["probe_finite"],
            "probe_reason": attempt_reason(p),
            "probe_full_step_ratio": p["probe_full_ratio"],
            "probe_accepted_ratio": p["probe_accepted_ratio"],
            "probe_backtracks": p["probe_backtracks"],
            "probe_step_over_separation": p["probe_step_over_sep"],
            "probe_det_fraction": p["probe_det_fraction"],
            "driver_pair_stable": p["probe_driver_stable"],
            "post_driver_pair": p["probe_driver"],
            "post_driver_max_step": p["probe_post_driver_step"],
            "post_pair_external_max_step": p["probe_post_outer_step"],
            "v2_pair_iterations": o["pair_iters"],
            "probe_pair_iterations": p["pair_iters"],
            "v2_pair_backtracks": o["backtracks"],
            "probe_pair_backtracks": p["backtracks"],
            "v2_verify_sweeps": o["verify_iters"],
            "probe_verify_sweeps": p["verify_iters"],
            "probe_verify_converged": p["verify_converged"],
            "probe_scalar_certificate": p["scalar_certificate"],
        })
    summary = {
        "attempts": len(merged),
        "v2_accepted": sum(r["v2_accepted"] == "1" for r in merged),
        "probe_viable": sum(r["probe_viable"] == "1" for r in merged),
        "probe_accepted": sum(r["probe_accepted"] == "1" for r in merged),
        "finite_accepted_probe_steps": sum(r["probe_finite"] == "1" and
                                            int(r["probe_backtracks"]) >= 0
                                            for r in merged),
        "accepted_overlap": sum(r["v2_accepted"] == r["probe_accepted"] == "1"
                                 for r in merged),
        "false_positive_probes": sum(r["v2_accepted"] != "1" and
                                      r["probe_viable"] == "1" for r in merged),
        "false_negative_probes": sum(r["v2_accepted"] == "1" and
                                      r["probe_viable"] != "1" for r in merged),
        "v2_pair_iterations": sum(int(r["v2_pair_iterations"]) for r in merged),
        "probe_pair_iterations": sum(int(r["probe_pair_iterations"]) for r in merged),
        "v2_backtracks": sum(int(r["v2_pair_backtracks"]) for r in merged),
        "probe_backtracks": sum(int(r["probe_pair_backtracks"]) for r in merged),
        "v2_global_verifier_sweeps": sum(int(r["v2_verify_sweeps"]) for r in merged),
        "probe_global_verifier_sweeps": sum(int(r["probe_verify_sweeps"]) for r in merged),
        "reject_reasons": dict(Counter(r["probe_reason"] for r in merged)),
        "gate": {
            "max_first_probe_backtracks": 3,
            "accepted_pair_residual_ratio_at_most": 0.5,
            "outer_step_threshold": 1e-20,
            "driver_stable_rule": "if pair remains the driver, all 12 outside roots must have max correction <= 1e-20; if the driver pair changes, the full post-probe maximum correction must be <= 1e-20",
            "certificate_note": "A viable probe is only permission to continue or perform the unchanged global verifier; it is never a root-set certificate.",
        },
    }
    return merged, summary


def rootwork() -> tuple[list[dict], list[dict], dict]:
    out: list[dict] = []
    avoided: list[dict] = []
    root_checks: list[dict] = []
    for case in CASES:
        v2_t = {root_key(r): r for r in rows(OLD / f"case{case}_pair_v2_timing.tsv")}
        v3_t = {root_key(r): r for r in rows(ROOT / f"case{case}_probe_timing.tsv")}
        base_t = {root_key(r): r for r in rows(OLD / f"case{case}_timing.tsv")}
        base_roots: dict[tuple[str, ...], list[dict[str, str]]] = {}
        v3_roots: dict[tuple[str, ...], list[dict[str, str]]] = {}
        for r in rows(OLD / f"case{case}_roots.tsv"):
            base_roots.setdefault(root_key(r), []).append(r)
        for r in rows(ROOT / f"case{case}_probe_roots.tsv"):
            v3_roots.setdefault(root_key(r), []).append(r)
        if set(v2_t) != set(v3_t) or set(base_t) != set(v3_t):
            raise RuntimeError(f"D14 row mismatch for case {case}")
        for key in sorted(v3_t):
            v2, v3, base = v2_t[key], v3_t[key], base_t[key]
            same = all(v2[k] == v3[k] for k in
                       ("topology_status", "cells", "events", "completeness_fails",
                        "root_count_bad", "conjugacy_bad", "vieta_bad", "qf_root_count"))
            out.append({
                **dict(zip(ROW_KEYS, key)),
                "topology_certificate_equal": same,
                "v2_qf_warm_sweeps": v2["qf_warm_sweeps"],
                "probe_qf_warm_sweeps": v3["qf_warm_sweeps"],
                "v2_qf_cold_calls": v2["qf_cold_calls"],
                "probe_qf_cold_calls": v3["qf_cold_calls"],
                "v2_qf_cold_sweeps": v2["qf_cold_sweeps"],
                "probe_qf_cold_sweeps": v3["qf_cold_sweeps"],
                "v2_whole_classify_ms": v2["whole_classify_ms"],
                "probe_whole_classify_ms": v3["whole_classify_ms"],
                "v2_qf_polish_ms": v2["qf_polish_ms"],
                "probe_qf_polish_ms": v3["qf_polish_ms"],
                "v2_completeness_fails": v2["completeness_fails"],
                "probe_completeness_fails": v3["completeness_fails"],
                "v2_root_count_bad": v2["root_count_bad"],
                "probe_root_count_bad": v3["root_count_bad"],
                "v2_conjugacy_bad": v2["conjugacy_bad"],
                "probe_conjugacy_bad": v3["conjugacy_bad"],
                "v2_vieta_bad": v2["vieta_bad"],
                "probe_vieta_bad": v3["vieta_bad"],
            })
            if int(base["qf_cold_calls"]) > int(v3["qf_cold_calls"]):
                avoided.append(dict(zip(ROW_KEYS, key)))
                check = assignment_match(base_roots[key], v3_roots[key])
                root_checks.append({
                    **dict(zip(ROW_KEYS, key)), **check,
                    "base_completeness_fails": base["completeness_fails"],
                    "probe_completeness_fails": v3["completeness_fails"],
                    "base_root_count_bad": base["root_count_bad"],
                    "probe_root_count_bad": v3["root_count_bad"],
                    "base_conjugacy_bad": base["conjugacy_bad"],
                    "probe_conjugacy_bad": v3["conjugacy_bad"],
                    "base_vieta_bad": base["vieta_bad"],
                    "probe_vieta_bad": v3["vieta_bad"],
                })
    summary = {
        "rows": len(out),
        "topology_certificate_mismatches": sum(not r["topology_certificate_equal"]
                                                for r in out),
        "no_pair_qf_cold_calls": sum(
            int(r["qf_cold_calls"])
            for case in CASES
            for r in rows(OLD / f"case{case}_timing.tsv")),
        "v2_qf_cold_calls": sum(int(r["v2_qf_cold_calls"]) for r in out),
        "probe_qf_cold_calls": sum(int(r["probe_qf_cold_calls"]) for r in out),
        "qf_cold_avoided_against_no_pair": avoided,
        "accepted_rootset_checks": root_checks,
    }
    return out, root_checks, summary


def whole_epoch() -> tuple[list[dict], dict]:
    names = ("v2_a", "v2_b", "v3_a", "v3_b")
    loaded = {name: {tuple(r[k] for k in EPOCH_KEYS): r
                     for r in rows(ROOT / f"whole_{name}.tsv")}
              for name in names}
    keys = set(loaded["v2_a"])
    if len(keys) != 16 or any(set(loaded[n]) != keys for n in names):
        raise RuntimeError("whole-epoch rows are not matched")
    result_rows: list[dict] = []
    lane_summaries: dict[str, dict] = {}
    parity: dict[str, dict] = {}
    status_counts: dict[str, dict] = {}
    checked_tail = ("topology_status", "topology_cells", "topology_events")
    quantiles = (0.50, 0.90, 0.95, 0.99, 1.0)
    for lane in LANES:
        before: list[float] = []
        after: list[float] = []
        mismatches = 0
        max_mu = 0.0
        max_error = 0.0
        old_status: Counter[str] = Counter()
        new_status: Counter[str] = Counter()
        old_conv = new_conv = 0
        for key in sorted(keys):
            old_runs = [loaded[n][key] for n in ("v2_a", "v2_b")]
            new_runs = [loaded[n][key] for n in ("v3_a", "v3_b")]
            old, new = old_runs[0], new_runs[0]
            old_ms = statistics.median(float(r[f"{lane}_ms"]) for r in old_runs)
            new_ms = statistics.median(float(r[f"{lane}_ms"]) for r in new_runs)
            before.append(old_ms)
            after.append(new_ms)
            check = checked_tail + tuple(f"{lane}_{f}" for f in
                ("value_converged", "stop", "status", "nodes", "evaluations",
                 "panels", "splits"))
            mismatches += sum(old[f] != new[f] for f in check)
            mu0 = float(old[f"{lane}_mu"])
            mu1 = float(new[f"{lane}_mu"])
            scaled = abs(mu1 - mu0) / max(1.0, abs(mu0))
            max_mu = max(max_mu, scaled)
            max_error = max(max_error, abs(float(new[f"{lane}_error"]) -
                                           float(old[f"{lane}_error"])))
            old_status[old[f"{lane}_status"]] += 1
            new_status[new[f"{lane}_status"]] += 1
            old_conv += old[f"{lane}_value_converged"] == "1"
            new_conv += new[f"{lane}_value_converged"] == "1"
            result_rows.append({
                **dict(zip(EPOCH_KEYS, key)), "lane": lane,
                "v2_ms": old_ms, "probe_ms": new_ms,
                "delta_ms": new_ms - old_ms,
                "v2_mu": mu0, "probe_mu": mu1, "scaled_mu_delta": scaled,
                "v2_value_error": old[f"{lane}_error"],
                "probe_value_error": new[f"{lane}_error"],
                "v2_value_converged": old[f"{lane}_value_converged"],
                "probe_value_converged": new[f"{lane}_value_converged"],
                "v2_status": old[f"{lane}_status"],
                "probe_status": new[f"{lane}_status"],
                "v2_nodes": old[f"{lane}_nodes"],
                "probe_nodes": new[f"{lane}_nodes"],
                "v2_evaluations": old[f"{lane}_evaluations"],
                "probe_evaluations": new[f"{lane}_evaluations"],
                "v2_panels": old[f"{lane}_panels"],
                "probe_panels": new[f"{lane}_panels"],
                "v2_splits": old[f"{lane}_splits"],
                "probe_splits": new[f"{lane}_splits"],
            })
        lane_summaries[lane] = {
            "rows": len(before),
            "v2_ms_percentiles": {str(q): percentile(before, q) for q in quantiles},
            "probe_ms_percentiles": {str(q): percentile(after, q) for q in quantiles},
            "v2_sum_ms": sum(before), "probe_sum_ms": sum(after),
            "sum_delta_ms": sum(after) - sum(before),
        }
        parity[lane] = {"checked_field_mismatches": mismatches,
                        "max_scaled_mu_delta": max_mu,
                        "max_abs_value_error_delta": max_error}
        status_counts[lane] = {"v2": dict(old_status), "probe": dict(new_status),
                               "v2_value_converged": old_conv,
                               "probe_value_converged": new_conv}
    return result_rows, {
        "lanes": lane_summaries, "parity": parity,
        "status_and_convergence": status_counts,
        "input_trajectories": 2, "epochs_per_trajectory": 4,
        "targets": [1e-3, 1e-4], "rows_per_lane": 16,
        "repeat_policy": "Each process reports the median of 3 repeats per row; two process medians are combined by median.",
        "scope_note": "Tail-selected value-only subset; full cold/warm include D14 and topology, radial-only reuses topology. Not a population-wide speed claim.",
    }


def main() -> None:
    attempts, attempt_summary = pair_attempts()
    rootwork_rows, accepted_roots, rootwork_summary = rootwork()
    whole_rows, whole_summary = whole_epoch()
    write_tsv(ROOT / "pair_probe_attempts.tsv", attempts)
    write_tsv(ROOT / "pair_probe_rootwork_ab.tsv", rootwork_rows)
    write_tsv(ROOT / "pair_probe_accepted_root_sets.tsv", accepted_roots)
    write_tsv(ROOT / "pair_probe_whole_epoch_ab.tsv", whole_rows)
    summary = {
        "schema": 1,
        "base_commit": "64a291a7cccc2d0793cb8ddbf7e0ba86a3bb3092",
        "candidate": "research-only one-step pair viability probe; production route unchanged",
        "pair_probe": attempt_summary,
        "rootwork": rootwork_summary,
        "whole_epoch": whole_summary,
        "precision_note": "All D14 pair equations and certificates use __float128, with the incumbent residual/completeness/topology gates unchanged.",
    }
    (ROOT / "pair_probe_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print("attempts", attempt_summary["attempts"],
          "viable", attempt_summary["probe_viable"],
          "accepted", attempt_summary["probe_accepted"],
          "whole rows/lane", whole_summary["rows_per_lane"])


if __name__ == "__main__":
    main()
