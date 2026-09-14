#!/usr/bin/env python3
"""Matched summary for the projective-fold 14,432-row final A/B."""
from __future__ import annotations

import argparse
import gzip
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def read_tsv(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        header = None
        rows = []
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            if header is None:
                header = fields
                continue
            if len(fields) != len(header):
                raise ValueError(f"{path}: got {len(fields)} fields, expected {len(header)}")
            rows.append(dict(zip(header, fields)))
    if header is None:
        raise ValueError(f"empty input: {path}")
    return rows


def percentile(xs, p):
    values=[float(x) for x in xs if math.isfinite(float(x))]
    if not values:
        return None
    v = sorted(values)
    x = (len(v) - 1) * p / 100.0
    lo = int(x)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] if lo == hi else v[lo] * (hi - x) + v[hi] * (x - lo)


def dist(xs):
    finite=[float(x) for x in xs if math.isfinite(float(x))]
    return {"n": len(xs), "finite_n":len(finite), "nonfinite_n":len(xs)-len(finite),
            "p50": percentile(finite, 50), "p90": percentile(finite, 90),
            "p95": percentile(finite, 95), "p99": percentile(finite, 99),
            "max": percentile(finite, 100)}


def as_float(row, key):
    x = float(row[key])
    return x


def as_int(row, key):
    return int(row[key])


def row_key(r):
    return tuple(r[k] for k in ("case_id", "configuration_id", "profile", "d_bin_index",
        "epoch_index", "trajectory_id", "trajectory_pos", "policy", "target"))


def merge_repetitions(files, expected_rows=14432):
    by_rep = []
    for path in files:
        rows = read_tsv(path)
        if len(rows) != expected_rows:
            raise ValueError(f"{path}: got {len(rows)} rows, expected {expected_rows}")
        m = {row_key(r): r for r in rows}
        if len(m) != expected_rows:
            raise ValueError(f"{path}: duplicate row keys")
        by_rep.append(m)
    keys = set(by_rep[0])
    if any(set(m) != keys for m in by_rep[1:]):
        raise ValueError("replicate row keys differ")
    merged = {}
    timing_cols = [f"{lane}_{part}" for lane in ("cold", "warm", "radial")
                   for part in ("ms", "topology_ms", "adaptive_ms")]
    value_cols = [f"{lane}_{part}" for lane in ("cold", "warm", "radial")
                  for part in ("mu", "value_error", "nodes", "evaluations", "panels", "splits")]
    grad_cols = [f"{lane}_g{j}{suffix}" for lane in ("cold", "warm", "radial")
                 for j in range(5) for suffix in ("", "_error")]
    for key in keys:
        reps = [m[key] for m in by_rep]
        base = dict(reps[0])
        for field in timing_cols + value_cols + grad_cols:
            vals = [as_float(r, field) for r in reps]
            base[field] = statistics.median(vals)
        for field in (f"{lane}_{part}" for lane in ("cold", "warm", "radial")
                      for part in ("value_converged", "value_stop", "stop", "status",
                                   "topology_status", "topology_cells", "topology_events",
                                   "topology_hash")):
            values = [r[field] for r in reps]
            if len(set(values)) != 1:
                raise ValueError(f"non-deterministic status/topology field {field} for {key}: {values}")
            base[field] = values[0]
        for field in (f"{lane}_g{j}_{suffix}" for lane in ("cold", "warm", "radial")
                      for j in range(5) for suffix in ("quality", "reason")):
            values = [r[field] for r in reps]
            if len(set(values)) != 1:
                raise ValueError(f"non-deterministic gradient quality {field} for {key}: {values}")
            base[field] = values[0]
        for field in ("warm_l1", "warm_l2", "warm_l3", "warm_rescreen_fail", "warm_seed_used"):
            vals = [as_int(r, field) for r in reps]
            if len(set(vals)) != 1:
                raise ValueError(f"non-deterministic warm route {field} for {key}: {vals}")
            base[field] = vals[0]
        merged[key] = base
    return merged, by_rep


def counter(rows, field):
    return dict(sorted(Counter(r[field] for r in rows).items()))


def subset(rows, target=None, profile=None, scope="all"):
    out = rows
    if target is not None:
        out = [r for r in out if abs(float(r["target"]) - target) < 1e-15]
    if profile is not None and profile != "all":
        out = [r for r in out if r["profile"] == profile]
    if scope == "steady":
        out = [r for r in out if int(r["trajectory_pos"]) > 0]
    elif scope == "first":
        out = [r for r in out if int(r["trajectory_pos"]) == 0]
    return out


def arm_quality(rows, policy, target):
    lanes = {}
    for lane in ("cold", "warm", "radial"):
        errors = []
        for r in rows:
            ref = as_float(r, "reference")
            errors.append(abs(as_float(r, f"{lane}_mu") - ref) / max(abs(ref), 1e-300))
        lanes[lane] = {
            "value_converged": sum(as_int(r, f"{lane}_value_converged") for r in rows),
            "value_stop_counts": counter(rows, f"{lane}_value_stop"),
            "stop_counts": counter(rows, f"{lane}_stop"),
            "status_counts": counter(rows, f"{lane}_status"),
            "relative_error_to_VBM_1e-6_input_reference": dist(errors),
            "reference_error_exceeds_requested_value_rtol": sum(e > target for e in errors),
            "median_nodes": percentile([as_float(r, f"{lane}_nodes") for r in rows], 50),
            "median_topology_ms": percentile([as_float(r, f"{lane}_topology_ms") for r in rows], 50),
            "median_adaptive_ms": percentile([as_float(r, f"{lane}_adaptive_ms") for r in rows], 50),
        }
    if policy == "ValueFirst":
        lanes["gradient_quality"] = {}
        for lane in ("cold", "warm", "radial"):
            lanes["gradient_quality"][lane] = {}
            for j in range(5):
                lanes["gradient_quality"][lane][str(j)] = {
                    "quality_counts": counter(rows, f"{lane}_g{j}_quality"),
                    "reason_counts": counter(rows, f"{lane}_g{j}_reason"),
                    "median_reported_error": percentile([as_float(r, f"{lane}_g{j}_error") for r in rows], 50),
                }
    return lanes


def quality_rank(s):
    return {"NotRequested": 0, "Invalid": 1, "FiniteUncertified": 2, "ToleranceMet": 3}.get(s, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    root = ap.parse_args().root
    variants = {}
    raw = {}
    for arm in ("baseline", "candidate"):
        variants[arm] = {}
        raw[arm] = {}
        for policy, filesuffix in (("None", "value"), ("ValueFirst", "jac")):
            files = [root / "raw" / f"{arm}_{filesuffix}_r{i}.tsv.gz" for i in range(3)]
            merged, replicates = merge_repetitions(files)
            variants[arm][policy] = merged
            raw[arm][policy] = replicates
    output = {
        "phase": "Phase 75 projective physical-fold final trajectory A/B",
        "rows_per_arm_policy": 14432,
        "physical_input_rows": 7216,
        "trajectories": 1804,
        "epochs_per_trajectory": 4,
        "repetitions": 3,
        "profiles": ["uniform", "linear"],
        "targets": [1e-3, 1e-4],
        "policies": {"None": "value-only", "ValueFirst": "value plus analytic 5Jac; grad_rtol=1e-3, default grad_atol"},
        "arms": {}, "comparisons": {}, "timing": {}, "checks": {}
    }
    total_topology_hash_mismatch = 0
    total_convergence_mismatch = 0
    total_status_mismatch = 0
    total_new_reference_rtol_violations = 0
    total_gradient_quality_regressions = 0
    for arm in ("baseline", "candidate"):
        output["arms"][arm] = {}
        for policy in ("None", "ValueFirst"):
            rows = list(variants[arm][policy].values())
            output["arms"][arm][policy] = {}
            for target in (1e-3, 1e-4):
                rs = subset(rows, target=target)
                output["arms"][arm][policy][f"RelTol={target:g}"] = {
                    "all_rows": arm_quality(rs, policy, target),
                    "steady_rows": arm_quality(subset(rs, scope="steady"), policy, target),
                    "first_rows": arm_quality(subset(rs, scope="first"), policy, target),
                }

    # Matched candidate-vs-baseline comparisons; preserve rows that differ.
    for policy in ("None", "ValueFirst"):
        bmap, cmap = variants["baseline"][policy], variants["candidate"][policy]
        if set(bmap) != set(cmap):
            raise ValueError(f"baseline/candidate row-key mismatch for {policy}")
        output["comparisons"][policy] = {}
        for target in (1e-3, 1e-4):
            rs = [bmap[k] for k in bmap if abs(float(k[-1]) - target) < 1e-15]
            comp = {"rows": len(rs), "by_lane": {}}
            for lane in ("cold", "warm", "radial"):
                mu_scaled, speed_ratios, dmu = [], [], []
                baseline_times,candidate_times=[],[]
                status_mismatch = convergence_mismatch = topology_hash_mismatch = 0
                count_mismatch = stop_mismatch = 0
                ref_worse = ref_better = ref_equal = ref_new_rtol_fail = 0
                error_gate_exceed = 0
                node_diffs = []
                for b in rs:
                    k = row_key(b);c = cmap[k]
                    base_mu=as_float(b,f"{lane}_mu");cand_mu=as_float(c,f"{lane}_mu")
                    ref=as_float(b,"reference");scale=max(abs(ref),1e-300)
                    err_b=abs(base_mu-ref);err_c=abs(cand_mu-ref)
                    mu_scaled.append(abs(cand_mu-base_mu)/scale);dmu.append(abs(cand_mu-base_mu))
                    bt=as_float(b,f"{lane}_ms");ct=as_float(c,f"{lane}_ms")
                    baseline_times.append(bt);candidate_times.append(ct)
                    speed_ratios.append(ct/max(bt,1e-300))
                    status_mismatch+=b[f"{lane}_status"]!=c[f"{lane}_status"]
                    convergence_mismatch+=b[f"{lane}_value_converged"]!=c[f"{lane}_value_converged"]
                    stop_mismatch+=b[f"{lane}_value_stop"]!=c[f"{lane}_value_stop"] or b[f"{lane}_stop"]!=c[f"{lane}_stop"]
                    topology_hash_mismatch+=b[f"{lane}_topology_hash"]!=c[f"{lane}_topology_hash"]
                    count_mismatch+=b[f"{lane}_topology_events"]!=c[f"{lane}_topology_events"] or b[f"{lane}_topology_cells"]!=c[f"{lane}_topology_cells"]
                    ref_worse+=err_c>err_b;ref_better+=err_c<err_b;ref_equal+=err_c==err_b
                    ref_new_rtol_fail+=(err_c/scale>target and err_b/scale<=target)
                    error_gate_exceed+=abs(cand_mu-base_mu)>(as_float(b,f"{lane}_value_error")+as_float(c,f"{lane}_value_error"))
                    node_diffs.append(as_float(c,f"{lane}_nodes")-as_float(b,f"{lane}_nodes"))
                comp["by_lane"][lane] = {
                    "candidate_over_baseline_time_ratio": dist(speed_ratios),
                    "candidate_speedup_from_ratio_of_medians": percentile(baseline_times,50)/max(percentile(candidate_times,50),1e-300),
                    "candidate_to_baseline_absolute_mu_difference": dist(dmu),
                    "candidate_to_baseline_mu_difference_scaled_by_reference": dist(mu_scaled),
                    "candidate_reference_error_worse_better_equal_rows": {
                        "worse":ref_worse,"better":ref_better,"equal":ref_equal},
                    "new_reference_rtol_exceedances":ref_new_rtol_fail,
                    "reported_error_sum_exceeded_by_mu_change":error_gate_exceed,
                    "status_mismatch":status_mismatch,
                    "value_convergence_mismatch":convergence_mismatch,
                    "value_stop_or_overall_stop_mismatch":stop_mismatch,
                    "topology_event_or_cell_count_mismatch":count_mismatch,
                    "topology_identity_hash_mismatch":topology_hash_mismatch,
                    "candidate_minus_baseline_node_count":dist(node_diffs),
                }
                for scope in ("all", "steady", "first"):
                    bscope=subset(rs,scope=scope)
                    ratios=[]
                    for b in bscope:
                        c=cmap[row_key(b)]
                        ratios.append(as_float(c,f"{lane}_ms")/max(as_float(b,f"{lane}_ms"),1e-300))
                    comp["by_lane"][lane][f"paired_time_ratio_{scope}"]=dist(ratios)
                total_topology_hash_mismatch+=topology_hash_mismatch
                total_convergence_mismatch+=convergence_mismatch
                total_status_mismatch+=status_mismatch
                total_new_reference_rtol_violations+=ref_new_rtol_fail
            if policy=="ValueFirst":
                grad={}
                for lane in ("cold","warm","radial"):
                    grad[lane]={}
                    for j in range(5):
                        qreg=0;reasons=Counter();qualities=Counter();scaled=[];gerr=[]
                        for b in rs:
                            c=cmap[row_key(b)]
                            qb=b[f"{lane}_g{j}_quality"];qc=c[f"{lane}_g{j}_quality"]
                            qreg+=quality_rank(qc)<quality_rank(qb)
                            qualities[(qb,qc)]+=1
                            reasons[(b[f"{lane}_g{j}_reason"],c[f"{lane}_g{j}_reason"])]+=1
                            gb=as_float(b,f"{lane}_g{j}");gc=as_float(c,f"{lane}_g{j}")
                            scaled.append(abs(gc-gb)/max(1.0,abs(gb),abs(gc)))
                            gerr.append(as_float(c,f"{lane}_g{j}_error"))
                        total_gradient_quality_regressions+=qreg
                        grad[lane][str(j)]={
                            "candidate_to_baseline_scaled_gradient_difference":dist(scaled),
                            "quality_transition_counts":{f"{a}->{b}":n for (a,b),n in sorted(qualities.items())},
                            "reason_transition_counts":{f"{a}->{b}":n for (a,b),n in sorted(reasons.items())},
                            "quality_regressions":qreg,
                            "candidate_median_reported_gradient_error":percentile(gerr,50),
                        }
                comp["gradient_comparison"]=grad
            output["comparisons"][policy][f"RelTol={target:g}"]=comp

    # Explicit latency distributions for cold/warm/radial, all rows and
    # trajectory steady state, split by profile, tolerance and policy.
    for policy in ("None","ValueFirst"):
        output["timing"][policy]={}
        for target in (1e-3,1e-4):
            output["timing"][policy][f"RelTol={target:g}"]={}
            for profile in ("all","uniform","linear"):
                output["timing"][policy][f"RelTol={target:g}"][profile]={}
                for scope in ("all","steady","first"):
                    key=f"{profile}/{scope}"
                    output["timing"][policy][f"RelTol={target:g}"][profile][scope]={}
                    for lane in ("cold","warm","radial"):
                        output["timing"][policy][f"RelTol={target:g}"][profile][scope][lane]={}
                        selected=subset(list(variants["baseline"][policy].values()),target,profile,scope)
                        for arm in ("baseline","candidate"):
                            rows_arm=subset(list(variants[arm][policy].values()),target,profile,scope)
                            output["timing"][policy][f"RelTol={target:g}"][profile][scope][lane][arm]=dist([as_float(r,f"{lane}_ms") for r in rows_arm])
                        base=variants["baseline"][policy];cand=variants["candidate"][policy]
                        ratios=[as_float(cand[row_key(b)],f"{lane}_ms")/max(as_float(b,f"{lane}_ms"),1e-300) for b in selected]
                        output["timing"][policy][f"RelTol={target:g}"][profile][scope][lane]["candidate_over_baseline_paired_ratio"]=dist(ratios)

    output["checks"]={
        "raw_rows_per_run_all_12_runs":14432,
        "matched_keys":all(set(variants[a][p])==set(variants["baseline"][p]) for a in ("baseline","candidate") for p in ("None","ValueFirst")),
        "topology_identity_mismatches_total":total_topology_hash_mismatch,
        "value_convergence_mismatches_total":total_convergence_mismatch,
        "numerical_status_mismatches_total":total_status_mismatch,
        "new_VBM_reference_rtol_exceedances_total":total_new_reference_rtol_violations,
        "analytic_gradient_quality_regressions_total":total_gradient_quality_regressions,
        "input_reference_note":"VBM RelTol=1e-6 values from the frozen 2026-09-13 benchmark snapshot; comparison is observed error, not a formal bound",
    }
    scan_path=root/"projective_scan.tsv"
    if scan_path.exists():
        scan=read_tsv(scan_path)
        accepted=[r for r in scan if r["decision"]=="accepted"]
        unique_accepted={(r["case_id"],r["configuration_id"],r["profile"],r["d_bin_index"],r["radius"]) for r in accepted}
        output["projective_scan"]={
            "physical_input_rows":7216,
            "chart_p4_records":len(scan),
            "accepted_projective_fold_records":len(accepted),
            "accepted_case_radius_locations":len(unique_accepted),
            "decision_counts":dict(sorted(Counter(r["decision"] for r in scan).items())),
        }
    print(json.dumps(output,indent=2,sort_keys=True,allow_nan=False))


if __name__=="__main__":
    main()
