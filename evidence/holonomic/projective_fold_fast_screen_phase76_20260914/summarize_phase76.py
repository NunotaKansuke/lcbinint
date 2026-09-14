#!/usr/bin/env python3
"""Summarize the Phase 76 three-arm whole and true-fold trajectory A/B."""
from __future__ import annotations

import argparse
import gzip
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ARMS = ("baseline", "qf", "screen")
POLICIES = ("None", "ValueFirst")
LANES = {"full-cold": "cold", "full-warm": "warm", "radial-only": "radial"}
FD_A = -2.4046891642370833


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
                raise ValueError(f"{path}: {len(fields)} fields, expected {len(header)}")
            rows.append(dict(zip(header, fields)))
    if header is None:
        raise ValueError(f"empty input: {path}")
    return rows


def quantile(xs, p):
    v = sorted(float(x) for x in xs if math.isfinite(float(x)))
    if not v:
        return None
    x = (len(v) - 1) * p / 100.0
    lo = int(x)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] if hi == lo else v[lo] * (hi - x) + v[hi] * (x - lo)


def dist(xs):
    xs = list(xs)
    finite = [float(x) for x in xs if math.isfinite(float(x))]
    return {"n": len(xs), "finite_n": len(finite), "nonfinite_n": len(xs)-len(finite),
            "p50": quantile(finite, 50), "p90": quantile(finite, 90),
            "p95": quantile(finite, 95), "p99": quantile(finite, 99),
            "max": quantile(finite, 100)}


def row_key(r):
    return tuple(r[k] for k in ("case_id", "configuration_id", "profile", "d_bin_index",
        "epoch_index", "trajectory_id", "trajectory_pos", "policy", "target"))


def merge_reps(root: Path, arm: str, policy: str):
    suffix = "value" if policy == "None" else "jac"
    reps = []
    for i in range(3):
        path = root / "raw" / f"{arm}_{suffix}_r{i}.tsv.gz"
        rows = read_tsv(path)
        if len(rows) != 14432:
            raise ValueError(f"{path}: got {len(rows)} data rows, expected 14432")
        keyed = {row_key(r): r for r in rows}
        if len(keyed) != len(rows):
            raise ValueError(f"duplicate keys in {path}")
        reps.append(keyed)
    if any(set(reps[0]) != set(r) for r in reps[1:]):
        raise ValueError(f"row key mismatch for {arm}/{policy}")
    floats = [f"{lane}_{field}" for lane in ("cold", "warm", "radial")
              for field in ("ms", "topology_ms", "adaptive_ms", "mu", "value_error",
                            "nodes", "evaluations", "panels", "splits")]
    floats += [f"{lane}_g{j}{suffix}" for lane in ("cold", "warm", "radial")
               for j in range(5) for suffix in ("", "_error")]
    string_status = [f"{lane}_{field}" for lane in ("cold", "warm", "radial")
                     for field in ("value_converged", "value_stop", "stop", "status",
                                   "topology_status", "topology_cells", "topology_events",
                                   "topology_hash")]
    string_status += [f"{lane}_g{j}_{suffix}" for lane in ("cold", "warm", "radial")
                      for j in range(5) for suffix in ("quality", "reason")]
    string_status += ["warm_l1", "warm_l2", "warm_l3", "warm_rescreen_fail", "warm_seed_used"]
    merged = {}
    for key in reps[0]:
        rr = [r[key] for r in reps]
        base = dict(rr[0])
        for field in floats:
            base[field] = statistics.median(float(r[field]) for r in rr)
        for field in string_status:
            vals = [r[field] for r in rr]
            if len(set(vals)) != 1:
                raise ValueError(f"replicate mismatch {arm}/{policy}/{key}/{field}: {vals}")
            base[field] = vals[0]
        merged[key] = base
    return merged


def selected(rows, target, profile="all", scope="all"):
    out = [r for r in rows if abs(float(r["target"]) - target) < 1e-15]
    if profile != "all":
        out = [r for r in out if r["profile"] == profile]
    if scope == "steady":
        out = [r for r in out if int(r["trajectory_pos"]) > 0]
    elif scope == "first":
        out = [r for r in out if int(r["trajectory_pos"]) == 0]
    return out


def arm_quality(rows, lane, target):
    errors = [abs(float(r[f"{lane}_mu"])-float(r["reference"])) /
              max(abs(float(r["reference"])), 1e-300) for r in rows]
    return {
        "rows": len(rows),
        "value_converged": sum(int(r[f"{lane}_value_converged"]) for r in rows),
        "status_counts": dict(sorted(Counter(r[f"{lane}_status"] for r in rows).items())),
        "value_stop_counts": dict(sorted(Counter(r[f"{lane}_value_stop"] for r in rows).items())),
        "relative_error_to_VBM_1e-6": dist(errors),
        "observed_errors_above_requested_rtol": sum(e > target for e in errors),
        "median_nodes": quantile([float(r[f"{lane}_nodes"]) for r in rows], 50),
    }


def compare_rows(left, right, policy, lane):
    if set(left) != set(right):
        raise ValueError("arm key sets differ")
    rows = [(left[k], right[k]) for k in left]
    mu_diff = [abs(float(b[f"{lane}_mu"])-float(a[f"{lane}_mu"])) for a,b in rows]
    ref_scaled = [d/max(abs(float(a["reference"])),1e-300)
                  for d,(a,b) in zip(mu_diff,rows)]
    result = {
        "rows": len(rows),
        "right_over_left_time_ratio": dist(float(b[f"{lane}_ms"])/max(float(a[f"{lane}_ms"]),1e-300)
                                           for a,b in rows),
        "speedup_left_over_right_ratio_of_paired_medians":
            quantile([float(a[f"{lane}_ms"]) for a,b in rows],50) /
            max(quantile([float(b[f"{lane}_ms"]) for a,b in rows],50),1e-300),
        "absolute_mu_difference": dist(mu_diff),
        "mu_difference_scaled_by_VBM_reference": dist(ref_scaled),
        "status_mismatches": sum(a[f"{lane}_status"] != b[f"{lane}_status"] for a,b in rows),
        "value_convergence_mismatches": sum(a[f"{lane}_value_converged"] != b[f"{lane}_value_converged"] for a,b in rows),
        "stop_mismatches": sum(a[f"{lane}_value_stop"] != b[f"{lane}_value_stop"] or
                               a[f"{lane}_stop"] != b[f"{lane}_stop"] for a,b in rows),
        "topology_event_cell_count_mismatches": sum(
            a[f"{lane}_topology_events"] != b[f"{lane}_topology_events"] or
            a[f"{lane}_topology_cells"] != b[f"{lane}_topology_cells"] for a,b in rows),
        "topology_hash_mismatches": sum(a[f"{lane}_topology_hash"] != b[f"{lane}_topology_hash"] for a,b in rows),
        "node_count_mismatches": sum(a[f"{lane}_nodes"] != b[f"{lane}_nodes"] for a,b in rows),
    }
    if policy == "ValueFirst":
        result["gradient"] = {}
        for j in range(5):
            result["gradient"][str(j)] = {
                "value_difference": dist(abs(float(b[f"{lane}_g{j}"])-float(a[f"{lane}_g{j}"]))
                                          for a,b in rows),
                "quality_mismatches": sum(a[f"{lane}_g{j}_quality"] != b[f"{lane}_g{j}_quality"] for a,b in rows),
                "reason_mismatches": sum(a[f"{lane}_g{j}_reason"] != b[f"{lane}_g{j}_reason"] for a,b in rows),
            }
    return result


def whole_summary(root):
    data = {arm: {p: merge_reps(root, arm, p) for p in POLICIES} for arm in ARMS}
    out = {"physical_input_rows":7216,"trajectory_count":1804,"epochs_per_trajectory":4,
           "result_rows_per_arm_policy":14432,"repetitions":3,
           "policies":{"None":"value-only","ValueFirst":"value + analytic 5Jac"},
           "tol_targets":[1e-3,1e-4],"arms":{},"comparisons":{},"timing":{},"checks":{}}
    for arm in ARMS:
        out["arms"][arm] = {}
        for policy in POLICIES:
            rows = list(data[arm][policy].values())
            out["arms"][arm][policy] = {}
            for target in (1e-3,1e-4):
                subset_rows = selected(rows,target)
                out["arms"][arm][policy][f"RelTol={target:g}"] = {
                    lane: arm_quality(subset_rows,lane,target) for lane in ("cold","warm","radial")}
    for policy in POLICIES:
        out["comparisons"][policy] = {}
        out["timing"][policy] = {}
        for target in (1e-3,1e-4):
            tkey=f"RelTol={target:g}"
            out["comparisons"][policy][tkey]={}
            out["timing"][policy][tkey]={}
            for left,right in (("baseline","qf"),("baseline","screen"),("qf","screen")):
                pair=f"{right}_vs_{left}"
                out["comparisons"][policy][tkey][pair]={}
                for lane in ("cold","warm","radial"):
                    lmap={k:r for k,r in data[left][policy].items()
                          if abs(float(k[-1])-target)<1e-15}
                    rmap={k:r for k,r in data[right][policy].items()
                          if abs(float(k[-1])-target)<1e-15}
                    out["comparisons"][policy][tkey][pair][lane]=compare_rows(
                        lmap,rmap,policy,lane)
            for profile in ("all","uniform","linear"):
                out["timing"][policy][tkey][profile]={}
                for scope in ("all","first","steady"):
                    out["timing"][policy][tkey][profile][scope]={}
                    for lane in ("cold","warm","radial"):
                        cell={}
                        group={arm:selected(list(data[arm][policy].values()),target,profile,scope)
                               for arm in ARMS}
                        for arm in ARMS:
                            cell[arm]=dist(float(r[f"{lane}_ms"]) for r in group[arm])
                        for left,right in (("baseline","qf"),("baseline","screen"),("qf","screen")):
                            lmap=data[left][policy];rmap=data[right][policy]
                            keys={row_key(r) for r in group[left]}
                            ratios=[float(rmap[k][f"{lane}_ms"])/max(float(lmap[k][f"{lane}_ms"]),1e-300)
                                    for k in keys]
                            cell[f"{right}_over_{left}_paired_ratio"]=dist(ratios)
                        out["timing"][policy][tkey][profile][scope][lane]=cell
    out["checks"]={"all_three_arms_have_matched_keys":True,
                   "input_reference":"VBM RelTol=1e-6 values in the frozen Phase 75 input snapshot; observed comparison, not a formal bound"}
    return out


def targeted_summary(root):
    target_rows={arm:{pol:read_tsv(root/"raw"/f"targeted_{arm}_{'value' if pol=='None' else 'jac'}.tsv")
                      for pol in POLICIES} for arm in ARMS}
    for arm in ARMS:
        for pol in POLICIES:
            if len(target_rows[arm][pol])!=2304:
                raise ValueError(f"targeted {arm}/{pol} rows {len(target_rows[arm][pol])}, expected 2304")
    out={"fixture":"LensParams{X=.12,Y=[-1e-6,0,+1e-6,0],rho=.02,q=.5,a=1}; 48 measured repeats after one warmup trajectory",
         "meaning":"synthetic topology trajectory toggling no-hit/hit; not a population benchmark",
         "arms":{},"pairwise":{},"projective_fold_pattern":{},"independent_reference":{}}
    for arm in ARMS:
        out["arms"][arm]={}
        for pol in POLICIES:
            rows=target_rows[arm][pol]
            out["arms"][arm][pol]={}
            # Count the fold classification once per epoch rather than once per timing lane/repeat.
            patterns={}
            for r in rows:
                k=(int(r["epoch"]),float(r["ys"]))
                patterns[k]=(int(r["chart_events"]),int(r["projective_folds"]),r["topology_status"])
            event_pattern = [
                {"epoch":k[0],"Y":k[1],"chart_events":v[0],"projective_folds":v[1],"topology_status":v[2]}
                for k,v in sorted(patterns.items())]
            out["arms"][arm][pol]["event_pattern"] = event_pattern
            if not out["projective_fold_pattern"]:
                out["projective_fold_pattern"] = event_pattern
            elif event_pattern != out["projective_fold_pattern"]:
                raise ValueError(f"targeted {arm}/{pol} event pattern differs from baseline")
            for target in (1e-3,1e-4):
                out["arms"][arm][pol][f"RelTol={target:g}"]={}
                for mode,lane in LANES.items():
                    rs=[r for r in rows if abs(float(r["target"])-target)<1e-15 and r["mode"]==mode]
                    out["arms"][arm][pol][f"RelTol={target:g}"][lane]={
                        "rows":len(rs),"time_ms":dist(float(r["wall_ms"]) for r in rs),
                        "value_converged":sum(int(r["value_converged"]) for r in rs),
                        "status_counts":dict(sorted(Counter(r["status"] for r in rs).items())),
                        "max_nodes":max(int(r["nodes"]) for r in rs),
                    }
    for pol in POLICIES:
        out["pairwise"][pol]={}
        for left,right in (("baseline","qf"),("qf","screen"),("baseline","screen")):
            out["pairwise"][pol][f"{right}_vs_{left}"]={}
            lrows=target_rows[left][pol];rrows=target_rows[right][pol]
            if len(lrows)!=len(rrows):raise ValueError("targeted row counts differ")
            for target in (1e-3,1e-4):
                for mode,lane in LANES.items():
                    pairs=[(a,b) for a,b in zip(lrows,rrows)
                           if abs(float(a["target"])-target)<1e-15 and a["mode"]==mode]
                    cell={"rows":len(pairs),
                          "right_over_left_time_ratio":dist(float(b["wall_ms"])/max(float(a["wall_ms"]),1e-300) for a,b in pairs),
                          "mu_abs_difference":dist(abs(float(b["mu"])-float(a["mu"])) for a,b in pairs),
                          "value_convergence_mismatches":sum(a["value_converged"]!=b["value_converged"] for a,b in pairs),
                          "status_mismatches":sum(a["status"]!=b["status"] for a,b in pairs),
                          "topology_status_mismatches":sum(a["topology_status"]!=b["topology_status"] for a,b in pairs),
                          "cell_count_mismatches":sum(a["cells"]!=b["cells"] for a,b in pairs),
                          "event_count_mismatches":sum(a["events"]!=b["events"] for a,b in pairs)}
                    if pol=="ValueFirst":
                        cell["gradient_abs_difference"]={str(j):dist(abs(float(b[f"g{j}"])-float(a[f"g{j}"])) for a,b in pairs) for j in range(5)}
                        cell["gradient_quality_mismatches"]={str(j):sum(a[f"g{j}_quality"]!=b[f"g{j}_quality"] for a,b in pairs) for j in range(5)}
                    out["pairwise"][pol][f"{right}_vs_{left}"][f"RelTol={target:g}/{lane}"]=cell
    screen_rows=target_rows["screen"]["ValueFirst"]
    fdrows=[r for r in screen_rows if float(r["u"])==0.0 and float(r["ys"])==0.0 and
            r["mode"]=="full-cold" and abs(float(r["target"])-1e-4)<1e-15]
    out["independent_reference"]={"parameter":"a","u":0.0,"fixture_X":0.12,"Y":0.0,
        "fd_reference_phase74":FD_A,"source":"Phase 74 independent cold-topology value finite-difference/GL study; observed reference, not formal bound",
        "baseline_gradient_a_error":dist(abs(float(r["g4"])-FD_A) for r in
             [x for x in target_rows["baseline"]["ValueFirst"] if float(x["u"])==0 and float(x["ys"])==0 and x["mode"]=="full-cold" and abs(float(x["target"])-1e-4)<1e-15]),
        "qf_projective_gradient_a_error":dist(abs(float(r["g4"])-FD_A) for r in
             [x for x in target_rows["qf"]["ValueFirst"] if float(x["u"])==0 and float(x["ys"])==0 and x["mode"]=="full-cold" and abs(float(x["target"])-1e-4)<1e-15]),
        "screen_gradient_a_error":dist(abs(float(r["g4"])-FD_A) for r in fdrows)}
    return out


def screen_scan_summary(root):
    out={}
    for key in ("reference","trajectory"):
        rows=read_tsv(root/f"{key}_screen_scan.tsv")
        old=Counter(r["old_decision"] for r in rows)
        screen=Counter(r["fast_screen"] for r in rows)
        rejected=[r for r in rows if r["fast_screen"] in
                  ("no_d14_overlap","p3_outside_qf_contact_tolerance")]
        out[key]={
            "input_rows":110 if key=="reference" else 7216,
            "chart_p4_records":len(rows),
            "old_qf_decision_counts":dict(sorted(old.items())),
            "fast_screen_decision_counts":dict(sorted(screen.items())),
            "false_rejects_of_old_accepted":sum(int(r["fast_reject_old_accept"]) for r in rows),
            "old_qf_probe_ms_sum":sum(float(r["old_probe_ms"]) for r in rows),
            "fast_screen_ms_sum":sum(float(r["fast_screen_ms"]) for r in rows),
            "fast_screen_ms_on_early_rejects":sum(float(r["fast_screen_ms"]) for r in rejected),
            "old_p4_newton_steps_sum":sum(int(r["old_p4_steps"]) for r in rows),
        }
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,required=True)
    root=ap.parse_args().root
    result={"phase":"76 cheap fail-open screen for projective fold detection",
            "screen_scans":screen_scan_summary(root),
            "whole_epoch":whole_summary(root),
            "true_fold_trajectory":targeted_summary(root)}
    print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))


if __name__=="__main__":
    main()
