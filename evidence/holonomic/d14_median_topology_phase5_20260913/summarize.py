#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
KW = dict(sep=r"\s+", comment="#", low_memory=False)

root_base = pd.read_csv(
    ROOT / "evidence/holonomic/d14_event_contract_interval_20260913/"
           "rootwork_baseline_final_timing.tsv", **KW)
pair = pd.read_csv(HERE / "pair_shared_rootwork.tsv.gz", **KW)
structured = pd.read_csv(HERE / "structured_residual_rootwork.tsv.gz", **KW)
whole_base = pd.read_csv(
    ROOT / "evidence/holonomic/d14_median_tail_phase4_20260913/"
           "whole_baseline.tsv.gz", **KW)
mid = pd.read_csv(HERE / "mid_schedule_whole.tsv.gz", **KW)
no_soft = pd.read_csv(HERE / "no_soft_current.tsv.gz", **KW)
root_keys = ["case_id", "configuration_id", "profile", "d_bin_index",
             "epoch_index", "lane"]
whole_keys = ["case_id", "configuration_id", "profile", "d_bin_index",
              "epoch_index", "target"]

def root_ab(candidate):
    x = root_base.merge(candidate, on=root_keys, suffixes=("_base", "_candidate"))
    result = {}
    for lane in ("cold", "warm"):
        y = x[x.lane == lane]
        result[lane] = {
            "rows": len(y),
            "topology_status_mismatches": int(
                (y.topology_status_base != y.topology_status_candidate).sum()),
            "event_count_mismatches": int((y.events_base != y.events_candidate).sum()),
            "qf_cold_base": int(y.qf_cold_calls_base.sum()),
            "qf_cold_candidate": int(y.qf_cold_calls_candidate.sum()),
        }
        for column in ("whole_classify_ms", "d14_solve_ms", "presearch_ms",
                       "residual_eval_ms"):
            a, b = y[column + "_base"], y[column + "_candidate"]
            result[lane][column] = {
                "base_p50_ms": float(a.median()),
                "candidate_p50_ms": float(b.median()),
                "delta_percent": float(100 * (b.median() / a.median() - 1)),
            }
    return result

def whole_ab(candidate):
    x = whole_base.merge(candidate, on=whole_keys,
                         suffixes=("_base", "_candidate"))
    result = {}
    for target in sorted(x.target.unique(), reverse=True):
        y = x[x.target == target]
        result[f"{target:g}"] = {}
        for lane in ("cold", "warm", "radial"):
            a, b = y[lane + "_ms_base"], y[lane + "_ms_candidate"]
            result[f"{target:g}"][lane] = {
                "coverage": int(y[lane + "_value_converged_candidate"].sum()),
                "status_mismatches": int(
                    (y[lane + "_status_base"] != y[lane + "_status_candidate"]).sum()),
                "mu_max_absdiff": float(np.max(np.abs(
                    y[lane + "_mu_base"] - y[lane + "_mu_candidate"]))),
                "p50_delta_percent": float(100 * (b.median() / a.median() - 1)),
                "p99_delta_percent": float(100 * (b.quantile(.99) / a.quantile(.99) - 1)),
                "max_delta_percent": float(100 * (b.max() / a.max() - 1)),
            }
    return result

def no_soft_audit():
    x = whole_base.merge(no_soft, on=whole_keys,
                         suffixes=("_base", "_no_soft"))
    result = {}
    for target in sorted(x.target.unique(), reverse=True):
        y = x[x.target == target]
        result[f"{target:g}"] = {}
        for lane in ("cold", "warm", "radial"):
            lost = ((y[lane + "_value_converged_base"] == 1) &
                    (y[lane + "_value_converged_no_soft"] == 0))
            result[f"{target:g}"][lane] = {
                "coverage": int(y[lane + "_value_converged_no_soft"].sum()),
                "lost_vs_all_soft": int(lost.sum()),
                "status_mismatches": int(
                    (y[lane + "_status_base"] != y[lane + "_status_no_soft"]).sum()),
            }
    return result

summary = {
    "pair_shared_jacobi": root_ab(pair),
    "structured_residual": root_ab(structured),
    "mid_schedule": whole_ab(mid),
    "no_projected_complex_soft": no_soft_audit(),
}
(HERE / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
