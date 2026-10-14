#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
KEYS = ["case_id", "configuration_id", "profile", "d_bin_index",
        "epoch_index", "target"]
ROOT_KEYS = ["case_id", "configuration_id", "profile", "d_bin_index",
             "epoch_index", "rep", "lane"]


def read(name, comment=None):
    path = HERE / name
    if not path.exists():
        path = HERE / (name + ".gz")
    return pd.read_csv(path, sep=r"\s+", comment=comment)


baseline = read("whole_baseline_final.tsv", comment="#")
candidate = read("whole_interval_final.tsv", comment="#")
paired = baseline.merge(candidate, on=KEYS, suffixes=("_baseline", "_candidate"),
                        validate="one_to_one")
root_base = read("rootwork_baseline_final_timing.tsv")
root_new = read("rootwork_interval_final_timing.tsv")
roots = root_base.merge(root_new, on=ROOT_KEYS,
                        suffixes=("_baseline", "_candidate"),
                        validate="one_to_one")
clusters = read("clusters_final.tsv")
# The input intentionally contains linear and log profiles of the same geometry.
unique_clusters = clusters[(clusters.profile == "linear") &
                           (clusters.component_size > 1)].copy()


def percentiles(base, new):
    result = {}
    for quantile, name in [(0.5, "p50"), (0.9, "p90"), (0.95, "p95"),
                           (0.99, "p99"), (1.0, "max")]:
        b = float(base.quantile(quantile))
        n = float(new.quantile(quantile))
        result[name] = {"baseline_ms": b, "candidate_ms": n,
                        "delta_percent": 100.0 * (n / b - 1.0)}
    return result


summary = {"rows": len(paired), "whole": {}, "root_work": {},
           "correctness": {}, "clusters": {}}
for target in sorted(paired.target.unique(), reverse=True):
    frame = paired[paired.target == target]
    target_result = {}
    for lane in ["cold", "warm", "radial"]:
        stats = percentiles(frame[f"{lane}_ms_baseline"],
                            frame[f"{lane}_ms_candidate"])
        reference_error = np.abs(frame[f"{lane}_mu_candidate"] -
                                 frame.reference_candidate)
        limit = np.maximum(1e-12, target * np.abs(frame.reference_candidate))
        stats["coverage"] = int(frame[f"{lane}_value_converged_candidate"].sum())
        stats["observed_reference_violations"] = int(
            (reference_error > limit).sum())
        target_result[lane] = stats
    summary["whole"][f"{target:g}"] = target_result

discrete = ["topology_status", "topology_cells", "topology_events"]
for lane in ["cold", "warm", "radial"]:
    discrete += [f"{lane}_value_converged", f"{lane}_stop", f"{lane}_status"]
summary["correctness"]["discrete_mismatches"] = {
    name: int((paired[f"{name}_baseline"].astype(str) !=
               paired[f"{name}_candidate"].astype(str)).sum())
    for name in discrete
}
for lane in ["cold", "warm", "radial"]:
    delta = np.abs(paired[f"{lane}_mu_baseline"] -
                   paired[f"{lane}_mu_candidate"])
    summary["correctness"][f"{lane}_mu"] = {
        "changed_rows": int(np.count_nonzero(delta)),
        "max_abs": float(delta.max()),
        "max_scaled": float(np.max(delta / np.maximum(
            1.0, np.abs(paired[f"{lane}_mu_baseline"])))),
    }

accepted_keys = set()
for lane in ["cold", "warm"]:
    frame = roots[roots.lane == lane]
    accepted = frame[frame.event_contract_accepts_candidate > 0]
    accepted_keys.update((int(row.case_id), int(row.configuration_id), row.profile,
                          int(row.d_bin_index), int(row.epoch_index), lane)
                         for row in accepted.itertuples())
    lane_result = {
        "rows": len(frame),
        "attempts": int(frame.event_contract_attempts_candidate.sum()),
        "accepts": int(frame.event_contract_accepts_candidate.sum()),
        "qf_cold_calls_baseline": int(frame.qf_cold_calls_baseline.sum()),
        "qf_cold_calls_candidate": int(frame.qf_cold_calls_candidate.sum()),
        "qf_cold_sweeps_baseline": int(frame.qf_cold_sweeps_baseline.sum()),
        "qf_cold_sweeps_candidate": int(frame.qf_cold_sweeps_candidate.sum()),
        "event_count_mismatches": int(
            (frame.events_baseline != frame.events_candidate).sum()),
        "physical_count_mismatches": int(
            (frame.physical_real_events_baseline !=
             frame.physical_real_events_candidate).sum()),
        "soft_count_mismatches": int(
            (frame.soft_events_baseline != frame.soft_events_candidate).sum()),
        "classify_all": percentiles(frame.whole_classify_ms_baseline,
                                     frame.whole_classify_ms_candidate),
    }
    if len(accepted):
        lane_result["classify_accepted"] = percentiles(
            accepted.whole_classify_ms_baseline,
            accepted.whole_classify_ms_candidate)
    failed = frame[(frame.event_contract_attempts_candidate > 0) &
                   (frame.event_contract_accepts_candidate == 0)]
    if len(failed):
        overhead = failed.whole_classify_ms_candidate - failed.whole_classify_ms_baseline
        lane_result["failed_attempt_overhead_ms"] = {
            "count": len(failed), "p50": float(overhead.quantile(0.5)),
            "p90": float(overhead.quantile(0.9)), "max": float(overhead.max())}
    summary["root_work"][lane] = lane_result

summary["clusters"] = {
    "components": len(unique_clusters),
    "size_histogram": {str(int(k)): int(v) for k, v in
                       unique_clusters.component_size.value_counts().sort_index().items()},
    "soft_only": int(((unique_clusters.physical_real == 0) &
                       (unique_clusters.positive_real == 0)).sum()),
    "physical_containing": int((unique_clusters.physical_real > 0).sum()),
    "cluster_certificates": int(unique_clusters.cluster_certificate.sum()),
    "certificates_crossing_real_axis": int(unique_clusters.loc[
        unique_clusters.cluster_certificate == 1, "crosses_real_axis"].sum()),
}

(HERE / "summary.json").write_text(json.dumps(summary, indent=2,
                                                sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
