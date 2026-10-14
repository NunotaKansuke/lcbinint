#!/usr/bin/env python3
"""Summarize paired uniform-only Phase 71 primary and diagnostic raw files."""
import csv
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_space_table(path):
    with path.open() as stream:
        header = stream.readline().split()
        return [dict(zip(header, line.split())) for line in stream if line.strip()]


def finite(row, key):
    try:
        value = float(row[key])
        return math.isfinite(value)
    except (KeyError, ValueError):
        return False


def nearest_rank(values, q):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)] if ordered else None


primary_runs = {
    "rep1": read_space_table(ROOT / "trajectory.tsv"),
    "rep2": read_space_table(ROOT / "trajectory_rep2.tsv"),
}
trajectory = [row for run in primary_runs.values() for row in run]
micro_path = ROOT / "trajectory_microtimed.tsv"
micro_trajectory = read_space_table(micro_path) if micro_path.exists() else []
panels = read_space_table(ROOT / "panels.tsv")
summary = {
    "source_commit": "9e409a3c096bab7ca2cec7c9a8d87e47c0cc1d6d",
    "mode": "uniform u=0, value-only, two paired primary passes: full cold, full warm, radial-only",
    "input": "evidence/holonomic/v2_adaptive_best_trajectory_20260913/input_snapshot.tsv",
    "raw_rows": len(trajectory),
    "trajectory_epoch_count_per_tolerance": len(trajectory) // 2,
    "timing_quantile_method": "nearest rank",
    "machine": "Intel Xeon Gold 6530, benchmark pinned to CPU 0",
    "trajectory_repetitions": len(primary_runs),
    "primary_wall_timing": "trajectory.tsv and trajectory_rep2.tsv, built without per-jet/model microtimers",
    "microtiming_source": ("trajectory_microtimed.tsv, separate diagnostic run; do not use its whole-epoch wall values"
                           if micro_trajectory else None),
    "cold_definition": "D14/topology built each epoch; no cross-epoch seed",
    "warm_definition": "prepared L2 D14 warm roots; topology reuse disabled; first row naturally cold-seeded",
    "cold_warm": {},
    "primary_replicates": {},
    "shadow_work": {},
    "panel_audit": {},
}

for tol in sorted({row["tol"] for row in trajectory}):
    rows = [row for row in trajectory if row["tol"] == tol]
    tol_summary = {"rows": len(rows), "cold": {}, "warm": {}}
    for mode in ("cold", "warm"):
        base = [float(row[f"{mode}_ms"]) for row in rows]
        shadow = [float(row[f"shadow_{mode}_ms"]) for row in rows]
        tol_summary[mode] = {
            "baseline_ms": {f"p{int(q*100)}": nearest_rank(base, q)
                            for q in (0.5, 0.9, 0.95, 0.99)},
            "shadow_ms": {f"p{int(q*100)}": nearest_rank(shadow, q)
                          for q in (0.5, 0.9, 0.95, 0.99)},
            "baseline_max_ms": max(base),
            "shadow_max_ms": max(shadow),
            "baseline_sum_ms": sum(base),
            "shadow_sum_ms": sum(shadow),
            "paired_sum_change_percent": 100.0 * (sum(shadow) / sum(base) - 1.0),
            "baseline_value_converged": sum(row[f"{mode}_ok"] == "1" for row in rows),
            "shadow_value_converged": sum(row[f"shadow_{mode}_ok"] == "1" for row in rows),
            "status_mismatches": sum(row[f"{mode}_status"] != row[f"shadow_{mode}_status"] for row in rows),
            "mu_mismatches": sum(row[f"{mode}_mu"] != row[f"shadow_{mode}_mu"] for row in rows),
            "node_count_mismatches": sum(row[f"{mode}_nodes"] != row[f"shadow_{mode}_nodes"] for row in rows),
        }
    radial = [float(row["radial_ms"]) for row in rows]
    shadow_radial = [float(row["shadow_radial_ms"]) for row in rows]
    tol_summary["radial_only"] = {
        "baseline_ms": {f"p{int(q*100)}": nearest_rank(radial, q)
                        for q in (0.5, 0.9, 0.95, 0.99)},
        "shadow_ms": {f"p{int(q*100)}": nearest_rank(shadow_radial, q)
                      for q in (0.5, 0.9, 0.95, 0.99)},
        "baseline_max_ms": max(radial),
        "shadow_max_ms": max(shadow_radial),
        "baseline_sum_ms": sum(radial),
        "shadow_sum_ms": sum(shadow_radial),
        "paired_sum_change_percent": 100.0 * (sum(shadow_radial) / sum(radial) - 1.0),
        "baseline_value_converged": sum(row["radial_ok"] == "1" for row in rows),
        "shadow_value_converged": sum(row["shadow_radial_ok"] == "1" for row in rows),
        "status_mismatches": sum(row["radial_status"] != row["shadow_radial_status"] for row in rows),
        "mu_mismatches": sum(row["radial_mu"] != row["shadow_radial_mu"] for row in rows),
        "node_count_mismatches": sum(row["radial_nodes"] != row["shadow_radial_nodes"] for row in rows),
        "timing_boundary": "classify_cells / D14 topology excluded; flux_adaptive_integrate included",
    }
    summary["cold_warm"][tol] = tol_summary
    for replicate, replicate_rows in primary_runs.items():
        rep_rows = [row for row in replicate_rows if row["tol"] == tol]
        summary["primary_replicates"].setdefault(replicate, {})[tol] = {
            mode: {
                "rows": len(rep_rows),
                "baseline_sum_ms": sum(float(row[f"{mode}_ms"]) for row in rep_rows),
                "shadow_sum_ms": sum(float(row[f"shadow_{mode}_ms"]) for row in rep_rows),
                "paired_sum_change_percent": 100.0 * (
                    sum(float(row[f"shadow_{mode}_ms"]) for row in rep_rows) /
                    sum(float(row[f"{mode}_ms"]) for row in rep_rows) - 1.0),
            }
            for mode in ("cold", "warm")
        }
        summary["primary_replicates"][replicate][tol]["radial_only"] = {
            "rows": len(rep_rows),
            "baseline_sum_ms": sum(float(row["radial_ms"]) for row in rep_rows),
            "shadow_sum_ms": sum(float(row["shadow_radial_ms"]) for row in rep_rows),
            "paired_sum_change_percent": 100.0 * (
                sum(float(row["shadow_radial_ms"]) for row in rep_rows) /
                sum(float(row["radial_ms"]) for row in rep_rows) - 1.0),
        }
    summary["shadow_work"][tol] = {
        "cold_jet_attempts": sum(int(row["cold_jet_attempts"]) for row in rows),
        "cold_jet_successes": sum(int(row["cold_jet_successes"]) for row in rows),
        "cold_panels_shadowed": sum(int(row["cold_shadow_panels"]) for row in rows),
        "cold_panel_local_budget_passes_not_global": sum(int(row["cold_local_budget_passes"]) for row in rows),
        "warm_jet_attempts": sum(int(row["warm_jet_attempts"]) for row in rows),
        "warm_jet_successes": sum(int(row["warm_jet_successes"]) for row in rows),
        "warm_panels_shadowed": sum(int(row["warm_shadow_panels"]) for row in rows),
        "warm_panel_local_budget_passes_not_global": sum(int(row["warm_local_budget_passes"]) for row in rows),
        "radial_jet_attempts": sum(int(row["radial_jet_attempts"]) for row in rows),
        "radial_jet_successes": sum(int(row["radial_jet_successes"]) for row in rows),
        "radial_panels_shadowed": sum(int(row["radial_shadow_panels"]) for row in rows),
        "radial_panel_local_budget_passes_not_global": sum(int(row["radial_local_budget_passes"]) for row in rows),
    }
    if micro_trajectory:
        micro_rows = [row for row in micro_trajectory if row["tol"] == tol]
        summary.setdefault("separate_microtiming", {})[tol] = {
            mode: {
                "rows": len(micro_rows),
                "jet_ms_sum": sum(float(row[f"{mode}_jet_ms"]) for row in micro_rows),
                "panel_model_ms_sum": sum(float(row[f"{mode}_model_ms"]) for row in micro_rows),
                "instrumented_baseline_whole_sum_ms": sum(float(row[f"{mode}_ms"]) for row in micro_rows),
                "instrumented_shadow_whole_sum_ms": sum(float(row[f"shadow_{mode}_ms"]) for row in micro_rows),
                "instrumented_paired_sum_change_percent": 100.0 * (
                    sum(float(row[f"shadow_{mode}_ms"]) for row in micro_rows) /
                    sum(float(row[f"{mode}_ms"]) for row in micro_rows) - 1.0),
            }
            for mode in ("cold", "warm")
        }
        summary["separate_microtiming"][tol]["radial_only"] = {
            "rows": len(micro_rows),
            "jet_ms_sum": sum(float(row["radial_jet_ms"]) for row in micro_rows),
            "panel_model_ms_sum": sum(float(row["radial_model_ms"]) for row in micro_rows),
            "instrumented_baseline_whole_sum_ms": sum(float(row["radial_ms"]) for row in micro_rows),
            "instrumented_shadow_whole_sum_ms": sum(float(row["shadow_radial_ms"]) for row in micro_rows),
            "instrumented_paired_sum_change_percent": 100.0 * (
                sum(float(row["shadow_radial_ms"]) for row in micro_rows) /
                sum(float(row["radial_ms"]) for row in micro_rows) - 1.0),
        }

usable = [row for row in panels
          if row.get("jet_count") == "7" and finite(row, "qh_error")
          and finite(row, "ref_unc") and finite(row, "ref128")
          and float(row["ref_unc"]) < 1e-7 * max(1.0, abs(float(row["ref128"]))) ]
local_passes = [row for row in usable if row["panel_local_budget_pass"] == "1"]
model_only_passes = [row for row in usable
                     if row["model_budget_pass_ignoring_value_detail"] == "1"]
underestimates = [row for row in usable if row["model_under"] == "1"]
false_local_passes = [row for row in local_passes if row["model_under"] == "1"]
derivative_guard_rows = [row for row in usable if finite(row, "derivative_detail3")
                         and finite(row, "derivative_detail7")]
ift_rows = [float(row["jet_endpoint_ift_max_scaled"]) for row in usable
            if finite(row, "jet_endpoint_ift_max_scaled")]
higha_parent = [row for row in panels if row["name"] == "paper_highA"
                and row["cell"] == "11" and row["depth"] == "0"]
higha_false = [row for row in false_local_passes if row["name"] == "paper_highA"]
summary["panel_audit"] = {
    "raw_panel_rows": len(panels),
    "usable_independent_GL64_GL128_reference_rows": len(usable),
    "base_value_detail_resolved_rows": sum(row["value_resolved"] == "1" for row in usable),
    "base_value_detail_unresolved_rows": sum(row["value_resolved"] != "1" for row in usable),
    "derivative_detail_decay_pass_rows": sum(row["derivative_detail_decays"] == "1"
                                             for row in derivative_guard_rows),
    "base_unresolved_rows_also_failing_derivative_decay": sum(
        row["value_resolved"] != "1" and row["derivative_detail_decays"] != "1"
        for row in derivative_guard_rows),
    "hermite7_more_accurate_than_fejer7": sum(float(row["qh_error"]) < float(row["q7_error"]) for row in usable),
    "median_abs_error_fejer7": statistics.median(float(row["q7_error"]) for row in usable),
    "median_abs_error_hermite7": statistics.median(float(row["qh_error"]) for row in usable),
    "local_model_budget_passes_not_global": len(local_passes),
    "model_budget_passes_ignoring_required_value_detail": len(model_only_passes),
    "unresolved_panels_that_would_pass_model_budget_if_detail_guard_were_removed": sum(
        row["value_resolved"] != "1" for row in model_only_passes),
    "local_model_passes_that_could_save_nodes_under_existing_detail_guard": sum(
        row["value_resolved"] != "1" for row in local_passes),
    "model_underestimates_observed_H7_error": len(underestimates),
    "false_local_passes": len(false_local_passes),
    "max_false_local_pass_underestimate_ratio": max(
        (float(row["qh_error"]) / (float(row["model"]) + float(row["ref_unc"]))
         for row in false_local_passes), default=None),
    "false_local_pass_examples": [
        {key: row[key] for key in ("name", "tol", "panel", "cell", "depth", "qh_error", "model", "ref_unc")}
        for row in sorted(false_local_passes,
                          key=lambda row: float(row["qh_error"]) /
                          max(1e-300, float(row["model"]) + float(row["ref_unc"])),
                          reverse=True)[:8]
    ],
    "jet_vs_independent_endpoint_IFT_scaled_error": {
        "n": len(ift_rows),
        "p50": nearest_rank(ift_rows, 0.5),
        "p95": nearest_rank(ift_rows, 0.95),
        "p99": nearest_rank(ift_rows, 0.99),
        "max": max(ift_rows) if ift_rows else None,
        "reference": "separate reference_arcs + polished endpoints + endpoint_dR",
    },
    "paper_highA_parent_rows": [
        {key: row[key] for key in ("tol", "panel", "cell", "depth", "q7", "qh3", "qh7", "model",
                                   "panel_local_budget_pass", "ref128", "ref_unc", "qh_error", "q7_error")}
        for row in higha_parent
    ],
    "paper_highA_false_local_passes": [
        {key: row[key] for key in ("tol", "panel", "cell", "depth", "qh_error", "model", "ref_unc")}
        for row in higha_false
    ],
}

reference_path = ROOT / "reference_cold.csv"
if reference_path.exists():
    with reference_path.open(newline="") as stream:
        reference = list(csv.DictReader(stream))
    summary["value_reference"] = {
        "raw_rows": len(reference),
        "reference_usable_rows": sum(row["reference_usable"] == "1" for row in reference),
        "observed_violations": sum(row["violation"] == "1" for row in reference),
        "stop_counts": {name: sum(row["stop"] == name for row in reference)
                        for name in sorted({row["stop"] for row in reference})},
        "by_rtol": {
            tol: {
                "rows": sum(row["rtol"] == tol for row in reference),
                "reference_usable": sum(row["rtol"] == tol and row["reference_usable"] == "1"
                                        for row in reference),
                "violations": sum(row["rtol"] == tol and row["violation"] == "1"
                                  for row in reference),
            }
            for tol in sorted({row["rtol"] for row in reference}, key=float)
        },
        "interpretation": "existing value-only adaptive path; Hermite shadow is opt-in",
    }

jac_path = ROOT / "adaptive_jacobian_reference.csv"
if jac_path.exists():
    with jac_path.open(newline="") as stream:
        jac_rows = list(csv.DictReader(stream))
    summary["jacobian_reference_control"] = {
        "raw_rows": len(jac_rows),
        "stop_counts": {name: sum(row["stop"] == name for row in jac_rows)
                        for name in sorted({row["stop"] for row in jac_rows})},
        "scope": "existing LD/Jacobian route; Hermite request predicate excludes gradients and u != 0",
    }

checks = {
    "primary_replicates_present": len(primary_runs) == 2,
    "primary_rows_per_tolerance_per_replicate": all(
        sum(row["tol"] == tol for row in run) == 3608
        for run in primary_runs.values() for tol in ("0.001", "0.0001")),
    "all_primary_value_converged": all(
        row[f"{mode}_ok"] == "1" and row[f"shadow_{mode}_ok"] == "1"
        for row in trajectory for mode in ("cold", "warm")) and all(
        row["radial_ok"] == "1" and row["shadow_radial_ok"] == "1"
        for row in trajectory),
    "zero_primary_mu_status_node_mismatches": all(
        tol_summary[mode][key] == 0
        for tol_summary in summary["cold_warm"].values()
        for mode in ("cold", "warm")
        for key in ("mu_mismatches", "status_mismatches", "node_count_mismatches")),
    "zero_radial_only_mu_status_node_mismatches": all(
        tol_summary["radial_only"][key] == 0
        for tol_summary in summary["cold_warm"].values()
        for key in ("mu_mismatches", "status_mismatches", "node_count_mismatches")),
    "all_primary_wall_times_finite_nonnegative": all(
        finite(row, key) and float(row[key]) >= 0
        for row in trajectory for key in
        ("cold_ms", "shadow_cold_ms", "warm_ms", "shadow_warm_ms",
         "radial_ms", "shadow_radial_ms")),
    "reference_observed_violations_zero": summary.get("value_reference", {}).get("observed_violations") == 0,
    "no_false_panel_model_passes": summary["panel_audit"]["false_local_passes"] == 0,
    "no_unresolved_panel_model_passes":
        summary["panel_audit"]["unresolved_panels_that_would_pass_model_budget_if_detail_guard_were_removed"] == 0,
}
summary["validation_checks"] = checks
if not all(checks.values()):
    raise SystemExit("summary validation failed: " + repr(checks))

(ROOT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
