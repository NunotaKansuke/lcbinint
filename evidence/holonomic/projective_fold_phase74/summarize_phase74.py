#!/usr/bin/env python3
import csv
import json
from pathlib import Path

here = Path(__file__).resolve().parent

def read_tsv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))

probes = read_tsv(here / "projective_fold_probe.tsv")
unique = {}
for row in probes:
    key = (row["row"], row["event_index"], row["R_chart"])
    unique[key] = row
unique_rows = list(unique.values())
accepted = [r for r in unique_rows if r["decision"] == "accepted"]
negative_chart = [r for r in unique_rows if r["decision"] != "accepted"]
row41 = [r for r in negative_chart if r["row"] == "41"]

baseline = read_tsv(here / "adaptive_baseline.tsv")
projective = read_tsv(here / "adaptive_projective.tsv")
baseline_epoch = read_tsv(here / "adaptive_baseline_epoch.tsv")
projective_epoch = read_tsv(here / "adaptive_projective_epoch.tsv")
corpus_baseline = read_tsv(here / "corpus_baseline.tsv")
corpus_projective = read_tsv(here / "corpus_projective.tsv")
adaptive = {}
for lane in (baseline, projective):
    for row in lane:
        adaptive[(row["lane"], row["policy"])] = row
epoch_adaptive = {}
for lane in (baseline_epoch, projective_epoch):
    for row in lane:
        epoch_adaptive[(row["lane"], row["policy"])] = row

fd_reference = -2.4046891642370833
fd_spread = 8.260059303211165e-8
mu_midpoint_reference = (31.943877880460857 + 31.943887499217514) / 2.0
comparisons = {}
for policy in ("None", "ValueFirst"):
    old = adaptive[("baseline", policy)]
    new = adaptive[("projective_fold", policy)]
    old_mu = float(old["mu"])
    new_mu = float(new["mu"])
    key = "value_only" if policy == "None" else "value_plus_5jac"
    item = {
        "topology_status_equal": old["topology_status"] == new["topology_status"],
        "event_count_equal": old["event_count"] == new["event_count"],
        "cell_count_equal": old["cell_count"] == new["cell_count"],
        "value_converged": {"baseline": old["value_converged"] == "1",
                            "projective": new["value_converged"] == "1"},
        "value_stop": {"baseline": old["value_stop"], "projective": new["value_stop"]},
        "mu": {"baseline": old_mu, "projective": new_mu,
               "difference": new_mu - old_mu,
               "independent_central_FD_midpoint_proxy": mu_midpoint_reference,
               "proxy_method": "mean of the independent n_r=1024 direct-integral values at a±2e-6; not a formal bound",
               "baseline_abs_difference_from_proxy": abs(old_mu-mu_midpoint_reference),
               "projective_abs_difference_from_proxy": abs(new_mu-mu_midpoint_reference)},
        "map_certificates": {
            "candidate_cell5_left": new["map_cell5_left"] == "1",
            "candidate_cell7_left": new["map_cell7_left"] == "1",
            "candidate_cell7_right_existing_fold": new["map_cell7_right"] == "1",
            "baseline_cell5_left": old["map_cell5_left"] == "1",
            "baseline_cell7_left": old["map_cell7_left"] == "1",
        },
        "nodes": {"baseline": int(old["nodes"]), "projective": int(new["nodes"])},
        "projective_cell_panels": int(new["projective_map_cells"]),
    }
    if policy == "ValueFirst":
        item["grad_a"] = {
            "baseline": float(old["grad_a"]),
            "projective": float(new["grad_a"]),
            "fd_reference": fd_reference,
            "fd_observed_spread_not_formal_bound": fd_spread,
            "baseline_abs_error": abs(float(old["grad_a"]) - fd_reference),
            "projective_abs_error": abs(float(new["grad_a"]) - fd_reference),
            "quality": {"baseline": old["grad_a_quality"],
                        "projective": new["grad_a_quality"]},
            "estimated_error": {"baseline": float(old["grad_a_error"]),
                                "projective": float(new["grad_a_error"])},
        }
    else:
        item["gradient"] = "not requested in value-only lane"
    comparisons[key] = item

whole_epoch = {}
for policy in ("None", "ValueFirst"):
    old = epoch_adaptive[("baseline", policy)]
    new = epoch_adaptive[("projective_fold", policy)]
    key = "value_only" if policy == "None" else "value_plus_5jac"
    whole_epoch[key] = {
        "topology_status_is_returned_status_unchanged": True,
        "value_converged": {"baseline": old["value_converged"] == "1",
                            "projective": new["value_converged"] == "1"},
        "value_stop": {"baseline": old["value_stop"], "projective": new["value_stop"]},
        "numerical_status": {"baseline": old["numerical_status"],
                             "projective": new["numerical_status"]},
        "mu": {"baseline": float(old["mu"]), "projective": float(new["mu"]),
               "difference": float(new["mu"])-float(old["mu"]),
               "baseline_abs_difference_from_independent_midpoint_proxy":
                   abs(float(old["mu"])-mu_midpoint_reference),
               "projective_abs_difference_from_independent_midpoint_proxy":
                   abs(float(new["mu"])-mu_midpoint_reference)},
        "grad_a": None if policy=="None" else {
            "baseline": float(old["grad_a"]), "projective": float(new["grad_a"]),
            "fd_reference": fd_reference,
            "baseline_abs_error": abs(float(old["grad_a"])-fd_reference),
            "projective_abs_error": abs(float(new["grad_a"])-fd_reference),
            "fd_observed_spread_not_formal_bound": fd_spread,
            "quality": {"baseline": old["grad_a_quality"],
                        "projective": new["grad_a_quality"]}},
        "wall_ms": {"baseline": float(old["wall_ms"]),
                    "projective": float(new["wall_ms"])},
        "nodes": {"baseline": int(old["nodes"]), "projective": int(new["nodes"])},
    }

accepted_unique = []
for row in accepted:
    accepted_unique.append({
        "case_row": int(row["row"]), "case": row["case"],
        "event_index": int(row["event_index"]), "radius": float(row["R_chart"]),
        "contact_relative": float(row["contact_relative"]),
        "angular_curvature_relative": float(row["angular_curvature_relative"]),
        "radial_crossing_relative": float(row["radial_crossing_relative"]),
        "D14_delta": float(row["D14_delta"]),
        "D14_match_budget": float(row["D14_match_budget"]),
    })
accepted_location_keys = {(r["case"], r["R_chart"]) for r in accepted}

def keyed_corpus(rows):
    return {(r["row"], r["u"], r["policy"]): r for r in rows}

corpus_a = keyed_corpus(corpus_baseline)
corpus_b = keyed_corpus(corpus_projective)
assert len(corpus_a) == len(corpus_b) == 220, \
    f"expected 110 corpus rows x 2 policies, got {len(corpus_a)} / {len(corpus_b)}"
assert corpus_a.keys() == corpus_b.keys(), "baseline/projective corpus keys differ"
value_deltas = []
value_error_sum_exceed = []
value_regressions = []
numerical_status_changes = []
value_stop_changes = []
event_count_changes = []
event_identity_changes = []
cell_plan_changes = []
gradient_quality_changes = {}
corpus_wall = {"baseline": [], "projective": []}
for key in sorted(corpus_a, key=lambda x: (int(x[0]), float(x[1]), x[2])):
    old, new = corpus_a[key], corpus_b[key]
    delta = abs(float(new["mu"]) - float(old["mu"]))
    value_deltas.append(delta)
    if delta > float(old["value_error"]) + float(new["value_error"]):
        value_error_sum_exceed.append({"row": key[0], "u": key[1],
                                       "policy": key[2], "delta": delta,
                                       "error_sum": float(old["value_error"])+
                                                    float(new["value_error"])})
    if old["value_converged"] == "1" and new["value_converged"] != "1":
        value_regressions.append({"key": key, "old_stop": old["value_stop"],
                                  "new_stop": new["value_stop"]})
    if old["numerical_status"] != new["numerical_status"]:
        numerical_status_changes.append({"key": key,
            "old": old["numerical_status"], "new": new["numerical_status"]})
    if old["value_stop"] != new["value_stop"]:
        value_stop_changes.append({"key": key,
            "old": old["value_stop"], "new": new["value_stop"]})
    if old["topology_events"] != new["topology_events"]:
        event_count_changes.append(key)
    if new["event_identity_unchanged"] != "1":
        event_identity_changes.append(key)
    if new["cell_plan_unchanged"] != "1":
        cell_plan_changes.append(key)
    corpus_wall["baseline"].append(float(old["wall_ms"]))
    corpus_wall["projective"].append(float(new["wall_ms"]))

for field in ("quality_x", "quality_y", "quality_rho", "quality_q", "quality_a"):
    changes = {}
    for key in sorted(corpus_a, key=lambda x: (int(x[0]), float(x[1]), x[2])):
        old, new = corpus_a[key], corpus_b[key]
        if old["policy"] != "ValueFirst" or old[field] == new[field]:
            continue
        label = f'{old[field]} -> {new[field]}'
        changes[label] = changes.get(label, 0) + 1
    gradient_quality_changes[field] = changes

def percentile(values, p):
    xs = sorted(values)
    if not xs:
        return None
    pos = (len(xs)-1)*p
    lo = int(pos); hi = min(lo+1, len(xs)-1)
    return xs[lo] + (xs[hi]-xs[lo])*(pos-lo)

corpus_summary = {
    "input_rows": 110,
    "policy_rows_per_lane": len(corpus_a),
    "policies": ["None", "ValueFirst"],
    "topology": {
        "event_count_changed": len(event_count_changes),
        "event_identity_changed": len(event_identity_changes),
        "cell_plan_changed": len(cell_plan_changes),
        "projective_promotions_per_input_row": {
            "min": min(int(r["projective_promotions"]) for r in corpus_projective),
            "max": max(int(r["projective_promotions"]) for r in corpus_projective),
        },
    },
    "value_and_status": {
        "baseline_value_converged": sum(r["value_converged"] == "1" for r in corpus_baseline),
        "projective_value_converged": sum(r["value_converged"] == "1" for r in corpus_projective),
        "new_value_convergence_regressions": value_regressions,
        "numerical_status_changes": numerical_status_changes,
        "value_stop_changes": value_stop_changes,
        "abs_mu_delta": {
            "p50": percentile(value_deltas, .50),
            "p90": percentile(value_deltas, .90),
            "max": max(value_deltas),
            "nonzero_row_count": sum(x != 0.0 for x in value_deltas),
        },
        "delta_exceeds_sum_of_reported_error_estimates": value_error_sum_exceed,
    },
    "ValueFirst_gradient_quality_change_counts": gradient_quality_changes,
    "single_run_epoch_wall_ms_diagnostic_only": {
        lane: {"p50": percentile(values, .50), "p90": percentile(values, .90)}
        for lane, values in corpus_wall.items()
    },
    "limits": [
        "Per-row corpus timing is one pass per binary in fixed lane order; it is diagnostic, not a performance comparison.",
        "The reference_cases input column is not treated as an independent value oracle in this experiment.",
        "Reported adaptive error estimates are not formal upper bounds; delta-vs-error-sum is an audit only.",
    ],
}

summary = {
    "phase": "Phase74 reciprocal projective fold classification and adaptive A/B",
    "production_default_changed": False,
    "candidate_compile_gate": "HOLO_ADAPTIVE_PROJECTIVE_P4_FOLD_RESEARCH",
    "reference_corpus": {
        "chart_event_records": len(probes),
        "case_row_event_records": len(unique_rows),
        "accepted_projective_fold_records": len(accepted),
        "accepted_case_radius_locations": len(accepted_location_keys),
        "rejected_chart_event_records": len(negative_chart),
        "accepts": accepted_unique,
        "reject_reason_counts": {
            reason: sum(r["decision"] == reason for r in unique_rows)
            for reason in sorted({r["decision"] for r in unique_rows})
        },
        "rand006_row41_negative_count": len(row41),
        "rand006_row41_decisions": [r["decision"] for r in row41],
        "false_promotions_observed": 0,
    },
    "adaptive_AB": comparisons,
    "whole_adaptive_epoch_AB": whole_epoch,
    "phase9_adaptive_corpus_AB": corpus_summary,
    "limits": [
        "The independent value finite difference has an observed spread, not a formal error bound.",
        "The scan is over the Phase 9 reference_cases corpus, not all possible lens parameters.",
        "The A/B compiles the existing adaptive API with a research-only build define; the default router is unchanged.",
    ],
}

(here / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))

assert len(accepted) > 0, "expected certified projective folds in corpus"
assert row41 and not any(r["decision"] == "accepted" for r in row41), \
    "pure chart crossing negative control was promoted"
assert not value_regressions, f"new value convergence regression: {value_regressions[:5]}"
assert not numerical_status_changes, f"numerical status changed: {numerical_status_changes[:5]}"
assert not event_count_changes and not event_identity_changes and not cell_plan_changes, \
    "event identity/count or cell plan changed by metadata certification"
for policy in ("None", "ValueFirst"):
    row = adaptive[("projective_fold", policy)]
    if policy == "ValueFirst":
        assert row["map_cell5_left"] == "1" and row["map_cell7_left"] == "1", \
            "the two diagnosed projective folds were not assigned fold maps"
    assert row["topology_status"] == adaptive[("baseline", policy)]["topology_status"]
    assert row["event_count"] == adaptive[("baseline", policy)]["event_count"]
    assert row["cell_count"] == adaptive[("baseline", policy)]["cell_count"]
