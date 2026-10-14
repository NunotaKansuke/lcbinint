#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
raw = pd.read_csv(ROOT / "corpus.tsv", sep="\t")
assert len(raw) == 3520
assert raw.groupby(["h_split_min_level"]).size().to_dict() == {0: 880, 4: 880, 5: 880, 8: 880}
keys = ["rep", "row", "name", "u", "value_rtol", "path"]
base = raw[raw.h_split_min_level == 0].set_index(keys).sort_index()

def qstats(a):
    a = np.asarray(a, dtype=float)
    return {k: float(np.quantile(a, p)) for k, p in
            (("p50", .50), ("p90", .90), ("p95", .95), ("p99", .99), ("max", 1.0))}

whole = {}
for split in (4, 5, 8):
    cand = raw[raw.h_split_min_level == split].set_index(keys).sort_index()
    assert cand.index.equals(base.index)
    arm = {}
    for (rtol, path), g in cand.groupby(["value_rtol", "path"]):
        b = base.loc[g.index]
        ratio = g.ms.to_numpy() / b.ms.to_numpy()
        quality = {}
        reasons = {}
        for j in range(5):
            tr = (b[f"quality{j}"].astype(str) + ">" +
                  g[f"quality{j}"].astype(str)).value_counts().to_dict()
            quality[str(j)] = {k: int(v) for k, v in tr.items()}
            rr = (b[f"reason{j}"].astype(str) + ">" +
                  g[f"reason{j}"].astype(str)).value_counts().to_dict()
            reasons[str(j)] = {k: int(v) for k, v in rr.items()}
        arm[f"{rtol:g}/{path}"] = {
            "pairs": int(len(g)),
            "baseline_ms": qstats(b.ms),
            "candidate_ms": qstats(g.ms),
            "paired_time_ratio": qstats(ratio),
            "sum_time_change_percent": float(100 * (g.ms.sum() / b.ms.sum() - 1)),
            "node_sum_change_percent": float(100 * (g.nodes.sum() / b.nodes.sum() - 1)),
            "baseline_node_mean": float(b.nodes.mean()),
            "candidate_node_mean": float(g.nodes.mean()),
            "mu_bitwise_mismatches": int(np.count_nonzero(b.mu.to_numpy() != g.mu.to_numpy())),
            "value_stop_mismatches": int(np.count_nonzero(b.value_stop.to_numpy() != g.value_stop.to_numpy())),
            "gradient_stop_mismatches": int(np.count_nonzero(b.gradient_stop.to_numpy() != g.gradient_stop.to_numpy())),
            "value_convergence_mismatches": int(np.count_nonzero(
                b.value_converged.to_numpy() != g.value_converged.to_numpy())),
            "new_value_nonconvergence": int(np.count_nonzero(
                b.value_converged.astype(bool).to_numpy() & ~g.value_converged.astype(bool).to_numpy())),
            "baseline_nonconvergence_recovered": int(np.count_nonzero(
                ~b.value_converged.astype(bool).to_numpy() & g.value_converged.astype(bool).to_numpy())),
            "baseline_value_converged": int(b.value_converged.sum()),
            "candidate_value_converged": int(g.value_converged.sum()),
            "baseline_value_stop_counts": {str(k): int(v) for k, v in b.value_stop.value_counts().items()},
            "candidate_value_stop_counts": {str(k): int(v) for k, v in g.value_stop.value_counts().items()},
            "gradient_quality_transitions": quality,
            "gradient_reason_transitions": reasons,
            "tolerance_met_to_uncertified": {
                str(j): int(((b[f"quality{j}"] == "ToleranceMet") &
                             (g[f"quality{j}"] == "FiniteUncertified")).sum())
                for j in range(5)},
            "new_invalid_components": {
                str(j): int(((b[f"quality{j}"] != "Invalid") &
                             (g[f"quality{j}"] == "Invalid")).sum())
                for j in range(5)},
        }
    whole[str(split)] = arm

hard = {}
rows = {}
for line in (ROOT / "panel_audit.tsv").read_text().splitlines():
    if not line or line.startswith("TYPE "):
        continue
    x = line.split()
    typ = x[0]
    row, name, u = int(x[1]), x[2], float(x[3])
    key = f"{row}:{name}:u={u:g}"
    hard.setdefault(key, {"row": row, "name": name, "u": u,
                          "runs": {}, "panel_contributions": {},
                          "cancellation": {}, "finite_difference": []})
    h = hard[key]
    if typ == "RUN":
        mode, split = int(x[4]), int(x[5])
        run_key = str(mode)
        gradients = []
        for j in range(5):
            k = 13 + 4*j
            gradients.append({"value": float(x[k]), "estimated_error": float(x[k+1]),
                              "quality": x[k+2], "reason": x[k+3]})
        ledger_start = 33
        ledgers = []
        for j in range(6):
            k = ledger_start + 5*j
            ledgers.append({"radial": float(x[k]), "inner": float(x[k+1]),
                            "geometry": float(x[k+2]), "event": float(x[k+3]),
                            "roundoff": float(x[k+4])})
        ledger_names = ["mu", "X", "Y", "rho", "q", "a"]
        h["runs"][run_key] = {"split_min_level": split,
            "value_converged": bool(int(x[6])), "value_stop": x[7],
            "mu": float(x[9]), "nodes": int(x[10]), "splits": int(x[11]),
            "evaluations": int(x[12]),
            "gradients": [{**g, "ledger": ledgers[j+1]}
                          for j, g in enumerate(gradients)],
            "ledger_by_quantity": dict(zip(ledger_names, ledgers))}
    elif typ == "CANCEL":
        mode, j = int(x[4]), int(x[5])
        h["cancellation"].setdefault(str(mode), {})[str(j)] = {
            "signed_sum": float(x[6]), "positive_sum": float(x[7]),
            "negative_sum": float(x[8]), "sum_abs": float(x[9]),
            "cancellation_ratio": float(x[10]),
            "reported_gradient": float(x[11]), "sum_minus_reported": float(x[12])}
    elif typ == "PANEL":
        mode, panel, cell, depth, level, j = map(int, (x[4], x[5], x[6], x[8], x[9], x[14]))
        panels = h["panel_contributions"].setdefault(str(mode), {}).setdefault(str(j), [])
        panels.append({"panel": panel, "cell": cell, "depth": depth, "level": level,
            "contribution": float(x[15]), "detail": float(x[16]),
            "previous_detail": float(x[17]), "detail_decay": bool(int(x[18])),
            "radial_candidate": float(x[19]), "inner": float(x[20]),
            "geometry": float(x[21]), "event": float(x[22]),
            "roundoff": float(x[23]), "total_error": float(x[24]),
            "global_tolerance": float(x[25]),
            "eligible_for_local_h_split": bool(int(x[26])),
            "gradient_resolved": bool(int(x[27]))})
    elif typ == "FD":
        h["finite_difference"].append({"parameter": int(x[4]), "radial_order": int(x[5]),
            "step": float(x[6]), "plus_topology_status": int(x[7]),
            "minus_topology_status": int(x[8]), "plus_value": float(x[9]),
            "minus_value": float(x[10]), "gradient": float(x[11])})

for key, case in hard.items():
    usable = [r for r in case["finite_difference"] if np.isfinite(r["gradient"])]
    by_param = {}
    for r in usable:
        by_param.setdefault(str(r["parameter"]), []).append(r)
    case["fd_available_count"] = len(usable)
    case["fd_step_study"] = {}
    for param, rec in by_param.items():
        by_order = {n: [z for z in rec if z["radial_order"] == n]
                    for n in (128, 256)}
        preferred_order = 256 if by_order[256] else 128
        preferred = min(by_order[preferred_order], key=lambda z: z["step"])
        selected_parameter = int(param)
        run_errors = {}
        for mode, run in case["runs"].items():
            g = run["gradients"][selected_parameter]["value"]
            run_errors[mode] = {"reference": preferred["gradient"],
                                "abs_error": abs(g-preferred["gradient"]),
                                "relative_error": abs(g-preferred["gradient"]) /
                                                  max(abs(preferred["gradient"]), 1e-300)}
        case["fd_step_study"][param] = {"records": rec,
                "preferred_reference": preferred,
                "reference_scope_status": ("cross-order and step sweep available"
                    if by_order[128] and by_order[256] else
                    "limited: no valid order-256 sweep; diagnostic only"),
                "analytic_gradient_observed_error": run_errors,
                "order128_range": float(max(z["gradient"] for z in rec if z["radial_order"] == 128) -
                                        min(z["gradient"] for z in rec if z["radial_order"] == 128))
                                if any(z["radial_order"] == 128 for z in rec) else None,
                "order256_range": float(max(z["gradient"] for z in rec if z["radial_order"] == 256) -
                                        min(z["gradient"] for z in rec if z["radial_order"] == 256))
                                if any(z["radial_order"] == 256 for z in rec) else None}
    case["local_h_split_candidate_panel_counts"] = {
        mode: {j: sum(int(p["eligible_for_local_h_split"]) for p in plist)
               for j, plist in comps.items()}
        for mode, comps in case["panel_contributions"].items()}

summary = {
    "head": "64bb1e3dd88d66be6a98eb9d77c20da6c45733e9",
    "corpus": {"rows_per_arm": 880, "source_inputs": 110,
               "value_tolerances": [1e-3, 1e-4],
               "path_definitions": {"cold": "epoch_adaptive full D14/topology plus gradient adaptive integration",
                                    "warm": "prepared D14 root warm-start; same-point unmeasured warm-up, topology reuse disabled"},
               "value_first_gradient_round_budget": 4,
               "gradient_tolerance": "existing defaults (atol 1e-6, rtol 1e-3)",
               "primary_candidate": "gradient_local_refinement=true, gradient_split_min_level=4",
               "controls": {"0": "p-refinement baseline",
                            "5": "h-split only after level 5",
                            "8": "estimator/priority control; no h split reached under 4-round budget"}},
    "whole_epoch": whole,
    "hard_case_settings": {"value_rtol": 1e-3,
                           "gradient_policy": "ValueFirst",
                           "gradient_atol": 1e-4,
                           "gradient_rtol": 1e-2,
                           "gradient_round_budget": 64,
                           "gradient_node_budget": 8192,
                           "arms": {"0": "p-only, max level 8",
                                    "1": "p to level 4 then conditional h-split",
                                    "2": "p to level 5 then conditional h-split"}},
    "hard_cases": hard,
    "independent_reference_scope": "Gauss-Legendre radial 128/256 with direct angular Chebyshev value integral; fresh topology for each parameter perturbation; FD step scaled by rho. Only explicitly selected components have this reference. Some rand035 perturbations fail the independent reference topology/quadrature and remain unavailable.",
    "baseline_nonconverged_inputs_by_value_tolerance": {},
}

for tol, group in base.reset_index().groupby("value_rtol"):
    failed = group.loc[~group.value_converged.astype(bool), ["row", "name", "u", "value_stop"]]
    summary["baseline_nonconverged_inputs_by_value_tolerance"][f"{tol:g}"] = [
        {"row": int(r.row), "name": r.name, "u": float(r.u),
         "stop": str(r.value_stop)}
        for r in failed.drop_duplicates(["row", "name", "u"]).itertuples(index=False)]

# Acceptance-oriented invariants. Status downgrades are recorded, never hidden.
checks = {"all_arms_preserve_baseline_value_convergence_status": True,
          "corpus_row_count": len(raw),
          "paired_mu_bitwise_equal_all_arms": True,
          "paired_value_stop_equal_all_arms": True,
          "paired_gradient_stop_equal_all_arms": True,
          "no_new_invalid_gradient_components_all_arms": True,
          "h_split4_quality_downgrades_present": False,
          "h_split5_quality_downgrades_present": False}
for split, out in whole.items():
    for group in out.values():
        checks["paired_mu_bitwise_equal_all_arms"] &= group["mu_bitwise_mismatches"] == 0
        checks["paired_value_stop_equal_all_arms"] &= group["value_stop_mismatches"] == 0
        checks["paired_gradient_stop_equal_all_arms"] &= group["gradient_stop_mismatches"] == 0
        checks["no_new_invalid_gradient_components_all_arms"] &= not any(
            n > 0 for n in group["new_invalid_components"].values())
        checks["all_arms_preserve_baseline_value_convergence_status"] &= (
            group["value_convergence_mismatches"] == 0)
        if split in ("4", "5"):
            checks[f"h_split{split}_quality_downgrades_present"] |= any(
                n > 0 for n in group["tolerance_met_to_uncertified"].values())

value_ref = pd.read_csv(ROOT / "value_reference_check.tsv")
grad_ref = pd.read_csv(ROOT / "gradient_reference_corpus.tsv")
grad_fd = pd.to_numeric(grad_ref["finite_difference"], errors="coerce").to_numpy()
unit_text = (ROOT / "adaptive_unit.log").read_text().strip()
ctest_text = (ROOT / "ctest.log").read_text()
summary["verification"] = {
    "adaptive_unit": unit_text,
    "ctest_selected_tests": {"passed": ctest_text.count("...   Passed"),
                             "failed": ctest_text.count("...***Failed"),
                             "all_passed": "100% tests passed" in ctest_text},
    "value_reference_corpus": {
        "rows": int(len(value_ref)),
        "usable": int((value_ref.reference_usable == 1).sum()),
        "unusable": int((value_ref.reference_usable != 1).sum()),
        "observed_violations": int((value_ref.violation == 1).sum())},
    "existing_independent_gradient_reference": {
        "rows": int(len(grad_ref)),
        "finite_difference_rows": int(np.isfinite(grad_fd).sum()),
        "nonfinite_rows": int((~np.isfinite(grad_fd)).sum()),
        "stop_counts": {str(k): int(v) for k, v in
                        grad_ref.stop.value_counts().items()}}
}
summary["validation_checks"] = checks
def json_finite(value):
    """Encode unavailable/nonfinite diagnostic values as JSON null."""
    if isinstance(value, dict):
        return {key: json_finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_finite(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value

summary = json_finite(summary)
(ROOT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
print(json.dumps({"validation_checks": checks,
                  "whole_epoch": {s: {k: {"sum_time_change_percent": v["sum_time_change_percent"],
                                          "node_sum_change_percent": v["node_sum_change_percent"],
                                          "p50_baseline_ms": v["baseline_ms"]["p50"],
                                          "p50_candidate_ms": v["candidate_ms"]["p50"],
                                          "quality_downgrades": v["tolerance_met_to_uncertified"]}
                                    for k, v in bypath.items()} for s, bypath in whole.items()}},
                 indent=2))
