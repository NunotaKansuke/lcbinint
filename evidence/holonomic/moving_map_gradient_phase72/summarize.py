#!/usr/bin/env python3
"""Summarize the diagnostic-only Phase72 moving-map gradient shadow."""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read_tsv(name):
    with (HERE / name).open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def f(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return math.nan


def finite(x):
    return math.isfinite(x)


case_rows = read_tsv("cases.tsv")
cell_rows = read_tsv("cells.tsv")
total_rows = read_tsv("totals.tsv")
node_rows = read_tsv("nodes.tsv")
event_rows = read_tsv("events.tsv")
all_event_rows = read_tsv("all_events.tsv")
pointwise_rows = read_tsv("pointwise_fd.tsv")

refs = {}
rand_summary_path = HERE / "rand035_reference/summary.json"
rand_summary = json.loads(rand_summary_path.read_text())
for c in rand_summary["comparisons"]:
    refs[(c["case"], float(c["u"]), c["axis"])] = {
        "value": c["reference"]["value_gradient"],
        "observed_spread": c["reference"]["observed_stability_spread"],
        "method": c["reference"]["estimator"],
        "source": "rand035_reference/summary.json",
    }

def near_step(x, target):
    return abs(x["h"]-target) <= 1e-12*target

# Independent higher-order reference: each perturbation gets fresh topology,
# then the direct phi/arc value integral is evaluated with n=512/1024 GL radial
# rules. Use the smallest step and record the radial/step stability spread.
refined_fd = read_tsv("reference_fd.tsv")
refined_caustic = next(r for r in refined_fd if int(r["n"]) == 1024 and
                       near_step({"h": float(r["h"])}, 2.0e-6))
caustic_neighborhood = [r for r in refined_fd if int(r["n"]) in (512, 1024) and
                        any(near_step({"h": float(r["h"])}, target)
                            for target in (2.0e-6, 6.0e-6, 2.0e-5))]
refs[("caustic-cross", 0.0, "a")] = {
    "value": float(refined_caustic["fd"]),
    "observed_spread": max(abs(float(r["fd"])-float(refined_caustic["fd"]))
                            for r in caustic_neighborhood),
    "method": "fresh topology + direct arc flux, Gauss-Legendre radial value FD",
    "source": "evidence/holonomic/moving_map_gradient_phase72/reference_fd.tsv",
    "resolution": {"radial_order": 1024, "selected_step": float(refined_caustic["h"])},
}

by_key = defaultdict(list)
for r in total_rows:
    key = (int(r["row"]), r["case"], float(r["u"]), r["parameter"])
    by_key[key].append(r)
for values in by_key.values():
    values.sort(key=lambda r: int(r["level"]))

case_meta = {(int(r["row"]), r["case"], float(r["u"]), r["parameter"]): r
             for r in case_rows}
cells_by_key = defaultdict(list)
for r in cell_rows:
    key = (int(r["row"]), r["case"], float(r["u"]), r["parameter"])
    cells_by_key[key].append(r)

nodes_by_key = defaultdict(list)
for r in node_rows:
    key = (int(r["row"]), r["case"], float(r["u"]), r["parameter"])
    nodes_by_key[key].append(r)

events_by_key = defaultdict(list)
for r in event_rows:
    key = (int(r["row"]), r["case"], float(r["u"]), r["parameter"])
    events_by_key[key].append(r)

all_events_by_key = defaultdict(list)
for r in all_event_rows:
    key = (int(r["row"]), r["case"], float(r["u"]), r["parameter"])
    all_events_by_key[key].append(r)

pointwise_by_key = defaultdict(list)
for r in pointwise_rows:
    key = (int(r["row"]), r["case"], float(r["u"]), r["parameter"])
    pointwise_by_key[key].append(r)

results = []
for key, series in sorted(by_key.items()):
    row, name, u, par = key
    ref = refs[(name, u, par)]
    level_data = []
    for r in series:
        level = int(r["level"])
        cells_at = [x for x in cells_by_key[key] if int(x["level"]) == level]
        old_detail = math.fsum(f(x, "old_detail") for x in cells_at)
        moving_detail = math.fsum(f(x, "moving_detail") for x in cells_at)
        old = f(r, "old_gradient_valid_only")
        moving = f(r, "moving_gradient_valid_only")
        complete = r["all_cells_complete"] == "1"
        level_data.append({
            "level": level, "nodes_per_cell": int(2**level-1),
            "value_integral": f(r, "mu_q"), "old_integral": old,
            "moving_integral": moving, "moving_minus_old": moving-old,
            "complete_cells": int(r["complete_cells"]),
            "all_cells_complete": r["all_cells_complete"] == "1",
            "valid_node_count": int(r["valid_node_count"]),
            "old_abs_cell_contribution": f(r, "old_abs_contribution"),
            "moving_abs_cell_contribution": f(r, "moving_abs_contribution"),
            "old_cancellation_ratio": f(r, "old_cancellation"),
            "moving_cancellation_ratio": f(r, "moving_cancellation"),
            "old_weighted_detail_l2_sum": old_detail,
            "moving_weighted_detail_l2_sum": moving_detail,
            "old_abs_error_vs_independent_fd": abs(old-ref["value"]),
            "moving_abs_error_vs_independent_fd": abs(moving-ref["value"]),
            "whole_domain_old_abs_error_vs_independent_fd":
                abs(old-ref["value"]) if complete else None,
            "whole_domain_moving_abs_error_vs_independent_fd":
                abs(moving-ref["value"]) if complete else None,
            "complete_gradient_series": r["all_cells_complete"] == "1",
            "all_cells_have_finite_nodes": r["all_cells_complete"] == "1",
        })

    top_nodes = [n for n in nodes_by_key[key] if int(n["level"]) == 8]
    fold_cells = {int(c["cell"]): c for c in cells_by_key[key]
                  if int(c["level"]) == 8}
    endpoint_stats = []
    for cell_id, c in sorted(fold_cells.items()):
        if c["kind"] != "arcs" or (c["left_fold"] != "1" and c["right_fold"] != "1"):
            continue
        selected = [n for n in top_nodes if int(n["cell"]) == cell_id and
                    int(n["k"]) in (1, 255)]
        for n in selected:
            endpoint_stats.append({
                "cell": cell_id, "side": "near_x_plus" if int(n["k"]) == 1 else "near_x_minus",
                "x": f(n, "x"), "R": f(n, "R"),
                "old_integrand": f(n, "old"),
                "moving_integrand": f(n, "moving"),
                "fixed_term": f(n, "old"), "radial_motion_term": f(n, "JFRRp"),
                "jacobian_motion_term": f(n, "FJp"),
                "sample_ok": n["sample_ok"] == "1",
                "sample_reject_reason": n["reject_reason"],
                "sample_reject_reason_code": int(n["reject_code"]),
                "radial_jet_ok": n["fr_ok"] == "1",
                "fold_is_left": c["left_fold"] == "1",
                "fold_is_right": c["right_fold"] == "1",
            })

    method_counts = defaultdict(int)
    reject_counts = defaultdict(int)
    failed_nodes = 0
    radial_abs, fixed_abs, jac_abs = [], [], []
    level8_stride = 1
    for n in top_nodes:
        if n["ok"] != "1":
            failed_nodes += 1
            reject_counts[n["reject_reason"]] += 1
            continue
        method_counts[n["fr_method"]] += 1
        k = int(n["k"])
        # Fejer-II level 8 weights are the common finest-node weights.
        # The raw-node absolute terms are reported separately below; retain their
        # unweighted maxima here, while weighted sums are available in totals.
        if k <= 0 or level8_stride != 1:
            pass
        fixed_abs.append(abs(f(n, "old")))
        radial_abs.append(abs(f(n, "JFRRp")))
        jac_abs.append(abs(f(n, "FJp")))

    base_gradient_ledger_l1 = {}
    for component in ("inner", "geometry", "roundoff"):
        base_gradient_ledger_l1[component] = math.fsum(
            abs(f(n, "fejer_weight") * f(n, "J") * f(n, component))
            for n in top_nodes if n["ok"] == "1")

    emeta = case_meta[key]
    case_events = events_by_key[key]
    case_all_events = all_events_by_key[key]
    pointwise_checks = pointwise_by_key[key]
    maxP = max((abs(f(e, "P")) for e in case_events), default=math.nan)
    maxPt = max((abs(f(e, "Pt")) for e in case_events), default=math.nan)
    result = {
        "row": row, "case": name, "u": u, "parameter": par,
        "input": {k: (float(emeta[k]) if k in ("xs", "ys", "rho", "q", "a") else emeta[k])
                  for k in ("barycentric", "xs", "ys", "rho", "q", "a")},
        "topology": {"status": int(emeta["topology_status"]), "r_max": f(emeta, "r_max"),
                     "event_count": int(emeta["events"]),
                     "physical_event_count": int(emeta["physical_events"]),
                     "cell_count": int(emeta["topology_cells"]),
                     "adaptive_cell_count": int(emeta["adaptive_cells"]),
                     "endpoint_sensitivity_map_valid": emeta["map_ok"] == "1"},
        "independent_fd": ref,
        "endpoint_fold_residual_max_abs": {"P": maxP, "Pt": maxPt},
        "event_sensitivities": [{k: (float(v) if k not in ("event_index", "tier", "ordinary", "valid") else
                                      (int(v) if k in ("event_index", "tier") else v == "1"))
                                  for k,v in e.items() if k not in ("row", "case", "u", "parameter")}
                                 for e in case_events],
        "topology_event_classes": [{
            "event_index": int(e["event_index"]), "kind": e["kind"],
            "physically_real": e["physically_real"] == "1", "detail": e["detail"],
            "R": f(e, "R"), "R_lo": f(e, "R_lo"),
            "uncertainty": f(e, "uncertainty"),
            "fold_t_seed_valid": e["fold_t_seed_valid"] == "1",
        } for e in case_all_events],
        "pointwise_fixed_R_fd_checks": [{
            "cell": int(x["cell"]), "R": f(x, "R"), "step": f(x, "step"),
            "analytic_Fp": f(x, "analytic_Fp"), "finite_difference_Fp": f(x, "fd_Fp"),
            "relative_difference": f(x, "relative_difference"),
            "all_samples_reliable": all(x[k] == "1" for k in ("base_ok", "plus_ok", "minus_ok")),
        } for x in pointwise_checks],
        "levels": level_data,
        "all_cell_integrals_certifiable_from_shadow_nodes":
            all(x["all_cells_complete"] for x in level_data),
        "endpoint_near_samples_level8": endpoint_stats,
        "radial_jet_method_counts_level8": dict(method_counts),
        "invalid_or_nonfinite_radial_samples_level8": failed_nodes,
        "invalid_sample_reason_counts_level8": dict(reject_counts),
        "maximum_abs_node_terms_level8": {
            "fixed_JFp": max(fixed_abs, default=math.nan),
            "radial_JFRRp": max(radial_abs, default=math.nan),
            "map_FJp": max(jac_abs, default=math.nan),
        },
        "ledger_diagnostics_level8": {
            "radial_old_weighted_detail_l2_sum":
                level_data[-1]["old_weighted_detail_l2_sum"],
            "radial_moving_weighted_detail_l2_sum":
                level_data[-1]["moving_weighted_detail_l2_sum"],
            "base_fixed_R_gradient_nodewise_abs_weighted_l1": base_gradient_ledger_l1,
            "event_radius_uncertainty_max": max(
                (f(e, "uncertainty") for e in case_events), default=math.nan),
            "event_radius_uncertainty_sum": math.fsum(
                f(e, "uncertainty") for e in case_events),
            "note": "inner/geometry/roundoff are absolute Fejer-weighted node contributions from the existing fixed-R sample ledger; event uncertainties are reported as endpoint-radius diagnostics. Error propagation for the added F_R*R_p and F*J_p terms is not certified in this shadow.",
        },
    }
    results.append(result)

summary = {
    "phase": "Phase72 fixed-R vs moving-map gradient shadow",
    "base_commit": "dfe09dee62a99bbd18f3ce01246c84c6e130464b",
    "input": {
        "path": "evidence/holonomic/adaptive_radial_20260911/reference_cases.tsv",
        "sha256": "5d625cbc9f4dc16de2a81754bbff131f1cb1ba9780fccdff4ddb210417347548",
    },
    "production_path_modified": False,
    "hermite_used": False,
    "gradient_integrand": {
        "old": "J * F_p / D",
        "moving": "(J*F_p + J*F_R*R_p + F*J_p) / D",
        "endpoint_derivative": "R_p=-P_p/P_R; t_p=-(P_tR*R_p+P_tp)/P_tt at local DD-corrected topology fold seed",
        "fold_map_derivatives": "analytic endpoint differentiation of affine, one-sided quadratic, and two-sided sine-squared FoldRadialMap",
        "F_R_uniform": "analytic root-pair width_R with implicit angular-endpoint derivative as fallback",
        "F_R_limb_darkening": "directional derivative of the same 16-node K-rule; same 8-vs-16 acceptance gate; direct angular-rescue derivative only when K representation rejects",
        "global_identity": "moving-old = integral d_x(F*R_p)/D; internal nonphysical cuts held fixed; shared physical-event boundary terms cancel between adjacent cells. Only the newly created/vanishing fold-local flux has F_fold->0; the total flux at the event need not vanish",
    "independent_fd_is_formal_bound": False,
    },
    "references": {
        "rand035": {
            "method": "fresh cold topology and qf angular/root direct-integral 5-point FD",
            "observed_spread_is_formal_bound": False,
            "raw_snapshot_base_commit": "f83e30f398fa1a7918f28de0afa30165f47a5cdc",
            "raw_files": ["rand035_reference/reference.tsv", "rand035_reference/reference_x_high.tsv"],
            "fresh_topology_ok": "32/32",
            "direct_integrals_ok": "144/144",
            "relevant_source_diff_to_base": False,
        },
        "caustic-cross": {
            "method": "fresh topology + direct arc-flux value; central FD",
            "radial_orders": [128, 256, 512, 1024],
            "parameter_step_range": [2e-6, 2e-4],
            "observed_spread_is_formal_bound": False,
        },
    },
    "results": results,
}
(HERE / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")

with (HERE / "references.tsv").open("w") as out:
    out.write("case\tu\tparameter\tfd_reference\tobserved_spread\tmethod\tsource\n")
    for (name,u,par),ref in sorted(refs.items()):
        out.write(f"{name}\t{u}\t{par}\t{ref['value']:.17g}\t{ref['observed_spread']:.17g}\t"
                  f"{ref['method']}\t{ref['source']}\n")

print(json.dumps({"rows": len(results), "status": "summarized",
                  "moving_map_to_old_detail_ratios": {
                      f"{r['case']}/u{r['u']}/{r['parameter']}":
                      r["levels"][-1]["moving_weighted_detail_l2_sum"] /
                      r["levels"][-1]["old_weighted_detail_l2_sum"]
                      if r["levels"][-1]["old_weighted_detail_l2_sum"] else None
                      for r in results},
                  "old_valid_only_to_fd_level8": {
                      f"{r['case']}/u{r['u']}/{r['parameter']}": r["levels"][-1]["old_abs_error_vs_independent_fd"]
                      for r in results},
                  "moving_valid_only_to_fd_level8": {
                      f"{r['case']}/u{r['u']}/{r['parameter']}": r["levels"][-1]["moving_abs_error_vs_independent_fd"]
                      for r in results},
                  "whole_domain_complete": {
                      f"{r['case']}/u{r['u']}/{r['parameter']}":
                      r["levels"][-1]["all_cells_complete"] for r in results}}, indent=2))
