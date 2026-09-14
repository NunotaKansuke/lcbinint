#!/usr/bin/env python3
"""Build Phase73 cell-error ranking and machine-readable summary."""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PHASE72 = ROOT / "evidence/holonomic/moving_map_gradient_phase72"


def tsv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def num(row, field, cast=float):
    return cast(row[field])


def key(row, param, cell):
    return (int(row), param, int(cell))


base_cells = tsv(PHASE72 / "cells.tsv")
production = {
    key(r["row"], r["parameter"], r["cell"]): r
    for r in base_cells if r["level"] == "8"
}
same = {
    key(r["row"], r["parameter"], r["cell"]): r
    for r in tsv(HERE / "same_node_cells.tsv")
}
caustic_rows = tsv(HERE / "caustic_fold_map_diagnostic.tsv")
rand_rows = tsv(HERE / "rand_direct.tsv")
rand_high_rows = tsv(HERE / "rand_cell3_direct_high.tsv")


def choose(rows, *, row, param, cell, nr, subdivisions, angular):
    found = [r for r in rows if int(r["row"]) == row and r["parameter"] == param
             and int(r["cell"]) == cell and int(r["nr"]) == nr
             and int(r["subdivisions"]) == subdivisions
             and int(r["angular"]) == angular]
    if len(found) != 1:
        raise RuntimeError((row, param, cell, nr, subdivisions, angular, len(found)))
    return found[0]


def radial_ref(row, param, cell):
    if row == 17:
        return choose(caustic_rows, row=row, param=param, cell=cell,
                      nr=256, subdivisions=1, angular=128)
    if cell in (2, 3, 8):
        path = HERE / f"rand_cell{cell}_direct_high.tsv"
        rows = tsv(path)
        return choose(rows, row=row, param=param, cell=cell,
                      nr=128, subdivisions=16, angular=1024)
    return choose(rand_rows, row=row, param=param, cell=cell,
                  nr=112, subdivisions=8, angular=1024)


def radial_series(row, param, cell):
    if row == 17:
        out = []
        for n in (128, 256, 512, 1024):
            r = choose(caustic_rows, row=row, param=param, cell=cell,
                       nr=n, subdivisions=1, angular=128)
            out.append({"radial_order": n,
                        "value": float(r["mu0_grad"]),
                        "whole": float(r["mu0_partial_sum"])})
        return out
    if cell in (2, 3, 8):
        rows = tsv(HERE / f"rand_cell{cell}_direct_high.tsv")
        candidates = [r for r in rows if int(r["row"]) == row and
                      r["parameter"] == param and int(r["cell"]) == cell and
                      int(r["angular"]) == 1024]
        # Preserve unique settings (the common 112x8 reference is repeated
        # when the targeted extended run is enabled).
        unique = {}
        for r in candidates:
            setting = (int(r["nr"]), int(r["subdivisions"]))
            unique[setting] = r
        return [{"radial_order": n, "subdivisions": s,
                 "value": float(r["mu0_grad"] if float(r["u"]) == 0 else r["muhalf_grad"])}
                for (n, s), r in sorted(unique.items(), key=lambda x: x[0][0]*x[0][1])]
    return []


phase72 = json.load((PHASE72 / "summary.json").open())
fd = {}
for r in phase72["results"]:
    fd[(int(r["row"]), r["parameter"])] = r["independent_fd"]

same_nodes = tsv(HERE / "same_node_nodes.tsv")
same_node_arcs = tsv(HERE / "same_node_arcs.tsv")
same_node_cells = tsv(HERE / "same_node_cells.tsv")
same_node_valid = [r for r in same_nodes
                   if r["qf128_ok"] == "1" and r["qf256_ok"] == "1"]
same_node_angular_audit = {
    "node_rows": len(same_nodes), "qf_valid_rows": len(same_node_valid),
    "qf_failures": sum(int(r["qf_failures"]) for r in same_node_cells),
    "max_abs_qf_Fp_128_vs_256_at_nodes": max(
        (abs(float(r["qf_Fp_128"]) - float(r["qf_Fp_256"]))
         for r in same_node_valid), default=0.0),
    "max_abs_qf_old_128_vs_256_at_nodes": max(
        (abs(float(r["qf_old_128"]) - float(r["qf_old_256"]))
         for r in same_node_valid), default=0.0),
    "max_abs_q8_cell_sum_128_vs_256": max(
        (abs(float(r["qf_old_q8_128"]) - float(r["qf_old_q8_256"]))
         for r in same_node_cells), default=0.0),
    "comparison_precision": "serialized binary64 columns; zero means identical at recorded output precision",
}
rank_rows = []
case_specs = [(17, "a"), (99, "X"), (99, "Y"), (100, "X"), (100, "Y")]
case_summaries = []
for row, param in case_specs:
    case_prod = {cell: r for (rrow, p, cell), r in production.items()
                 if rrow == row and p == param}
    case_same = {cell: r for (rrow, p, cell), r in same.items()
                 if rrow == row and p == param}
    cell_ids = sorted(set(case_prod) | set(case_same))
    cell_records = []
    for cell in cell_ids:
        pr = case_prod.get(cell)
        sr = case_same.get(cell)
        rr = radial_ref(row, param, cell)
        source_component = "mu0_grad" if float(pr["u"]) == 0.0 else "muhalf_grad"
        qf_ref = float(rr[source_component])
        prod_q8 = float(pr["old_q"]) if pr else 0.0
        same_q8 = float(sr["qf_old_q8_256"]) if sr else 0.0
        prod_complete = bool(int(pr["complete"])) if pr else True
        local_abs = float(sr["local_derivative_delta"]) if sr else 0.0
        signed_prod_ref = prod_q8 - qf_ref
        signed_same_ref = same_q8 - qf_ref
        signed_local = prod_q8 - same_q8
        complete_local = signed_local if prod_complete else None
        rec = {
            "row": row, "case": "caustic-cross" if row == 17 else "rand035",
            "u": float(pr["u"]) if pr else (0.0 if row != 100 else 0.5),
            "parameter": param, "cell": cell,
            "kind": pr["kind"] if pr else "unknown",
            "production_complete": prod_complete,
            "production_valid_nodes": int(sr["prod_valid_nodes"]) if sr else 0,
            "production_invalid_nodes": int(sr["production_invalid_nodes"]) if sr else 0,
            "independent_qf_method": "direct QF fixed-R derivative + radial Gauss-Legendre",
            "reference_radial": {"order": int(rr["nr"]),
                                 "subdivisions": int(rr["subdivisions"]),
                                 "angular_nodes": int(rr["angular"])},
            "reference_radial_sequence": radial_series(row, param, cell),
            "production_q8": prod_q8,
            "independent_qf_same_production_nodes_q8": same_q8,
            "independent_qf_radial_reference": qf_ref,
            "production_minus_radial_reference": signed_prod_ref,
            "same_nodes_minus_radial_reference": signed_same_ref,
            "same_node_local_derivative_signed_difference": complete_local,
            "same_node_local_derivative_abs_weighted_difference": local_abs,
            "radial_quadrature_delta_same_nodes_minus_high_gl": signed_same_ref,
            "classification": "production incomplete; cell contribution is partial" if not prod_complete else "complete",
        }
        cell_records.append(rec)
        rank_rows.append(rec)
    case_fd = fd[(row, param)]
    prod_total = sum(r["production_q8"] for r in cell_records)
    same_total = sum(r["independent_qf_same_production_nodes_q8"] for r in cell_records)
    ref_total = sum(r["independent_qf_radial_reference"] for r in cell_records)
    abs_prod_cell = sum(abs(r["production_q8"]) for r in cell_records)
    abs_ref_cell = sum(abs(r["independent_qf_radial_reference"]) for r in cell_records)
    sorted_cells = sorted(cell_records,
                          key=lambda r: abs(r["production_minus_radial_reference"]),
                          reverse=True)
    arc_records = [r for r in same_node_arcs
                   if int(r["row"]) == row and r["parameter"] == param]
    arc_contribs = [float(r["combined_arc_contribution"]) for r in arc_records]
    arc_abs_sum = sum(abs(x) for x in arc_contribs)
    arc_net = sum(arc_contribs)
    arc_cells = {}
    for r in arc_records:
        arc_cells.setdefault(int(r["cell"]), []).append(
            float(r["combined_arc_contribution"]))
    per_cell_arc_cancel = {
        str(ci): (sum(abs(v) for v in vals) / abs(sum(vals))
                  if sum(vals) else None)
        for ci, vals in sorted(arc_cells.items())
    }
    abs_cell_error = sum(abs(r["production_minus_radial_reference"])
                         for r in cell_records)
    net_cell_error = sum(r["production_minus_radial_reference"]
                         for r in cell_records)
    for r in cell_records:
        r["share_of_absolute_cell_error"] = (
            abs(r["production_minus_radial_reference"]) / abs_cell_error
            if abs_cell_error else None)
        r["share_of_net_cell_error"] = (
            r["production_minus_radial_reference"] / net_cell_error
            if net_cell_error else None)
    if row == 17:
        ref_note = (
            "The diagnostic-only theta=pi fold map gives close 128/256 radial results; "
            "512/1024 drift as double R nodes approach the endpoint and are limited by "
            "representability, so higher order is not treated as monotonically better."
        )
    else:
        ref_note = (
            "Cell 3 agrees across 112x8, 128x16, and 160x32. Cell 2 has a nonmonotone "
            "160x32 outlier and is referenced at 128x16; cell 8 drifts mildly with "
            "resolution. These are observed convergence checks, not formal bounds."
        )
    if row == 17:
        ref_sweep = []
        for n in (128, 256, 512, 1024):
            selected = [r for r in caustic_rows if int(r["nr"]) == n]
            totals = {(float(r["mu0_partial_sum"])) for r in selected}
            refs_for_n = [float(r["mu0_grad"]) for r in selected
                          if int(r["cell"]) in (1, 5, 6, 7, 12)]
            ref_sweep.append({"radial_order": n,
                              "whole": next(iter(totals)),
                              "cell5": next(float(r["mu0_grad"]) for r in selected if int(r["cell"]) == 5),
                              "cell7": next(float(r["mu0_grad"]) for r in selected if int(r["cell"]) == 7),
                              "max_active_cell": max(refs_for_n) if refs_for_n else None})
    else:
        ref_sweep = [{"cell": r["cell"], "sequence": r["reference_radial_sequence"]}
                     for r in cell_records if r["kind"] == "arcs"]
    case_summaries.append({
        "row": row, "case": "caustic-cross" if row == 17 else "rand035",
        "u": float(case_prod[0]["u"]) if case_prod else (0.0 if row != 100 else 0.5),
        "parameter": param,
        "production_whole_q8_valid_sum": prod_total,
        "production_whole_q8_status_complete": all(r["production_complete"] for r in cell_records),
        "independent_qf_same_node_q8_sum": same_total,
        "independent_qf_radial_reference_sum": ref_total,
        "independent_value_fd": float(case_fd["value"]),
        "fd_observed_spread_not_formal_bound": float(case_fd["observed_spread"]),
        "production_minus_fd": prod_total - float(case_fd["value"]),
        "qf_radial_reference_minus_fd": ref_total - float(case_fd["value"]),
        "q8_production_minus_same_node_qf": prod_total - same_total,
        "cell_abs_sum_over_net_production": abs_prod_cell / abs(prod_total) if prod_total else None,
        "cell_abs_sum_over_net_reference": abs_ref_cell / abs(ref_total) if ref_total else None,
        "same_node_arc_abs_sum_over_net": arc_abs_sum / abs(arc_net) if arc_net else None,
        "same_node_arc_cell_interarc_cancellation_ratios": per_cell_arc_cancel,
        "largest_error_cell_share_of_absolute_cell_error": [
            {"cell": r["cell"], "share": r["share_of_absolute_cell_error"]}
            for r in sorted_cells[:5]],
        "largest_error_cell_share_of_net_cell_error": [
            {"cell": r["cell"], "share": r["share_of_net_cell_error"]}
            for r in sorted_cells[:5]],
        "reference_resolution_note": ref_note,
        "reference_resolution_sequence": ref_sweep,
        "largest_cell_error_contributors": sorted_cells[:5],
        "cells": cell_records,
    })

endpoint_rows = tsv(HERE / "endpoint_audit.tsv")
endpoint_audit = []
for r in endpoint_rows:
    endpoint_audit.append({
        "cell": int(r["cell"]), "R_chart_double": float(r["R_chart_double"]),
        "R_p4_qf": r["R_p4_qf"], "R_delta_qf": r["R_delta_qf"],
        "p4_at_double": r["p4_at_double"], "p4_at_qf_root": r["p4_at_root"],
        "phi_at_theta_pi": r["phi_pi"], "phi_theta_at_pi": r["phi_theta_pi"],
        "phi_R_pi": float(r["phi_R_pi"]),
        "phi_thetatheta_pi": float(r["phi_thetatheta_pi"]),
        "g_pi": float(r["g_pi"]), "ordinary_fold_check": bool(int(r["ordinary_fold_check"])),
    })

event_rows = tsv(PHASE72 / "all_events.tsv")
caustic_event_cell_map_audit = []
for cell_id in (5, 7):
    cell_row = next(r for r in base_cells if r["row"] == "17" and
                    r["parameter"] == "a" and r["level"] == "8" and
                    int(r["cell"]) == cell_id)
    radius = float(cell_row["a"])
    coincident = [r for r in event_rows if r["row"] == "17" and
                  r["parameter"] == "a" and
                  abs(float(r["R"]) - radius) < 1e-14]
    caustic_event_cell_map_audit.append({
        "cell": cell_id, "left_radius": radius,
        "cell_left_fold_map": bool(int(cell_row["left_fold"])),
        "cell_right_fold_map": bool(int(cell_row["right_fold"])),
        "coincident_topology_events": [{
            "kind": r["kind"], "physically_real": bool(int(r["physically_real"])),
            "fold_t_seed": float(r["fold_t_seed"]),
            "fold_t_seed_valid": bool(int(r["fold_t_seed_valid"])),
            "radius": float(r["R"]),
        } for r in coincident],
    })

ld_rows = tsv(HERE / "ld_cells.tsv")
ld_summary = []
for param in ("X", "Y"):
    rows = [r for r in ld_rows if r["parameter"] == param]
    valid = [r for r in rows if int(r["cell"]) in (2, 3, 8)]
    uniform = sum(float(r["uniform_contribution"]) for r in valid)
    dvk = sum(float(r["dvK_contribution"]) for r in valid)
    vdk = sum(float(r["v_dK_contribution"]) for r in valid)
    ldnet = sum(float(r["LD_total"]) for r in valid)
    abs_uniform = sum(abs(float(r["uniform_contribution"])) for r in valid)
    abs_ld = sum(abs(float(r["LD_total"])) for r in valid)
    abs_dvk_cell = sum(abs(float(r["dvK_contribution"])) for r in valid)
    abs_vdk_cell = sum(abs(float(r["v_dK_contribution"])) for r in valid)
    source_invalid = sum(int(r["source_invalid_nodes"]) for r in rows)
    k_rejects = sum(int(r["rejected_K_nodes"]) for r in rows)
    per_cell_ld = []
    for r in rows:
        rec = {k: r[k] for k in r}
        if int(r["cell"]) in (2, 3, 8):
            dvk = float(r["dvK_contribution"])
            vdk = float(r["v_dK_contribution"])
            ld = float(r["LD_total"])
            rec["dvK_v_dK_cancellation_after_radial"] = (
                (abs(dvk) + abs(vdk)) / abs(ld) if ld else None)
        else:
            rec["dvK_v_dK_cancellation_after_radial"] = None
        per_cell_ld.append(rec)
    ld_summary.append({
        "parameter": param, "u": 0.5, "included_cells": [2, 3, 8],
        "cell11_status": "not decomposed: 48 source-invalid nodes and 207 K-rule rejections; direct fixed-R whole-cell contribution is O(1e-17)",
        "uniform_sum": uniform, "LD_dvK_sum": dvk, "LD_v_dK_sum": vdk,
        "LD_sum": ldnet, "combined_uniform_plus_LD": uniform + ldnet,
        "uniform_cross_cell_cancellation_ratio": abs_uniform / abs(uniform) if uniform else None,
        "LD_cross_cell_cancellation_ratio": abs_ld / abs(ldnet) if ldnet else None,
        "dvK_cross_cell_cancellation_ratio": abs_dvk_cell / abs(dvk) if dvk else None,
        "v_dK_cross_cell_cancellation_ratio": abs_vdk_cell / abs(vdk) if vdk else None,
        "dvK_vs_v_dK_cancellation_after_radial": (abs(dvk) + abs(vdk)) / abs(ldnet) if ldnet else None,
        "source_invalid_nodes": source_invalid, "K_rule_rejections": k_rejects,
        "max_pointwise_algebra_identity_residual": max(float(r["max_pointwise_residual"]) for r in valid),
        "cells": per_cell_ld,
    })

summary = {
    "phase": "Phase73 gradient cell attribution",
    "base_commit": "f928e3b42d25635195c7fba985cb72b811c383f3",
    "scope": "diagnostic-only; no production path, tolerance, estimator, scheduler, or router changed",
    "reference_methods": {
        "fixed_R_same_node": "independent QF lens phi derivative, qf Sturm angular arcs, direct endpoint derivative for F0, direct angular derivative for Fhalf; evaluated at the exact Phase72 Q8 Fejer R nodes; angular 128-vs-256 audit",
        "radial_cell_reference": "same fixed-R derivative integrated per base cell by Gauss-Legendre with independent direct angular evaluation; caustic-only p4-fold-map transform probe",
        "whole_gradient_reference": "Phase72 fresh-cold-topology independent value FD; observed spread is diagnostic, not a formal bound",
        "limits": [
            "Cellwise Gauss-Legendre results are empirical convergence checks, not interval-certified errors.",
            "Very high radial orders near a fold can lose distinct R nodes in binary64; a nonmonotone result is retained as a limitation, not selected as truth.",
            "The Phase72 rand035 production sum is incomplete at cell 11; its 48 invalid nodes are not silently filled by the independent reference."
        ],
    },
    "caustic_projective_endpoint_audit": endpoint_audit,
    "caustic_event_cell_map_audit": caustic_event_cell_map_audit,
    "same_node_angular_128_vs_256_audit": same_node_angular_audit,
    "cases": case_summaries,
    "limb_darkening_term_decomposition": ld_summary,
    "interpretation": {
        "caustic_cross": "The Q8 production derivative agrees with independent QF fixed-R derivatives on the exact same radial nodes (whole difference about 1.6e-8), so the 0.1762 whole discrepancy is not a local derivative evaluation error. It is concentrated in cells 5/7: production is low by 0.0917 in cell 5 and high by 0.2679 in cell 7. QF audit proves both chart_p4 radii are ordinary theta=pi tangencies (phi=0, phi_theta=0, nonzero phi_R and phi_thetatheta), while Phase72 cell plans have no fold map on cell 5's left endpoint and only the other endpoint map on cell 7. The classifier reports coincident chart_p4 and physical_complex events; its finite-t stationary-root path does not provide the projective t=infinity fold endpoint to the adaptive fold-map gate. Forcing the endpoint map only in the independent radial diagnostic changes the unregularized N=256 result from -2.27643 to -2.40468919, consistent with independent whole FD -2.40468916. This refines Phase72: moving-map cancellation was locally correct, but the particular two dominant projective folds were not mapped. Caustic cell contributions also cancel strongly across cells (sum absolute / net about 55.6); the two arcs in cell 7 have the same sign, so inter-arc cancellation there is not the cause.",
        "rand035": "On the exact same nodes, production and independent QF fixed-R derivatives agree closely (whole valid-only differences are about 3e-9 to 1e-8). The production-vs-FD mismatch is concentrated in radial integration, especially cell 3, whose independent direct-QF GL values are stable across 112x8, 128x16, and 160x32. Cell 3 is about 105% of the signed net production-minus-selected-QF-radial-reference cell difference (smaller opposite-sign cell errors offset it), about 91.1% of summed absolute cellwise discrepancies, and about 98.5% of the production-minus-FD whole mismatch. These ratios have different denominators; the FD-vs-QF reference discrepancy is about 1e-8 to 4e-8. Cross-cell cancellation is modest (absolute cell sum / net about 1.03-1.05). For u=0.5, dvK+v*dK has strong cell-local cancellation after radial integration in X cells 3/8 (ratios 6.71/14.44) and Y cell 8 (7.36); however, the pointwise sum matches the production K derivative to <=1.1e-14. This indicates conditioning, not a demonstrated algebra error. Cell 11 is not production-complete.",
        "cell11": "The production audit remains incomplete at the 4.18e-10-wide cell 11: 48/255 source nodes are invalid and 207 of the remainder reject the K rule. The independent fixed-R reference finds an O(1e-17) contribution, which is numerically tiny here but does not retroactively certify production completeness.",
    },
    "files": ["cell_ranking.tsv", "endpoint_audit.tsv", "same_node_nodes.tsv",
              "same_node_cells.tsv", "same_node_arcs.tsv", "ld_nodes.tsv",
              "ld_cells.tsv", "caustic_unregularized_direct.tsv",
              "caustic_fold_map_diagnostic.tsv", "rand_direct.tsv",
              "rand_cell2_direct_high.tsv", "rand_cell3_direct_high.tsv",
              "rand_cell8_direct_high.tsv"],
}

with (HERE / "cell_ranking.tsv").open("w", newline="") as f:
    fields = ["row", "case", "u", "parameter", "cell", "kind",
              "production_complete", "production_valid_nodes", "production_invalid_nodes",
              "reference_order", "reference_subdivisions", "reference_angular_nodes",
              "production_q8", "independent_qf_same_production_nodes_q8",
              "independent_qf_radial_reference", "production_minus_radial_reference",
              "same_nodes_minus_radial_reference", "same_node_local_derivative_signed_difference",
              "same_node_local_derivative_abs_weighted_difference",
              "share_of_absolute_cell_error", "share_of_net_cell_error"]
    w = csv.DictWriter(f, fields, delimiter="\t", lineterminator="\n")
    w.writeheader()
    for r in rank_rows:
        q = r["reference_radial"]
        w.writerow({"row": r["row"], "case": r["case"], "u": r["u"],
                    "parameter": r["parameter"], "cell": r["cell"], "kind": r["kind"],
                    "production_complete": int(r["production_complete"]),
                    "production_valid_nodes": r["production_valid_nodes"],
                    "production_invalid_nodes": r["production_invalid_nodes"],
                    "reference_order": q["order"], "reference_subdivisions": q["subdivisions"],
                    "reference_angular_nodes": q["angular_nodes"],
                    "production_q8": r["production_q8"],
                    "independent_qf_same_production_nodes_q8": r["independent_qf_same_production_nodes_q8"],
                    "independent_qf_radial_reference": r["independent_qf_radial_reference"],
                    "production_minus_radial_reference": r["production_minus_radial_reference"],
                    "same_nodes_minus_radial_reference": r["same_nodes_minus_radial_reference"],
                    "same_node_local_derivative_signed_difference": ("NA" if r["same_node_local_derivative_signed_difference"] is None else r["same_node_local_derivative_signed_difference"]),
                    "same_node_local_derivative_abs_weighted_difference": r["same_node_local_derivative_abs_weighted_difference"],
                    "share_of_absolute_cell_error": r["share_of_absolute_cell_error"],
                    "share_of_net_cell_error": r["share_of_net_cell_error"]})

with (HERE / "summary.json").open("w") as f:
    json.dump(summary, f, indent=2, allow_nan=False)
    f.write("\n")
