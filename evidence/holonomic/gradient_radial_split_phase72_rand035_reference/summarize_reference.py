#!/usr/bin/env python3
"""Summarize the Phase72 rand035 independent-reference follow-up."""
import csv
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
REF = HERE / "reference.tsv"
REF_HIGH = HERE / "reference_x_high.tsv"
CORPUS = REPO / "evidence/holonomic/gradient_radial_split_phase72/corpus.tsv"
INPUT = REPO / "evidence/holonomic/adaptive_radial_20260911/reference_cases.tsv"


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


raw_lines = []
for path in (REF, REF_HIGH):
    if path.exists():
        raw_lines.extend(line.rstrip("\n").split("\t") for line in path.open())
ref_rows = [x for x in raw_lines if x[0] == "REF"]
topo_rows = [x for x in raw_lines if x[0] == "TOPO"]
fd_rows = [x for x in raw_lines if x[0] == "FD5"]

ref_by_key = {}
for x in ref_rows:
    key = (x[3], int(x[4]), int(x[5]), int(x[6]))
    ref_by_key[key] = {
        "mu_uniform": float(x[10]), "mu_ld_u_0_5": float(x[11]),
        "failure": int(x[12]), "radial_nodes": int(x[18]),
        "arc_nodes": int(x[19]), "full_nodes": int(x[20]),
        "empty_nodes": int(x[21]), "min_phi": float(x[22]),
    }

fd_by_key = {}
for x in fd_rows:
    fd_by_key[(x[3], int(x[4]), int(x[6]))] = (float(x[7]), float(x[8]))

topo_status = [int(x[7]) for x in topo_rows]
topo_shapes = sorted({(int(x[8]), int(x[9])) for x in topo_rows})
failure_counts = {}
for x in ref_rows:
    failure_counts[x[12]] = failure_counts.get(x[12], 0) + 1

if len(topo_rows) != 32 or any(status != 0 for status in topo_status):
    raise RuntimeError(f"cold topology audit incomplete/failed: {len(topo_rows)} rows, {topo_status}")
if len(ref_rows) != 144 or any(int(x[12]) != 0 for x in ref_rows):
    raise RuntimeError(f"direct reference integral incomplete/failed: {len(ref_rows)} rows, {failure_counts}")
if not any(x[0] == "FD5" and int(x[6]) == 6 for x in raw_lines):
    raise RuntimeError("focused X-gradient resolution-6 FD rows are missing")

corpus = rows(CORPUS)
selected = [x for x in corpus if x["row"] in ("99", "100")
            and x["value_rtol"] == "0.001"
            and x["h_split_min_level"] in ("0", "4")]
mu_arms = {}
for x in selected:
    key = (x["row"], x["path"], x["rep"])
    mu_arms.setdefault(key, {})[int(x["h_split_min_level"])] = x["mu"]
if len(mu_arms) != 8 or any(set(v) != {0, 4} for v in mu_arms.values()):
    raise RuntimeError("paired p-only/h-split@4 value rows are incomplete")
mu_mismatches = sum(v[0] != v[4] for v in mu_arms.values())
if mu_mismatches:
    raise RuntimeError(f"paired mu values changed in {mu_mismatches}/8 rows")

input_cases = []
for i, line in enumerate(INPUT.read_text().splitlines(), 1):
    p = line.split()
    if i in (99, 100):
        input_cases.append({"row": i, "xs": float(p[0]), "ys": float(p[1]),
                            "rho": float(p[2]), "q": float(p[3]), "a": float(p[4]),
                            "barycentric": bool(int(p[5])), "u": float(p[6]),
                            "name": p[8]})

results = []
for case in ({"row": 99, "u": 0.0}, {"row": 100, "u": 0.5}):
    row, u = case["row"], case["u"]
    quantity_index = 0 if u == 0.0 else 1
    for axis, grad_index in (("X", 0), ("Y", 1)):
        high_x = axis == "X" and ("X", 2, 6) in fd_by_key
        final_resolution = 6 if high_x else 5
        fd_final = fd_by_key[(axis, 2, final_resolution)][quantity_index]
        fd_hmid = fd_by_key[(axis, 1, final_resolution)][quantity_index]
        fd_hbig = fd_by_key[(axis, 0, 4)][quantity_index]
        resolution_delta_small_h = abs(
            fd_by_key[(axis, 2, final_resolution - 1)][quantity_index] - fd_final)
        resolution_delta_mid_h = abs(
            fd_by_key[(axis, 1, final_resolution - 1)][quantity_index] - fd_hmid)
        step_delta = abs(fd_hmid - fd_final)
        reference_observed_spread = max(
            resolution_delta_small_h, resolution_delta_mid_h, step_delta)

        arms = []
        for arm in (0, 4):
            runset = [x for x in selected if int(x["row"]) == row
                      and int(x["h_split_min_level"]) == arm]
            representative = next(x for x in runset
                                  if x["path"] == "cold" and x["rep"] == "0")
            c = grad_index + 1
            ledger = {name: float(representative[f"{name}{c}"])
                      for name in ("radial", "inner", "geometry", "event", "roundoff")}
            estimated = float(representative[f"error{grad_index}"])
            gradient = float(representative[f"grad{grad_index}"])
            actual = abs(gradient - fd_final)
            arms.append({
                "mode": "p_only" if arm == 0 else "h_split_at_4",
                "representative": {"path": "cold", "repeat": 0},
                "nodes": int(representative["nodes"]),
                "mu": float(representative["mu"]),
                "gradient": gradient,
                "gradient_quality": representative[f"quality{grad_index}"],
                "gradient_reason": representative[f"reason{grad_index}"],
                "ledger": ledger,
                "ledger_total": estimated,
                "reference": fd_final,
                "observed_absolute_error": actual,
                "observed_error_ratio_to_p_only": None,
                "observed_error_over_ledger": actual / estimated if estimated else None,
                "ledger_over_observed_error": estimated / actual if actual else None,
                "paired_replicates": len(runset),
                "distinct_gradients_across_path_and_repeat": len({
                    x[f"grad{grad_index}"] for x in runset}),
                "distinct_qualities_across_path_and_repeat": sorted({
                    x[f"quality{grad_index}"] for x in runset}),
            })
        results.append({
            "row": row, "case": "rand035", "u": u, "axis": axis,
            "user_coordinate": "xs" if axis == "X" else "ys",
            "input_is_barycentric": False,
            "reference": {
                "estimator": "5-point central FD of direct physical-value integral",
                "fd_step_rho_multipliers": [0.001, 0.0005, 0.00025],
                "chosen_step": 0.00025,
                "chosen_resolution_id": final_resolution,
                "chosen_raw": "reference_x_high.tsv" if high_x else "reference.tsv",
                "chosen_resolution": {
                    "nr": 160 if high_x else 128,
                    "radial_subdivisions": 32 if high_x else 16,
                    "angular_nodes": 1024,
                },
                "value_gradient": fd_final,
                "mid_step_high_resolution_gradient": fd_hmid,
                "large_step_resolution4_gradient": fd_hbig,
                "step_difference_high_resolution": step_delta,
                "resolution_difference_small_step": resolution_delta_small_h,
                "resolution_difference_mid_step": resolution_delta_mid_h,
                "observed_stability_spread": reference_observed_spread,
                "previous_resolution_small_step_gradient": fd_by_key[
                    (axis, 2, final_resolution - 1)][quantity_index],
                "previous_resolution_mid_step_gradient": fd_by_key[
                    (axis, 1, final_resolution - 1)][quantity_index],
            },
            "arms": arms,
        })

for result in results:
    p_error, h_error = (arm["observed_absolute_error"] for arm in result["arms"])
    result["h_split_over_p_only_observed_error_ratio"] = h_error / p_error
    result["arms"][1]["observed_error_ratio_to_p_only"] = h_error / p_error
    if not h_error > p_error:
        raise RuntimeError(f"h-split@4 was not worse for {result['u']} {result['axis']}")

summary = {
    "phase": "Phase72 rand035 gradient-reference continuation",
    "base_commit": "f83e30f398fa1a7918f28de0afa30165f47a5cdc",
    "production_solver_modified": False,
    "input_rows": input_cases,
    "reference_method": {
        "topology": "fresh cold classify_cells for every +/-h and +/-2h point; no warm roots reused",
        "angular_boundary": "binary128 boundary quartic coefficients, qf Sturm root isolation, then safeguarded qf Newton against direct lens phi",
        "observable": "direct phi-based F0 and sqrt(phi) angular integration; no production K-rule or analytic gradient",
        "radial_rule": "composite Gauss-Legendre on the existing fold-mapped radial coordinate",
        "angular_rule": "direct Gauss-Chebyshev first-kind sqrt-weight integral on certified arcs; periodic trapezoid for full circle",
        "finite_difference": "5-point central derivative, fresh cold topology for each perturbed LensParams",
        "resolution_levels": [
            {"id": 0, "nr": 48, "radial_subdivisions": 1, "angular_nodes": 96},
            {"id": 1, "nr": 48, "radial_subdivisions": 1, "angular_nodes": 384},
            {"id": 2, "nr": 80, "radial_subdivisions": 1, "angular_nodes": 384},
            {"id": 3, "nr": 80, "radial_subdivisions": 4, "angular_nodes": 384},
            {"id": 4, "nr": 112, "radial_subdivisions": 8, "angular_nodes": 768},
            {"id": 5, "nr": 128, "radial_subdivisions": 16, "angular_nodes": 1024,
             "used_for_fd_steps": [0.0005, 0.00025]},
            {"id": 6, "nr": 160, "radial_subdivisions": 32, "angular_nodes": 1024,
             "used_for_fd_steps": [0.0005, 0.00025], "axis": "X"},
        ],
        "fd_step_rho_multipliers": [0.001, 0.0005, 0.00025],
        "topology_rows": len(topo_rows),
        "topology_status_ok": sum(s == 0 for s in topo_status),
        "topology_audit_complete": len(topo_rows) == 32 and all(s == 0 for s in topo_status),
        "topology_cell_event_shapes": [list(x) for x in topo_shapes],
        "reference_integrals": len(ref_rows),
        "reference_failures_by_code": failure_counts,
        "reference_audit_complete": len(ref_rows) == 144 and all(int(x[12]) == 0 for x in ref_rows),
        "reference_value_nonfinite_count": sum(
            not math.isfinite(float(x[i])) for x in ref_rows for i in (10, 11)),
        "note": "Observed convergence spread is diagnostic, not a formal interval enclosure.",
    },
    "whole_epoch_source": {
        "raw": "../gradient_radial_split_phase72/corpus.tsv",
        "value_rtol": 1e-3,
        "gradient_policy": "ValueFirst",
        "gradient_tolerance": {"atol": 1e-4, "rtol": 1e-2},
        "gradient_round_budget": 4,
        "p_only_arm": 0,
        "h_split_at_4_arm": 4,
        "paired_mu_bitwise_mismatches": mu_mismatches,
    },
    "comparisons": results,
    "decision": {
        "h_split_at_4_defaultization": "reject",
        "reason": "For all four rand035 X/Y cases, the 4-round h-split@4 derivative is materially farther from the converged direct-value FD reference than p-only; its FiniteUncertified status reflects real gradient degradation, not only an over-conservative ledger.",
        "estimator_followup": "Do not modify the ledger in this phase: the actual error is worse under h-split@4, so the condition for estimator-ledger investigation was not met.",
        "hermite_or_threshold_changes": False,
    },
}

(HERE / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
with (HERE / "comparison.tsv").open("w") as f:
    f.write("row\tu\taxis\tarm\tgradient\tquality\treference\tactual_abs_error\tledger_total\tradial\tinner\tgeometry\tevent\troundoff\tnodes\n")
    for result in results:
        for arm in result["arms"]:
            led = arm["ledger"]
            f.write("\t".join(map(str, [result["row"], result["u"], result["axis"],
                arm["mode"], arm["gradient"], arm["gradient_quality"], arm["reference"],
                arm["observed_absolute_error"], arm["ledger_total"], led["radial"],
                led["inner"], led["geometry"], led["event"], led["roundoff"], arm["nodes"]])) + "\n")

print(json.dumps({"summary": str(HERE / "summary.json"),
                  "comparison": str(HERE / "comparison.tsv"),
                  "topology_ok": summary["reference_method"]["topology_status_ok"],
                  "topology_total": summary["reference_method"]["topology_rows"],
                  "reference_integrals": summary["reference_method"]["reference_integrals"],
                  "reference_failures": summary["reference_method"]["reference_failures_by_code"]}, indent=2))
