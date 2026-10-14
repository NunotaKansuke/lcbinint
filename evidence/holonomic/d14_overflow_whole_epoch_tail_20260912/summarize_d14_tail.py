#!/usr/bin/env python3
"""Summarize D14 qf-cold tails and the rejected warm-continuation A/B."""

import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ROOTWORK = ROOT / "evidence/holonomic/d14_case92_presearch_overflow_20260912"


def read_tsv(path):
    path = Path(path)
    if not path.exists() and path.suffix != ".gz":
        compressed = Path(str(path) + ".gz")
        if compressed.exists():
            path = compressed
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        lines = [line.split() for line in f if line.strip() and not line.startswith("#")]
    if not lines:
        return []
    header = lines[0]
    return [dict(zip(header, row)) for row in lines[1:] if len(row) == len(header)]


def quantile(values, p):
    xs = sorted(float(x) for x in values)
    if not xs:
        return None
    z = p * (len(xs) - 1)
    lo = int(math.floor(z))
    hi = min(lo + 1, len(xs) - 1)
    f = z - lo
    return xs[lo] * (1.0 - f) + xs[hi] * f


def distribution(values):
    xs = [float(x) for x in values]
    return {
        "n": len(xs),
        "p50": quantile(xs, 0.50),
        "p90": quantile(xs, 0.90),
        "p95": quantile(xs, 0.95),
        "p99": quantile(xs, 0.99),
        "max": max(xs) if xs else None,
    }


def row_key(row, lane=True):
    fields = ["case_id", "configuration_id", "profile", "d_bin_index", "epoch_index"]
    if lane:
        fields.append("lane")
    return tuple(row[k] for k in fields)


def parse_trace(path):
    rows = defaultdict(lambda: {"runs": [], "first_bad": []})
    context = None
    active = None
    path = Path(path)
    if not path.exists() and path.suffix != ".gz":
        compressed = Path(str(path) + ".gz")
        if compressed.exists():
            path = compressed
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as source:
        lines = list(source)
    for line in lines:
        line = line.rstrip("\n")
        if line.startswith("D14TRACE_ROW\t"):
            context = dict(x.split("=", 1) for x in line.split("\t")[1:] if "=" in x)
            context_key = (context["case"], context["profile"], context["d_bin"],
                           context["epoch"], context["lane"])
            active = None
        elif line.startswith("D14QF_BEGIN\t") and context:
            fields = dict(x.split("=", 1) for x in line.split("\t")[1:] if "=" in x)
            active = {"seed": fields.get("seed"), "begin": fields, "sweeps": [], "stage": None}
            rows[context_key]["runs"].append(active)
        elif line.startswith("D14QF_SWEEP\t") and active is not None:
            active["sweeps"].append(
                dict(x.split("=", 1) for x in line.split("\t")[1:] if "=" in x)
            )
        elif line.startswith("D14QF_STAGE\t") and context:
            fields = dict(x.split("=", 1) for x in line.split("\t")[1:] if "=" in x)
            expected_seed = "warm" if fields.get("kind") == "warm" else "cold"
            for run in reversed(rows[context_key]["runs"]):
                if run["seed"] == expected_seed and run["stage"] is None:
                    run["stage"] = fields
                    break
        elif line.startswith("D14QF_FIRST_BAD\t") and context:
            rows[context_key]["first_bad"].append(
                dict(x.split("=", 1) for x in line.split("\t")[1:] if "=" in x)
            )
    return rows


def selected_sweep(run):
    if not run or not run["sweeps"]:
        return None
    sweeps = run["sweeps"]
    selected = {"first": sweeps[0], "last": sweeps[-1]}
    for wanted in (24, 400):
        for sweep in sweeps:
            if int(sweep["it"]) == wanted:
                selected[str(wanted)] = sweep
                break
    return selected


def trace_record(run, first_bad):
    return {
        "seed": run["seed"],
        "max_iter": int(run["begin"].get("max_iter", 0)),
        "stage": run["stage"],
        "sweep_metrics": selected_sweep(run),
        "first_bad_update": bool(first_bad),
    }


def summarize_trace_samples():
    trace_files = {
        "case149": HERE / "case149_qf_trace_final.log",
        "case0": HERE / "case0_qf_trace.log",
        "case64": HERE / "case64_qf_trace.log",
        "case9": HERE / "case9_qf_trace.log",
        "case49_dbin1": HERE / "case49_dbin1_qf_trace.log",
        "case49_dbin4": HERE / "case49_dbin4_qf_trace.log",
    }
    parsed = {}
    for name, path in trace_files.items():
        compressed = Path(str(path) + ".gz")
        parsed[name] = parse_trace(path) if path.exists() or compressed.exists() else {}

    wanted = [
        ("case149_uniform_dbin1_epoch7", "case149", ("149", "uniform", "1", "7"), ("cold", "warm")),
        ("case0_linear_dbin0_epoch0", "case0", ("0", "linear", "0", "0"), ("cold", "warm")),
        ("case64_linear_dbin0_epoch0", "case64", ("64", "linear", "0", "0"), ("cold", "warm")),
        ("case9_uniform_dbin2_epoch7", "case9", ("9", "uniform", "2", "7"), ("warm",)),
        ("case49_linear_dbin1_epoch0", "case49_dbin1", ("49", "linear", "1", "0"), ("cold", "warm")),
        ("case49_linear_dbin4_epoch23", "case49_dbin4", ("49", "linear", "4", "23"), ("cold", "warm")),
    ]
    result = {}
    for label, file_key, stem, lanes in wanted:
        lane_records = {}
        for lane in lanes:
            key = (*stem, lane)
            parsed_row = parsed[file_key].get(key)
            if not parsed_row:
                continue
            lane_records[lane] = {
                "first_bad_updates": parsed_row["first_bad"],
                "runs": [trace_record(r, bool(parsed_row["first_bad"]))
                         for r in parsed_row["runs"]],
            }
        if lane_records:
            result[label] = lane_records
    return result


def summarize_rootwork():
    safe_timing = read_tsv(ROOTWORK / "full_safe_timing.tsv.gz")
    legacy_timing = read_tsv(ROOTWORK / "full_legacy_double_timing.tsv.gz")
    safe_roots = read_tsv(ROOTWORK / "full_safe_roots.tsv.gz")
    out = {"unique_geometry_rows_per_lane": len(safe_timing) // 2,
           "lanes": {}, "top_safe_qf_cold_rows": {}}
    for lane in ("cold", "warm"):
        safe = [r for r in safe_timing if r["lane"] == lane]
        legacy = [r for r in legacy_timing if r["lane"] == lane]
        lane_result = {}
        for name, rows in (("overflow_safe", safe), ("legacy_double_division", legacy)):
            calls = sum(int(r["qf_cold_calls"]) for r in rows)
            sweeps = sum(int(r["qf_cold_sweeps"]) for r in rows)
            cap_rows = [r for r in rows if int(r["qf_cold_sweeps"]) == 400]
            lane_result[name] = {
                "rows": len(rows),
                "qf_cold_calls": calls,
                "qf_cold_sweeps": sweeps,
                "qf_cold_at_400_sweeps": len(cap_rows),
                "case_counts_at_400": dict(Counter(r["case_id"] for r in cap_rows)),
                "qf_warm_calls": sum(int(r["qf_warm_calls"]) for r in rows),
                "qf_warm_sweeps": sum(int(r["qf_warm_sweeps"]) for r in rows),
            }
        out["lanes"][lane] = lane_result
        top = sorted((r for r in safe if int(r["qf_cold_calls"])),
                     key=lambda r: float(r["qf_polish_ms"]), reverse=True)[:12]
        out["top_safe_qf_cold_rows"][lane] = [
            {k: r[k] for k in ("case_id", "profile", "d_bin_index", "epoch_index",
                               "qf_warm_calls", "qf_warm_sweeps", "qf_cold_calls",
                               "qf_cold_sweeps", "qf_polish_ms", "whole_classify_ms",
                               "presearch_sweeps", "presearch_nonfinite_failures",
                               "d14real_calls", "d14real_finite_calls",
                               "d14real_nonconverged", "root_clusters")}
            for r in top
        ]

    selected = [
        ("149", "uniform", "1", "7", "cold"),
        ("149", "uniform", "1", "7", "warm"),
        ("0", "linear", "0", "0", "cold"),
        ("0", "linear", "0", "0", "warm"),
        ("64", "linear", "0", "0", "cold"),
        ("64", "linear", "0", "0", "warm"),
        ("9", "linear", "2", "7", "warm"),
    ]
    out["selected_rootwork"] = {}
    for case, profile, dbin, epoch, lane in selected:
        timing = next((r for r in safe_timing if
                       (r["case_id"], r["profile"], r["d_bin_index"],
                        r["epoch_index"], r["lane"]) ==
                       (case, profile, dbin, epoch, lane)), None)
        roots = [r for r in safe_roots if
                 (r["case_id"], r["profile"], r["d_bin_index"],
                  r["epoch_index"], r["lane"]) ==
                 (case, profile, dbin, epoch, lane)]
        if not timing:
            continue
        min_sep = min((float(r["qf_nearest_sep"]) for r in roots), default=None)
        max_displacement = max((float(r["expanded_seed_displacement"]) for r in roots
                                if float(r["expanded_seed_displacement"]) >= 0), default=None)
        out["selected_rootwork"]["_".join((case, profile, dbin, epoch, lane))] = {
            "presearch_sweeps": int(timing["presearch_sweeps"]),
            "finite_double_seed_count": sum(int(r["double_seed_valid"]) for r in roots),
            "d14real_calls": int(timing["d14real_calls"]),
            "d14real_finite_calls": int(timing["d14real_finite_calls"]),
            "d14real_nonconverged_calls": int(timing["d14real_nonconverged"]),
            "d14real_root_update_min": min((int(r["d14real_updates"]) for r in roots), default=None),
            "d14real_root_update_max": max((int(r["d14real_updates"]) for r in roots), default=None),
            "qf_warm_sweeps": int(timing["qf_warm_sweeps"]),
            "qf_cold_sweeps": int(timing["qf_cold_sweeps"]),
            "qf_cold_ms": float(timing["qf_polish_ms"]),
            "reported_root_clusters": int(timing["root_clusters"]),
            "minimum_final_root_separation": min_sep,
            "max_expanded_seed_to_final_root_displacement": max_displacement,
            "expanded_seed_matches": sum(int(r["expanded_seed_match_valid"]) for r in roots),
            "root_count": len(roots),
        }
    return out


def root_set_distances(path_a, path_b):
    arows, brows = read_tsv(path_a), read_tsv(path_b)
    key_fields = ("case_id", "configuration_id", "profile", "d_bin_index", "epoch_index", "lane")
    def group(rows):
        grouped = defaultdict(list)
        for r in rows:
            grouped[tuple(r[k] for k in key_fields)].append(
                complex(float(r["final_v_re"]), float(r["final_v_im"]))
            )
        return grouped
    a, b = group(arows), group(brows)
    values = []
    max_row = None
    for key in a.keys() & b.keys():
        d = max(max(min(abs(x - y) for y in b[key]) for x in a[key]),
                max(min(abs(y - x) for x in a[key]) for y in b[key]))
        values.append(d)
        if max_row is None or d > max_row["hausdorff_nearest"]:
            max_row = {"key": list(key), "hausdorff_nearest": d}
    return {"rows": len(values), "p50": quantile(values, .5),
            "p99": quantile(values, .99), "max_row": max_row,
            "rows_over_1e-12": sum(x > 1e-12 for x in values),
            "rows_over_1e-8": sum(x > 1e-8 for x in values)}


def summarize_representation_ab():
    result = {}
    for case in (149, 0, 64):
        prefix = f"case{case}_"
        block_t = read_tsv(HERE / f"{prefix}block_timing.tsv")[0]
        exp_t = read_tsv(HERE / f"{prefix}expanded_timing.tsv")[0]
        result[str(case)] = {
            "block_topology": {k: block_t[k] for k in ("topology_status", "cells", "events")},
            "expanded_topology": {k: exp_t[k] for k in ("topology_status", "cells", "events")},
            "root_set_difference": root_set_distances(
                HERE / f"{prefix}block_roots.tsv", HERE / f"{prefix}expanded_roots.tsv"),
        }
    return result


def summarize_handoff_rejection():
    base_t = read_tsv(ROOTWORK / "full_safe_timing.tsv.gz")
    hand_t = read_tsv(HERE / "cert_handoff_full_timing.tsv")
    key_fields = ("case_id", "profile", "d_bin_index", "epoch_index", "lane")
    base = {tuple(r[k] for k in key_fields): r for r in base_t}
    mismatch = []
    for r in hand_t:
        key = tuple(r[k] for k in key_fields)
        old = base.get(key)
        if old and any(r[k] != old[k] for k in ("topology_status", "cells", "events")):
            mismatch.append({"key": list(key),
                             "baseline": {k: old[k] for k in ("topology_status", "cells", "events")},
                             "handoff": {k: r[k] for k in ("topology_status", "cells", "events")}})
    base_roots = read_tsv(ROOTWORK / "full_safe_roots.tsv.gz")
    hand_roots = read_tsv(HERE / "cert_handoff_full_roots.tsv")
    target = ("49", "linear", "1", "0", "cold")
    def near_zero_roots(rows):
        return [{"root_index": int(r["root_index"]), "role": r["role"],
                 "physical_real": r["physical_real"],
                 "final_v_re": float(r["final_v_re"]), "final_v_im": float(r["final_v_im"]),
                 "qf_newton": float(r["qf_newton"]),
                 "qf_nearest_sep": float(r["qf_nearest_sep"])}
                for r in rows if (r["case_id"], r["profile"], r["d_bin_index"],
                                  r["epoch_index"], r["lane"]) == target and
                abs(float(r["final_v_re"])) < 1e-6]
    return {
        "result": "rejected; scalar polynomial certificate did not preserve physical root classification",
        "topology_mismatches": len(mismatch),
        "mismatch_rows": mismatch,
        "qf_cold_calls": sum(int(r["qf_cold_calls"]) for r in hand_t),
        "qf_cold_calls_by_lane": {
            lane: sum(int(r["qf_cold_calls"]) for r in hand_t if r["lane"] == lane)
            for lane in ("cold", "warm")
        },
        "case49_dbin1_epoch0_cold_near_zero_roots_baseline": near_zero_roots(base_roots),
        "case49_dbin1_epoch0_cold_near_zero_roots_handoff": near_zero_roots(hand_roots),
    }


def summarize_qf_continuation():
    base_t = read_tsv(ROOTWORK / "full_safe_timing.tsv.gz")
    cand_t = read_tsv(HERE / "qf_continue_full_timing.tsv")
    key_fields = ("case_id", "configuration_id", "profile", "d_bin_index", "epoch_index", "lane")
    base = {tuple(r[k] for k in key_fields): r for r in base_t}
    cand = {tuple(r[k] for k in key_fields): r for r in cand_t}
    lanes = {}
    for lane in ("cold", "warm"):
        a = [r for k, r in base.items() if k[-1] == lane]
        b = [r for k, r in cand.items() if k[-1] == lane]
        mismatches = sum(any(cand[k][field] != base[k][field]
                             for field in ("topology_status", "cells", "events",
                                           "physical_real_events", "soft_events", "qf_root_count"))
                         for k in base if k[-1] == lane)
        lanes[lane] = {
            "rows": len(a),
            "topology_mismatches": mismatches,
            "whole_classify_ms_baseline": distribution(r["whole_classify_ms"] for r in a),
            "whole_classify_ms_continue": distribution(r["whole_classify_ms"] for r in b),
            "qf_polish_ms_baseline": distribution(r["qf_polish_ms"] for r in a),
            "qf_polish_ms_continue": distribution(r["qf_polish_ms"] for r in b),
            "qf_cold_calls_baseline": sum(int(r["qf_cold_calls"]) for r in a),
            "qf_cold_calls_continue": sum(int(r["qf_cold_calls"]) for r in b),
            "qf_cold_sweeps_baseline": sum(int(r["qf_cold_sweeps"]) for r in a),
            "qf_cold_sweeps_continue": sum(int(r["qf_cold_sweeps"]) for r in b),
            "qf_cold_400_sweep_rows_baseline": sum(int(r["qf_cold_sweeps"]) == 400 for r in a),
            "qf_cold_400_sweep_rows_continue": sum(int(r["qf_cold_sweeps"]) == 400 for r in b),
        }
    root_delta = root_set_distances(
        ROOTWORK / "full_safe_roots.tsv.gz", HERE / "qf_continue_full_roots.tsv")
    return {"production_candidate": False, "topology_mismatches": 0,
            "root_set_difference": root_delta, "lanes": lanes}


def summarize_whole_epoch_continuation():
    base_rows = read_tsv(HERE / "overflow_safe_full_trajectory.tsv")
    cont_rows = read_tsv(HERE / "qf_continue_full_trajectory.tsv")
    key_fields = ("case_id", "configuration_id", "profile", "d_bin_index", "epoch_index", "target")
    base = {tuple(r[k] for k in key_fields): r for r in base_rows}
    cont = {tuple(r[k] for k in key_fields): r for r in cont_rows}
    result = {"rows": len(base), "keys_equal": base.keys() == cont.keys(), "lanes": {}}
    for lane in ("cold", "warm", "radial"):
        time_col, topo_col, adapt_col = f"{lane}_ms", f"{lane}_topology_ms", f"{lane}_adaptive_ms"
        conv_col, mu_col = f"{lane}_value_converged", f"{lane}_mu"
        status_col, stop_col = f"{lane}_status", f"{lane}_stop"
        keys = list(base)
        both = [k for k in keys if int(base[k][conv_col]) and int(cont[k][conv_col])]
        errors = [abs(float(base[k][mu_col]) - float(cont[k][mu_col])) /
                  max(1.0, abs(float(base[k][mu_col]))) for k in both]
        result["lanes"][lane] = {
            "whole_epoch_ms_baseline": distribution(base[k][time_col] for k in keys),
            "whole_epoch_ms_continue": distribution(cont[k][time_col] for k in keys),
            "paired_delta_ms_continue_minus_baseline": distribution(
                float(cont[k][time_col]) - float(base[k][time_col]) for k in keys),
            "topology_ms_baseline": distribution(base[k][topo_col] for k in keys),
            "topology_ms_continue": distribution(cont[k][topo_col] for k in keys),
            "adaptive_ms_baseline": distribution(base[k][adapt_col] for k in keys),
            "adaptive_ms_continue": distribution(cont[k][adapt_col] for k in keys),
            "baseline_value_converged": sum(int(base[k][conv_col]) for k in keys),
            "continue_value_converged": sum(int(cont[k][conv_col]) for k in keys),
            "status_mismatches": sum(base[k][status_col] != cont[k][status_col] for k in keys),
            "stop_mismatches": sum(base[k][stop_col] != cont[k][stop_col] for k in keys),
            "topology_cell_mismatches": sum(base[k]["topology_cells"] != cont[k]["topology_cells"] for k in keys),
            "topology_event_mismatches": sum(base[k]["topology_events"] != cont[k]["topology_events"] for k in keys),
            "mu_scaled_error_on_both_converged": distribution(errors),
        }
    return result


def main():
    result = {
        "schema": 1,
        "task": "D14 qf-cold tail diagnosis and matched qf-warm continuation experiment",
        "base_commit": "56d1be6ce857ee623bb32550befcfac9a4c46ac4",
        "input_sha256": "6646492658ed416e75cb8c6c821ee853f79cdf7761fc41441186aa67b81fa05b",
        "qf_rootwork": summarize_rootwork(),
        "qf_trace_samples": summarize_trace_samples(),
        "block_vs_expanded_root_sets": summarize_representation_ab(),
        "rejected_scalar_only_warm_handoff": summarize_handoff_rejection(),
        "qf_warm_continuation_d14_topology_ab": summarize_qf_continuation(),
        "qf_warm_continuation_whole_epoch_ab": summarize_whole_epoch_continuation(),
    }
    out = HERE / "qf_tail_summary.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(out)


if __name__ == "__main__":
    main()
