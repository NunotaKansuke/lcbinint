#!/usr/bin/env python3
"""Paired fresh-seed timing and deterministic equivalence audit."""
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent / "seeded_paper_qrho_20261013"
KEY = ("case_id", "profile", "d_bin_index", "epoch_index", "rtol")
STAGES = ("seed_ms", "planning_ms", "integration_ms", "full_ms")


def read(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert all(None not in row and None not in row.values() for row in rows), path
    return rows


def medians(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in KEY)].append(row)
    result = {}
    for key, records in groups.items():
        row = dict(records[0])
        for field in STAGES:
            row[field] = statistics.median(float(r[field]) for r in records)
        result[key] = row
    return result


vbm = {}
for row in read(PAPER / "vbm_timing_reference.tsv"):
    key = tuple(row[k] for k in KEY[:-1]) + (str(float(row["target"])),)
    vbm[key] = float(row["vbm_ms"])
summary = {"timing_contract": "fresh magnifiers/workspace per repetition; seed included; caustic construction excluded; one thread", "groups": []}
for direction in ("accepted", "reverse"):
    before = medians(read(HERE / f"{direction}_before.tsv"))
    after = medians(read(HERE / f"{direction}_indexed.tsv"))
    assert before.keys() == after.keys()
    for profile in ("uniform", "linear"):
        for rtol in (1e-3, 1e-4):
            for low_only in (True, False):
                keys = [k for k, r in before.items() if r["profile"] == profile
                        and float(r["rtol"]) == rtol
                        and (not low_only or float(r["reference_1e-6"]) < 1.1)]
                group = {"direction": direction, "profile": profile, "rtol": rtol,
                         "domain": "A_lt_1.1" if low_only else "selected_all", "inputs": len(keys)}
                for label, data in (("before", before), ("after", after)):
                    group[label + "_us"] = {s: 1000*statistics.median(data[k][s] for k in keys) for s in STAGES}
                    ratios = [vbm[k] / data[k]["full_ms"] for k in keys]
                    group[label + "_vbm_over_full_median"] = statistics.median(ratios)
                    group[label + "_win_count"] = sum(r > 1 for r in ratios)
                group["paired_full_reduction_median"] = statistics.median(
                    1-after[k]["full_ms"]/before[k]["full_ms"] for k in keys)
                summary["groups"].append(group)

before = read(HERE / "full_before.tsv")
after = read(HERE / "full_indexed.tsv")
assert len(before) == len(after) == 14432
assert [tuple(r[k] for k in KEY) for r in before] == [tuple(r[k] for k in KEY) for r in after]
fields = ("mu", "estimated_abs_error", "value_converged", "support_proven", "seed_count", "nodes", "evaluations", "stop", "value_stop", "numerical_status")
summary["full_numerical_audit"] = {"rows": len(after), "changed": {
    k: sum(a[k] != b[k] for a, b in zip(after, before)) for k in fields}}
summary["full_numerical_audit"]["baseline_over_target_converged_rows"] = sum(
    r["value_converged"] == "1" and float(r["relative_error_reference"]) > float(r["rtol"]) for r in before)
summary["seed_equivalence"] = {
    "inputs": len((HERE / "audit_all_before.txt").read_text().splitlines()),
    "hex_positions_order_support_equal": (HERE / "audit_all_before.txt").read_bytes() == (HERE / "audit_all_indexed.txt").read_bytes()}
(HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
for group in summary["groups"]:
    if group["domain"] == "A_lt_1.1" and group["rtol"] == 1e-3:
        print(json.dumps(group))
print(json.dumps(summary["full_numerical_audit"]))
print(json.dumps(summary["seed_equivalence"]))
