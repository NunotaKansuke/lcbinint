#!/usr/bin/env python3
"""Paired summary for the D14 overflow-fix full adaptive trajectory A/B."""

import json
import gzip
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


LANES = ("cold", "warm", "radial")
TIMING = {"cold": "cold_ms", "warm": "warm_ms", "radial": "radial_ms"}
TOPOLOGY_TIMING = {"cold": "cold_topology_ms", "warm": "warm_topology_ms",
                   "radial": "radial_topology_ms"}
ADAPTIVE_TIMING = {"cold": "cold_adaptive_ms", "warm": "warm_adaptive_ms",
                   "radial": "radial_adaptive_ms"}
VALUE = {"cold": "cold_mu", "warm": "warm_mu", "radial": "radial_mu"}
CONVERGED = {"cold": "cold_value_converged", "warm": "warm_value_converged",
             "radial": "radial_value_converged"}
STOP = {"cold": "cold_stop", "warm": "warm_stop", "radial": "radial_stop"}
STATUS = {"cold": "cold_status", "warm": "warm_status", "radial": "radial_status"}
KEY = ("case_id", "configuration_id", "profile", "d_bin_index", "epoch_index", "target")


def read_rows(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        lines = f.read().splitlines()
    header = next(line.split() for line in lines if line and not line.startswith("#"))
    rows = []
    for line in lines:
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if fields == header:
            continue
        if len(fields) != len(header):
            raise ValueError(f"{path}: expected {len(header)} fields, got {len(fields)}")
        rows.append(dict(zip(header, fields)))
    return rows


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
    return {"n": len(xs), "p50": quantile(xs, .50), "p90": quantile(xs, .90),
            "p95": quantile(xs, .95), "p99": quantile(xs, .99),
            "max": max(xs) if xs else None}


def keyed(rows):
    out = {}
    for row in rows:
        key = tuple(row[k] for k in KEY)
        if key in out:
            raise ValueError(f"duplicate row key {key}")
        out[key] = row
    return out


def summarize(candidate_path, legacy_path):
    cand_rows = read_rows(candidate_path)
    old_rows = read_rows(legacy_path)
    cand, old = keyed(cand_rows), keyed(old_rows)
    if cand.keys() != old.keys():
        raise ValueError(f"input key mismatch: candidate={len(cand)} baseline={len(old)}")

    lane_summary = {}
    settings_summary = {}
    parity = {"rows": len(cand), "topology_status": 0, "topology_cells": 0,
              "topology_events": 0, "value_converged": {}, "status": {}, "stop": {}}
    status_mismatch_rows = {lane: [] for lane in LANES}

    for lane in LANES:
        t = TIMING[lane]
        mu = VALUE[lane]
        paired_valid = [k for k in cand
                        if int(cand[k][CONVERGED[lane]]) and
                        int(old[k][CONVERGED[lane]])]
        invalid_pairs = [k for k in cand
                         if not (int(cand[k][CONVERGED[lane]]) and
                                 int(old[k][CONVERGED[lane]]))]
        lane_summary[lane] = {
            "whole_epoch_ms": {
                "overflow_safe": distribution(r[t] for r in cand.values()),
                "legacy_double_division": distribution(r[t] for r in old.values()),
                "paired_delta_safe_minus_legacy": distribution(
                    float(cand[k][t]) - float(old[k][t]) for k in cand),
            },
            "topology_ms": {
                "overflow_safe": distribution(r[TOPOLOGY_TIMING[lane]]
                                               for r in cand.values()),
                "legacy_double_division": distribution(r[TOPOLOGY_TIMING[lane]]
                                                        for r in old.values()),
                "paired_delta_safe_minus_legacy": distribution(
                    float(cand[k][TOPOLOGY_TIMING[lane]]) -
                    float(old[k][TOPOLOGY_TIMING[lane]]) for k in cand),
            },
            "adaptive_integration_ms": {
                "overflow_safe": distribution(r[ADAPTIVE_TIMING[lane]]
                                               for r in cand.values()),
                "legacy_double_division": distribution(r[ADAPTIVE_TIMING[lane]]
                                                        for r in old.values()),
                "paired_delta_safe_minus_legacy": distribution(
                    float(cand[k][ADAPTIVE_TIMING[lane]]) -
                    float(old[k][ADAPTIVE_TIMING[lane]]) for k in cand),
            },
            "topology_events": {
                "overflow_safe": distribution(r["topology_events"]
                                               for r in cand.values()),
                "legacy_double_division": distribution(r["topology_events"]
                                                        for r in old.values()),
            },
            "topology_cells": {
                "overflow_safe": distribution(r["topology_cells"]
                                               for r in cand.values()),
                "legacy_double_division": distribution(r["topology_cells"]
                                                        for r in old.values()),
            },
            "mu_difference_on_both_converged_rows": {
                "rows": len(paired_valid),
                "abs": distribution(
                    abs(float(cand[k][mu]) - float(old[k][mu]))
                    for k in paired_valid),
                "scaled_by_max_1_or_abs_legacy_mu": distribution(
                    abs(float(cand[k][mu]) - float(old[k][mu])) /
                    max(1.0, abs(float(old[k][mu]))) for k in paired_valid),
                "max_row": max(
                    ({"key": list(k),
                      "safe_mu": float(cand[k][mu]),
                      "legacy_mu": float(old[k][mu]),
                      "scaled_abs_difference":
                          abs(float(cand[k][mu]) - float(old[k][mu])) /
                          max(1.0, abs(float(old[k][mu])))}
                     for k in paired_valid),
                    key=lambda x: x["scaled_abs_difference"], default=None),
            },
            "rows_not_both_value_converged": [
                {"key": list(k),
                 "safe_converged": bool(int(cand[k][CONVERGED[lane]])),
                 "legacy_converged": bool(int(old[k][CONVERGED[lane]])),
                 "safe_status": cand[k][STATUS[lane]],
                 "legacy_status": old[k][STATUS[lane]],
                 "safe_stop": cand[k][STOP[lane]],
                 "legacy_stop": old[k][STOP[lane]],
                 "safe_mu": float(cand[k][mu]),
                 "legacy_mu": float(old[k][mu])}
                for k in invalid_pairs],
            "overflow_safe": {
                "value_converged": sum(int(r[CONVERGED[lane]]) for r in cand.values()),
                "statuses": dict(Counter(r[STATUS[lane]] for r in cand.values())),
                "stops": dict(Counter(r[STOP[lane]] for r in cand.values())),
            },
            "legacy_double_division": {
                "value_converged": sum(int(r[CONVERGED[lane]]) for r in old.values()),
                "statuses": dict(Counter(r[STATUS[lane]] for r in old.values())),
                "stops": dict(Counter(r[STOP[lane]] for r in old.values())),
            },
        }
        parity["value_converged"][lane] = sum(
            int(cand[k][CONVERGED[lane]]) != int(old[k][CONVERGED[lane]]) for k in cand)
        parity["status"][lane] = sum(cand[k][STATUS[lane]] != old[k][STATUS[lane]] for k in cand)
        parity["stop"][lane] = sum(cand[k][STOP[lane]] != old[k][STOP[lane]] for k in cand)
        status_mismatch_rows[lane] = [
            {"key": list(k), "safe_status": cand[k][STATUS[lane]],
             "legacy_status": old[k][STATUS[lane]],
             "safe_stop": cand[k][STOP[lane]],
             "legacy_stop": old[k][STOP[lane]],
             "safe_converged": bool(int(cand[k][CONVERGED[lane]])),
             "legacy_converged": bool(int(old[k][CONVERGED[lane]]))}
            for k in cand if cand[k][STATUS[lane]] != old[k][STATUS[lane]]]

    for target in sorted({r["target"] for r in cand.values()}, key=float):
        for profile in sorted({r["profile"] for r in cand.values()}):
            keys = [k for k, r in cand.items() if r["target"] == target and r["profile"] == profile]
            label = f"{profile}_rtol_{target}"
            settings_summary[label] = {
                lane: {
                    "overflow_safe_ms": distribution(cand[k][TIMING[lane]] for k in keys),
                    "legacy_double_division_ms": distribution(old[k][TIMING[lane]] for k in keys),
                    "safe_value_converged": sum(int(cand[k][CONVERGED[lane]]) for k in keys),
                    "legacy_value_converged": sum(int(old[k][CONVERGED[lane]]) for k in keys),
                    "status_mismatches": sum(cand[k][STATUS[lane]] != old[k][STATUS[lane]] for k in keys),
                }
                for lane in LANES
            }

    parity["topology_status"] = sum(cand[k]["topology_status"] != old[k]["topology_status"] for k in cand)
    parity["topology_cells"] = sum(cand[k]["topology_cells"] != old[k]["topology_cells"] for k in cand)
    parity["topology_events"] = sum(cand[k]["topology_events"] != old[k]["topology_events"] for k in cand)

    return {
        "schema": 1,
        "task": "Whole-epoch matched A/B of D14 binary64 complex-division overflow fix",
        "provenance": {
            "base_commit": "56d1be6ce857ee623bb32550befcfac9a4c46ac4",
            "input_sha256": "6646492658ed416e75cb8c6c821ee853f79cdf7761fc41441186aa67b81fa05b",
            "input_trajectory_count": 1804,
            "input_epoch_rows": 7216,
            "targets": [1e-3, 1e-4],
            "value_only": True,
            "gradient_policy": "None",
            "value_atol": 1e-16,
            "fold_maps": True,
            "reuse_samples": True,
            "repeats_per_row": 3,
            "reported_timing": "median of 3 repeats per row",
            "machine": "Intel Xeon Gold 6530",
            "cpu_affinity": "taskset -c 0-7",
            "compiler": "GCC 11.5.0",
            "compile_flags": "-O3 -march=native -funroll-loops -ffp-contract=fast -fno-math-errno -DNDEBUG -std=gnu++17",
            "runner_source_sha256": "c32f57e9167aa3e6584b37203f979fb80ff186cdf7cbf81490d2f1e0b022b11f",
            "safe_binary_sha256": "1940df8532e2718ee1267576784a44374f200eab0c3dd80883882fbb2704c5e5",
            "legacy_binary_sha256": "9757e809a012737145123e32f374a642ab8c7fdff9127919186c5213c0db164a",
            "legacy_delta": "compile-time HOLO_D14_FORCE_LEGACY_DOUBLE_DIV=1; restores only the old binary64 complex quotient",
            "environment": {
                "HOLO_D14_LOCAL_PAIRS": "1",
                "HOLO_D14_ACTIVE_PRESEARCH": "1",
                "HOLO_D14_ACTIVE_TOL": "1e-12",
                "HOLO_D14_ACTIVE_PATIENCE": "2"
            }
        },
        "rows": len(cand),
        "row_key": list(KEY),
        "lanes": lane_summary,
        "by_profile_and_tolerance": settings_summary,
        "parity_mismatches": parity,
        "status_mismatch_rows": status_mismatch_rows,
    }


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: summarize_overflow_whole_ab.py SAFE.tsv LEGACY.tsv OUTPUT.json")
    result = summarize(sys.argv[1], sys.argv[2])
    Path(sys.argv[3]).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
