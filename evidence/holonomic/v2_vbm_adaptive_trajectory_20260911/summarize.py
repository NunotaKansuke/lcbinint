#!/usr/bin/env python3
"""Summarize the value-only adaptive cold/warm trajectory benchmark."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results.tsv"
INPUT = ROOT / "input_snapshot.tsv"
OLD_PURE = ROOT.parent / "v2_vbm_adaptive_qrho_20260911" / "v2_results.tsv"
LANES = ("cold", "warm", "radial")
PROFILES = ("uniform", "linear")
TARGETS = (1e-3, 1e-4)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path: Path):
    header = None
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split()
        if header is None:
            header = fields
            continue
        if len(fields) != len(header):
            raise ValueError(f"{path}: {len(fields)} fields, expected {len(header)}")
        rows.append(dict(zip(header, fields)))
    if not header:
        raise ValueError(f"no header in {path}")
    return header, rows


def value(row, key):
    return float(row[key])


def ints(rows, key):
    return sum(int(r[key]) for r in rows)


def percentile(values, p):
    if not values:
        return None
    xs = sorted(float(x) for x in values)
    x = p / 100.0 * (len(xs) - 1)
    lo = int(x)
    hi = min(lo + 1, len(xs) - 1)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - x) + xs[hi] * (x - lo)


def distribution(values):
    return {
        "n": len(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": percentile(values, 100),
    }


def filter_rows(rows, profile=None, target=None, steady=None):
    out = rows
    if profile is not None:
        out = [r for r in out if r["profile"] == profile]
    if target is not None:
        out = [r for r in out if abs(value(r, "target") - target) < 1e-15]
    if steady is True:
        out = [r for r in out if int(r["trajectory_pos"]) > 0]
    elif steady is False:
        out = [r for r in out if int(r["trajectory_pos"]) == 0]
    return out


def timing(rows, lane):
    names = {
        "whole": f"{lane}_ms",
        "topology": f"{lane}_topology_ms",
        "adaptive": f"{lane}_adaptive_ms",
        "setup": f"{lane}_setup_ms",
        "physical": f"{lane}_physical_ms",
        "estimator": f"{lane}_estimator_ms",
        "scheduler": f"{lane}_scheduler_ms",
    }
    out = {}
    for name, column in names.items():
        out[name] = distribution([value(r, column) for r in rows])
    gaps = []
    for r in rows:
        whole = value(r, names["whole"])
        stages = sum(value(r, names[x]) for x in ("topology", "setup", "physical", "estimator", "scheduler"))
        gaps.append(whole - stages)
    out["whole_minus_explicit_stages"] = distribution(gaps)
    return out


def lane_quality(rows, lane):
    mu = f"{lane}_mu"
    converged = f"{lane}_value_converged"
    status = f"{lane}_status"
    errors = []
    for r in rows:
        ref = value(r, "reference")
        errors.append(abs(value(r, mu) - ref) / max(abs(ref), 1e-300))
    return {
        "rows": len(rows),
        "value_converged": ints(rows, converged),
        "stop_counts": counts(rows, f"{lane}_stop"),
        "status_counts": counts(rows, status),
        "relative_error_to_input_reference": distribution(errors),
    }


def counts(rows, key):
    out = defaultdict(int)
    for r in rows:
        out[r[key]] += 1
    return dict(sorted(out.items()))


def parity(rows, left, right):
    ref_errors = []
    abs_errors = []
    status_mismatch = 0
    convergence_mismatch = 0
    for r in rows:
        scale = max(abs(value(r, "reference")), 1e-300)
        abs_errors.append(abs(value(r, f"{left}_mu") - value(r, f"{right}_mu")) / scale)
        ref_errors.append(abs(value(r, f"{left}_mu") - value(r, f"{right}_mu")))
        status_mismatch += r[f"{left}_status"] != r[f"{right}_status"]
        convergence_mismatch += r[f"{left}_value_converged"] != r[f"{right}_value_converged"]
    return {
        "relative_to_reference_scale": distribution(abs_errors),
        "absolute": distribution(ref_errors),
        "status_mismatch": status_mismatch,
        "value_convergence_mismatch": convergence_mismatch,
    }


def trajectory_sums(rows):
    grouped = defaultdict(lambda: defaultdict(float))
    for r in rows:
        key = (r["trajectory_id"], r["profile"], r["target"])
        for lane in LANES:
            grouped[key][lane] += value(r, f"{lane}_ms")
    out = {}
    for lane in LANES:
        vals = [g[lane] for g in grouped.values()]
        out[lane] = distribution(vals)
    out["by_profile_target"] = {}
    for profile in PROFILES:
        for target in TARGETS:
            key = f"{profile}/1e-{3 if target == 1e-3 else 4}"
            vals = [g["warm"] for k, g in grouped.items() if k[1] == profile and abs(float(k[2]) - target) < 1e-15]
            out["by_profile_target"][key] = {"warm_sum_ms": distribution(vals)}
    return out


def old_pure_stats():
    if not OLD_PURE.exists():
        return None
    old_columns = [
        "case_id", "configuration_id", "profile", "d_bin_index", "epoch_index", "target",
        "s", "q", "rho", "x", "y", "time", "u", "X", "reference", "topology_status",
        "topology_status_code", "topology_cells", "topology_uncertain_cells", "physical_events",
        "v2_mu", "value_error", "value_converged", "stop", "value_stop", "numerical_status",
        "status_code", "ms", "nodes", "evaluations", "reused_nodes", "panels", "splits",
        "setup_ms", "physics_ms", "estimator_ms", "scheduler_ms", "setup_event_ms",
        "event_topology_reuses", "event_radius_reuses", "event_qf_refinements",
    ]
    rows = []
    for line in OLD_PURE.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != len(old_columns):
            raise ValueError(f"old pure row has {len(fields)} fields")
        rows.append(dict(zip(old_columns, fields)))
    return {
        "rows": len(rows),
        "ms": distribution([value(r, "ms") for r in rows]),
        "by_profile_target": {
            f"{profile}/1e-{3 if target == 1e-3 else 4}": distribution([
                value(r, "ms") for r in rows
                if r["profile"] == profile and abs(value(r, "target") - target) < 1e-15
            ])
            for profile in PROFILES for target in TARGETS
        },
    }


def timing_sanity(rows):
    out = {}
    for lane in LANES:
        timing_columns = [
            f"{lane}_ms", f"{lane}_topology_ms", f"{lane}_adaptive_ms",
            f"{lane}_setup_ms", f"{lane}_physical_ms", f"{lane}_estimator_ms",
            f"{lane}_scheduler_ms",
        ]
        nonfinite = sum(
            not all(math.isfinite(value(r, c)) for c in timing_columns)
            for r in rows
        )
        negative = sum(
            any(value(r, c) < 0.0 for c in timing_columns) for r in rows
        )
        whole_lt_topology = sum(
            value(r, f"{lane}_ms") < value(r, f"{lane}_topology_ms")
            for r in rows
        )
        stage_gaps = []
        for r in rows:
            explicit = sum(
                value(r, f"{lane}_{part}_ms")
                for part in ("topology", "setup", "physical", "estimator", "scheduler")
            )
            stage_gaps.append(value(r, f"{lane}_ms") - explicit)
        if nonfinite or negative or whole_lt_topology:
            raise ValueError(
                f"timing sanity failed for {lane}: nonfinite={nonfinite} "
                f"negative={negative} whole_lt_topology={whole_lt_topology}"
            )
        out[lane] = {
            "nonfinite_rows": nonfinite,
            "negative_rows": negative,
            "whole_less_than_topology_rows": whole_lt_topology,
            "whole_minus_explicit_stages": distribution(stage_gaps),
        }
    return out


def main():
    header, rows = read_rows(RESULTS)
    if len(rows) != 14432:
        raise ValueError(f"expected 14432 measured rows, got {len(rows)}")
    trajectories = sorted({r["trajectory_id"] for r in rows})
    summary = {
        "metadata": {
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT.parent.parent.parent, text=True).strip(),
            "result_sha256": sha256(RESULTS),
            "input_sha256": sha256(INPUT),
            "result_columns": len(header),
            "rows": len(rows),
            "trajectories": len(trajectories),
            "epochs_per_trajectory": 4,
            "targets": [1e-3, 1e-4],
            "profiles": list(PROFILES),
            "repetitions": 3,
            "warm_policy": "L2 D14 warm seed; allow_topology_reuse=false; l2_drift=1e18",
            "radial_policy": "topology/D14 prebuilt outside timer; flux_adaptive_integrate only",
            "cold_policy": "epoch_adaptive; classify_cells/D14 inside timer for every epoch",
        },
        "coverage": {
            lane: lane_quality(rows, lane) for lane in LANES
        },
        "timing": {
            lane: {
                "all_epochs": timing(rows, lane),
                "steady_epochs": timing(filter_rows(rows, steady=True), lane),
                "first_epochs": timing(filter_rows(rows, steady=False), lane),
            }
            for lane in LANES
        },
        "by_profile_target": {},
        "parity": {
            "cold_vs_radial": parity(rows, "cold", "radial"),
            "warm_vs_radial": parity(rows, "warm", "radial"),
            "cold_vs_warm": parity(rows, "cold", "warm"),
        },
        "timing_sanity": timing_sanity(rows),
        "warm_reuse_mix": {
            key: ints(rows, key) for key in
            ("warm_l1", "warm_l2", "warm_l3", "warm_rescreen_fail", "warm_seed_used")
        },
        "trajectory_sums_ms": trajectory_sums(rows),
        "prior_pure_kernel": old_pure_stats(),
    }
    for profile in PROFILES:
        for target in TARGETS:
            name = f"{profile}/1e-{3 if target == 1e-3 else 4}"
            selected = filter_rows(rows, profile=profile, target=target)
            summary["by_profile_target"][name] = {
                "rows": len(selected),
                "all_epochs": {
                    "coverage": {lane: lane_quality(selected, lane) for lane in LANES},
                    "timing": {lane: timing(selected, lane) for lane in LANES},
                },
                "steady_epochs": {
                    "rows": len(filter_rows(selected, steady=True)),
                    "timing": {
                        lane: timing(filter_rows(selected, steady=True), lane)
                        for lane in LANES
                    },
                },
            }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
