#!/usr/bin/env python3
"""Build the machine-readable Phase D14 precision/warm evidence index."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TRAJ = ROOT / "trajectory"


def d14_ab(path: Path):
    text = path.read_text()
    out = {"path": str(path)}
    for name in ("base", "strt"):
        match = re.search(
            rf"^  {name} : median\s+([0-9.eE+-]+)\s+p90\s+([0-9.eE+-]+)\s+"
            rf"p99\s+([0-9.eE+-]+)\s+max\s+([0-9.eE+-]+)",
            text,
            re.MULTILINE,
        )
        if match:
            out[name] = {
                "median_ms": float(match.group(1)),
                "p90_ms": float(match.group(2)),
                "p99_ms": float(match.group(3)),
                "max_ms": float(match.group(4)),
            }
    match = re.search(r"speedup\s+median\s+([0-9.eE+-]+)x\s+p90\s+([0-9.eE+-]+)x\s+p99\s+([0-9.eE+-]+)x", text)
    if match:
        out["speedup_base_over_block"] = {
            "median": float(match.group(1)),
            "p90": float(match.group(2)),
            "p99": float(match.group(3)),
        }
    match = re.search(r"parity failures:\s+(\d+) / (\d+)", text)
    if match:
        out["parity_failures"] = int(match.group(1))
        out["parity_cases"] = int(match.group(2))
    match = re.search(r"root-set rel diff .*:\s+([0-9.eE+-]+)", text)
    if match:
        out["worst_root_set_relative_difference"] = float(match.group(1))
    match = re.search(r"worst __float128 residual\s+base\s+([0-9.eE+-]+)\s+strt\s+([0-9.eE+-]+)", text)
    if match:
        out["worst_qf_residual"] = {
            "base": float(match.group(1)),
            "block": float(match.group(2)),
        }
    match = re.search(r"whole radial_events .*: median\s+([0-9.eE+-]+)\s+p90\s+([0-9.eE+-]+)\s+p99\s+([0-9.eE+-]+)\s+max\s+([0-9.eE+-]+)", text)
    if match:
        out["whole_radial_events_ms"] = [float(x) for x in match.groups()]
    match = re.search(r"full value\+5-Jac epoch .*: median\s+([0-9.eE+-]+)\s+p90\s+([0-9.eE+-]+)\s+p99\s+([0-9.eE+-]+)\s+max\s+([0-9.eE+-]+)", text)
    if match:
        out["whole_value_plus_5jac_ms"] = [float(x) for x in match.groups()]
    return out


def profile_lines(path: Path):
    records = []
    for line in path.read_text().splitlines():
        if line.startswith("PROFILE ") or line.startswith("  D14 calls=") or line.startswith("  D14Real calls="):
            records.append(line)
    return records


def main():
    trajectory = json.loads((TRAJ / "summary.json").read_text())
    comparison = json.loads((TRAJ / "comparison.json").read_text())
    ctest = (ROOT / "micro" / "ctest_all.log").read_text()
    summary = {
        "metadata": {
            "spec": "docs/holonomic/d14_precision_and_warm_design_20260911_ja.md",
            "measured_tree_head": trajectory["metadata"]["git_head"],
            "baseline_tree_head": comparison["baseline"]["git_head"],
            "input_rows": trajectory["metadata"]["rows"],
            "trajectory_count": trajectory["metadata"]["trajectories"],
            "repetitions": trajectory["metadata"]["repetitions"],
            "precision_mode": "D14Real block kernel default, qf residual/completeness certificate retained",
            "direct_warm_shortcut": "disabled after A/B rejection",
        },
        "tests": {
            "ctest_all_passed": "100% tests passed" in ctest,
            "ctest_tail": ctest.splitlines()[-3:],
        },
        "micro_ab": {
            "legacy": d14_ab(ROOT / "micro" / "d14_structure_legacy.stderr"),
            "block": d14_ab(ROOT / "micro" / "d14_structure_block.stderr"),
            "profile_block_lines": profile_lines(ROOT / "micro" / "v2_profile_block.stderr"),
            "profile_legacy_lines": profile_lines(ROOT / "micro" / "v2_profile_legacy.stderr"),
        },
        "trajectory": {
            "summary_path": "trajectory/summary.json",
            "comparison_path": "trajectory/comparison.json",
            "coverage": trajectory["coverage"],
            "timing": trajectory["timing"],
            "timing_sanity": trajectory["timing_sanity"],
            "warm_reuse_mix": trajectory["warm_reuse_mix"],
        },
        "raw": {
            "trajectory_results": "trajectory/results.tsv",
            "trajectory_input": "trajectory/input_snapshot.tsv",
            "micro_profile_block": "micro/v2_profile_block.stderr",
            "micro_profile_legacy": "micro/v2_profile_legacy.stderr",
            "micro_d14_block": "micro/d14_structure_block.stderr",
            "micro_d14_legacy": "micro/d14_structure_legacy.stderr",
            "ctest": "micro/ctest_all.log",
        },
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
