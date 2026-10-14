#!/usr/bin/env python3
"""Compare the D14 candidate trajectory run with the preceding V2 run."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CURRENT = ROOT / "summary.json"
BASELINE = ROOT / "baseline_summary_26593fa.json"
OUT = ROOT / "comparison.json"
LANES = ("cold", "warm", "radial")
PROFILES = ("uniform", "linear")
TARGETS = ("1e-3", "1e-4")


def speed_stats(summary, profile, target, lane, scope):
    key = f"{profile}/{target}"
    return summary["by_profile_target"][key][scope]["timing"][lane]["whole"]


def ratio(old, new):
    return {
        key: old[key] / new[key]
        for key in ("p50", "p90", "p95", "p99", "max")
    }


def main():
    current = json.loads(CURRENT.read_text())
    baseline = json.loads(BASELINE.read_text())
    out = {
        "baseline": {
            "path": str(BASELINE),
            "git_head": baseline["metadata"]["git_head"],
            "rows": baseline["metadata"]["rows"],
        },
        "candidate": {
            "path": str(CURRENT),
            "git_head": current["metadata"]["git_head"],
            "rows": current["metadata"]["rows"],
        },
        "speedup_baseline_over_candidate": {},
    }
    for profile in PROFILES:
        for target in TARGETS:
            condition = f"{profile}/{target}"
            out["speedup_baseline_over_candidate"][condition] = {}
            for scope in ("all_epochs", "steady_epochs"):
                out["speedup_baseline_over_candidate"][condition][scope] = {}
                for lane in LANES:
                    old = speed_stats(baseline, profile, target, lane, scope)
                    new = speed_stats(current, profile, target, lane, scope)
                    out["speedup_baseline_over_candidate"][condition][scope][lane] = {
                        "baseline_ms": old,
                        "candidate_ms": new,
                        "speedup": ratio(old, new),
                        "percent_faster_at_p50": 100.0 * (old["p50"] / new["p50"] - 1.0),
                    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
