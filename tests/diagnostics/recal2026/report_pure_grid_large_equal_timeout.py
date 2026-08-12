#!/usr/bin/env python3
"""Report the large pure-grid benchmark with shared per-point timeouts."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from report_near_caustic_grid_vs_vbm import _figure


PROFILES = ("uniform", "linear")
TARGETS = (1.0e-2, 1.0e-3, 1.0e-4)


def _stats(payload, profile, target):
    rows = [
        row for row in payload["results"]
        if row["profile"] == profile
        and abs(float(row["target"]) - target) < 1.0e-15
    ]
    ratios = [
        float(value) for row in rows
        for value in row.get("ratios_vbm_over_lcbinint", ())
        if value is not None and math.isfinite(float(value))
    ]
    statuses = [
        status for row in rows for status in row.get("ratio_status", ())
    ]
    counts = Counter(statuses)
    measured_grid_wins = sum(value > 1.0 for value in ratios)
    measured_vbm_wins = len(ratios) - measured_grid_wins
    grid_wins = measured_grid_wins + counts["grid_win_vbm_timeout"]
    vbm_wins = measured_vbm_wins + counts["vbm_win_grid_timeout"]
    classified = grid_wins + vbm_wins
    total = len(statuses)
    return {
        "jobs": len(rows),
        "total_points": total,
        "measured": counts["measured"],
        "grid_wins": grid_wins,
        "vbm_wins": vbm_wins,
        "both_timeout": counts["both_timeout"],
        "unresolved": total - classified - counts["both_timeout"],
        "grid_win_rate_classified": None if not classified else grid_wins / classified,
        "grid_win_rate_all": None if not total else grid_wins / total,
        "median_ratio": None if not ratios else float(np.median(ratios)),
        "p10_ratio": None if not ratios else float(np.percentile(ratios, 10)),
        "p90_ratio": None if not ratios else float(np.percentile(ratios, 90)),
        "status_counts": dict(counts),
    }


def _markdown(payload, summary):
    timeout = payload.get("point_timeout")
    lines = [
        "# Large pure-grid VBM versus LCB-in comparison",
        "",
        "This benchmark contains only blocks whose recorded production route",
        "was exactly finite-source `grid`. The explicit LCB-in Cartesian/Polar",
        "grid is compared with direct VBM finite-source integration.",
        "",
        "## Conditions",
        "",
        "- `q >= 1e-4`; `d/rho < 2.01`; all ten strata from `0` through `2`.",
        "- Uniform and linear limb darkening (`c=0.5`).",
        "- Targets `1e-2`, `1e-3`, and `1e-4`.",
        f"- Shared per-reference-epoch timeout: `{timeout:g}` seconds; no job-level timeout.",
        "- A timeout is retained in the denominator. Only both-timeout or",
        "  otherwise unresolved points lack a directional winner.",
        "- `R = t_VBM / t_lcbinint`; `R > 1` means LCB-in is faster.",
        "",
        "## Speed result",
        "",
        "| profile | target | jobs | all points | measured | grid wins | VBM wins | both timeout | unresolved | grid win rate / classified | grid win rate / all | median R | p10 | p90 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for profile in PROFILES:
        for target in TARGETS:
            item = summary[profile][str(target)]
            lines.append(
                f"| {profile} | `{target:g}` | {item['jobs']} | "
                f"{item['total_points']} | {item['measured']} | "
                f"{item['grid_wins']} | {item['vbm_wins']} | "
                f"{item['both_timeout']} | {item['unresolved']} | "
                f"{item['grid_win_rate_classified']:.1%} | "
                f"{item['grid_win_rate_all']:.1%} | "
                f"{item['median_ratio']:.3f} | {item['p10_ratio']:.3f} | "
                f"{item['p90_ratio']:.3f} |"
            )
    lines += [
        "",
        "The associated `A_point` and `A_finite` coverage is reported in",
        "`apoint_distribution/` and `afinite_distribution/`.",
        "",
        "Figure: `figures/pure_grid_large_speed_ratio.png`.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.results.read_text())
    summary = {profile: {
        str(target): _stats(payload, profile, target)
        for target in TARGETS
    } for profile in PROFILES}

    args.output.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    (args.output / "REPORT_pure_grid_large_equal_timeout.md").write_text(
        _markdown(payload, summary)
    )
    _figure(payload, figure_dir / "pure_grid_large_speed_ratio")
    (args.output / "speed_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
