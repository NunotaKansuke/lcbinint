#!/usr/bin/env python3
"""Report the cache-warm, one-epoch finite-source kernel comparison."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from report_near_caustic_grid_vs_vbm import _figure


PROFILES = ("uniform", "linear")
# The public comparison uses only the two externally specified tolerances.
TARGETS = (1.0e-3, 1.0e-4)
BANDS = (
    ("inner", -1.0e-12, 0.8),
    ("tangent", 0.8, 1.05),
    ("outer-near", 1.05, 2.01),
)


def _values(rows):
    return [
        float(value)
        for row in rows
        for value in row.get("ratios_vbm_over_lcbinint", ())
        if value is not None and math.isfinite(float(value))
    ]


def _stats(values):
    values = np.asarray(values, dtype=float)
    if not values.size:
        return {"count": 0}
    return {
        "count": int(values.size),
        "win_rate": float(np.mean(values > 1.0)),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }


def _summary(rows):
    ratios = _values(rows)
    statuses = [
        status for row in rows for status in row.get("ratio_status", ())
    ]
    counts = Counter(statuses)
    grid_wins = sum(value > 1.0 for value in ratios)
    classified = len(ratios)
    bins = [
        int(value)
        for row in rows
        for value in row.get("chosen_nbin", ())
        if value is not None
    ]
    return {
        "jobs": len(rows),
        "points": len(statuses),
        "measured": len(ratios),
        "grid_wins": grid_wins,
        "vbm_wins": classified - grid_wins,
        "unresolved": len(statuses) - classified,
        "grid_win_rate": None if not classified else grid_wins / classified,
        "ratio": _stats(ratios),
        "nbin": _stats(bins),
        "status_counts": dict(counts),
    }


def _markdown(payload, summaries, bands, include_figure):
    filters = payload.get("filters", {})
    extension = payload.get("build_extension", "unknown")
    lines = [
        "# Pure one-epoch finite-source kernel comparison",
        "",
        "This is the cache-warm integrator comparison. It is deliberately not a",
        "`LightCurve` total-call benchmark.",
        "",
        "## Timing definition",
        "",
        "- lcbinint uses the stored minimum Nbin from the warm-up/speed-discovery",
        "  corpus for each requested epsilon; no Nbin search is mixed into the",
        "  timing run.",
        "- For lcbinint, two identical source positions are evaluated inside one",
        "  native `_evaluate_preplanned_xy` call. Source `(x, y)` is supplied",
        "  directly in the internal lens frame. The first position builds the",
        "  LensModel and caustic cache; only the second position's native",
        "  `seconds` value is used.",
        "- The preplanned point-source magnification used to build image seeds is",
        "  reused as the Cartesian walk hint, so its duplicate point-lens solve",
        "  is excluded from the timed epoch.",
        "- VBM is warmed once at `RelTol=target`, then the direct finite-source",
        "  call wall time is measured.",
        "- `R = t_VBM / t_LCB-in`; `R > 1` means lcbinint is faster.",
        "",
        "## Conditions",
        "",
        f"- q filter: `{filters.get('q_min')}`; d/rho filter: `< {filters.get('d_max')}`.",
        "- External tolerances: `1e-3` and `1e-4` only.",
        "- Only rows whose production auto route was finite-source grid were",
        "  selected; point-source/hexadecapole/source-plane rows are excluded.",
        f"- Build extension: `{extension}`.",
        "",
        "## Overall result",
        "",
        "| profile | target | jobs | points | measured | lcbinint wins | VBM wins | unresolved | lcbinint win rate | median R | p10 | p90 | median Nbin |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for profile in PROFILES:
        for target in TARGETS:
            item = summaries[profile][target]
            ratio = item["ratio"]
            nbin = item["nbin"]
            lines.append(
                f"| {profile} | `{target:g}` | {item['jobs']} | {item['points']} | "
                f"{item['measured']} | {item['grid_wins']} | {item['vbm_wins']} | "
                f"{item['unresolved']} | "
                f"{item['grid_win_rate']:.1%} | "
                f"{ratio.get('median', float('nan')):.3f} | "
                f"{ratio.get('p10', float('nan')):.3f} | "
                f"{ratio.get('p90', float('nan')):.3f} | "
                f"{nbin.get('median', float('nan')):.0f} |"
            )
    lines += [
        "",
        "## By d/rho",
        "",
        "| profile | target | region | measured | lcbinint wins | VBM wins | unresolved | win rate | median R |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for profile in PROFILES:
        for target in TARGETS:
            for label, low, high in BANDS:
                selected = [
                    row for row in payload["results"]
                    if row.get("status", "completed") == "completed"
                    and row["profile"] == profile
                    and abs(float(row["target"]) - target) < 1.0e-15
                    and low <= float(row["d_over_rho"]) < high
                ]
                item = _summary(selected)
                lines.append(
                    f"| {profile} | `{target:g}` | {label} | "
                    f"{item['measured']} | {item['grid_wins']} | {item['vbm_wins']} | "
                    f"{item['unresolved']} | "
                    f"{item['grid_win_rate']:.1%} | "
                    f"{item['ratio'].get('median', float('nan')):.3f} |"
                )
    if include_figure:
        lines += [
            "",
            "Figure: `figures/pure_kernel_speed_ratio.png`.",
        ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-figure", action="store_true")
    args = parser.parse_args()
    payload = json.loads(args.results.read_text())
    summaries = {
        profile: {
            target: _summary([
                row for row in payload["results"]
                if row.get("status", "completed") == "completed"
                and row["profile"] == profile
                and abs(float(row["target"]) - target) < 1.0e-15
            ])
            for target in TARGETS
        }
        for profile in PROFILES
    }
    args.output.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    (args.output / "REPORT_pure_kernel.md").write_text(
        _markdown(payload, summaries, BANDS, not args.skip_figure)
    )
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2))
    if not args.skip_figure:
        _figure(payload, figure_dir / "pure_kernel_speed_ratio")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
