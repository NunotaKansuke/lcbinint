#!/usr/bin/env python3
"""Summarise and plot the near-caustic grid/VBM benchmark.

The benchmark itself measures the two explicit finite-source integrators.  This
post-processing step keeps that result separate from the production route
audit: point-source and hexadecapole answers are reported, but never counted as
finite-source grid wins.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


TARGETS = (1.0e-2, 1.0e-3, 1.0e-4)
CHEAP = frozenset(("point_source", "hexadecapole"))


def _route(methods):
    methods = set(methods or ())
    has_grid = any(name.startswith("inverse_ray_") for name in methods)
    has_source = "source_plane_quadrature" in methods
    has_cheap = bool(methods & CHEAP)
    if has_cheap and has_grid:
        return "grid+point/hex"
    if has_cheap:
        return "point/hex"
    if has_grid and has_source:
        return "grid+source-plane"
    if has_source:
        return "source-plane"
    if has_grid:
        return "grid"
    return "+".join(sorted(methods)) or "unknown"


def _finite(values):
    return [float(value) for value in values
            if value is not None and math.isfinite(float(value))]


def _stats(values):
    values = np.asarray(_finite(values), dtype=float)
    if not values.size:
        return {"n": 0}
    return {
        "n": int(values.size),
        "win_rate": float(np.mean(values > 1.0)),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }


def _load_route_rows(path):
    rows = {}
    for block in sorted(Path(path).glob("block-*.json")):
        payload = json.loads(block.read_text())
        for row in payload.get("rows", ()):
            key = (int(row["case_id"]), row["profile"],
                   round(float(row["intended_distance_factor"]), 12))
            rows[key] = row
    return rows


def _result_route(result, route_rows):
    key = (int(result["case_id"]), result["profile"],
           round(float(result["d_over_rho"]), 12))
    source = route_rows.get(key)
    if source is None:
        return None
    target = float(result["target"])
    auto = [entry for entry in source.get("engines", ())
            if entry.get("engine") == "lcbinint_auto"
            and abs(float(entry.get("knob", -1.0)) - target) < 1.0e-15]
    return _route(auto[0].get("methods")) if auto else None


def _filtered_payload(payload, route_rows, route_filter):
    if route_filter == "all":
        return payload
    filtered = dict(payload)
    filtered["results"] = [
        result for result in payload["results"]
        if result.get("status", "completed") == "completed"
        and _result_route(result, route_rows) == route_filter
    ]
    filtered["route_filter"] = route_filter
    return filtered


def _summaries(payload, route_rows):
    summaries = {}
    route_summaries = {}
    conditioned = {}
    for profile in ("uniform", "linear"):
        summaries[profile] = {}
        route_summaries[profile] = {}
        conditioned[profile] = {}
        for target in TARGETS:
            rows = [row for row in payload["results"]
                    if row["profile"] == profile
                    and abs(float(row["target"]) - target) < 1.0e-15]
            ratios = [value for row in rows
                      for value in row["ratios_vbm_over_lcbinint"]]
            bins = [value for row in rows for value in row["chosen_nbin"]]
            summaries[profile][target] = {
                "blocks": len(rows),
                "censored": sum(
                    row.get("status", "completed") != "completed"
                    for row in rows
                ),
                "ratio": _stats(ratios),
                "nbin": _stats(bins),
            }

            routes = Counter()
            conditioned_values = defaultdict(list)
            for result in rows:
                if result.get("status", "completed") != "completed":
                    continue
                key = (int(result["case_id"]), result["profile"],
                       round(float(result["d_over_rho"]), 12))
                source = route_rows.get(key)
                if source is None:
                    continue
                auto = [entry for entry in source.get("engines", ())
                        if entry.get("engine") == "lcbinint_auto"
                        and abs(float(entry.get("knob", -1.0)) - target)
                        < 1.0e-15]
                if auto:
                    route = _route(auto[0].get("methods"))
                    routes[route] += 1
                    conditioned_values[route].extend(
                        value for value in result["ratios_vbm_over_lcbinint"]
                        if value is not None and math.isfinite(float(value))
                    )
            route_summaries[profile][target] = dict(routes)
            conditioned[profile][target] = {
                route: _stats(values)
                for route, values in conditioned_values.items()
            }
    return summaries, route_summaries, conditioned


def _markdown(payload, summaries, routes, conditioned):
    route_filter = payload.get("route_filter", "all")
    if route_filter == "grid":
        heading = "# Pure-grid finite-source speed comparison"
        introduction = [
            "This report contains only blocks where the ordinary",
            "`lcbinint_auto` route was exactly `grid`, meaning its methods",
            "were `inverse_ray_cartesian` and/or `inverse_ray_polar` with",
            "no point-source, hexadecapole, or source-plane shortcut.",
            "Within those blocks, the explicit lcbinint grid is compared",
            "against direct VBMicrolensing finite-source integration.",
        ]
    else:
        heading = "# Near-caustic finite-source speed comparison"
        introduction = [
            "This report compares the explicit Cartesian/Polar finite-source grid",
            "against direct VBMicrolensing integration on the same reference epochs.",
            "It is not a production-dispatch comparison: the separate route tables",
            "show where `lcbinint_auto` used point/hex or source-plane shortcuts.",
        ]
    lines = [
        heading,
        "",
        *introduction,
        "",
        "## Conditions",
        "",
        f"- `q >= {payload['filters']['q_min']:g}`; `d/rho < "
        f"{payload['filters']['d_max']:g}`.",
        "- Uniform source and linear limb darkening (`c=0.5`).",
        "- Delivered targets `1e-2`, `1e-3`, and `1e-4`.",
        "- 24 consecutive epochs per block; four independently certified",
        "  reference epochs decide whether a block is usable.",
        "- Ratio is `R = t_VBM / t_lcbinint`; `R > 1` means the grid is faster.",
        "",
        "## Finite-source integrator result",
        "",
        "| profile | target | jobs | ratio points | censored | grid win rate | median R | p10 | p90 | median Nbin |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for profile in ("uniform", "linear"):
        for target in TARGETS:
            item = summaries[profile][target]
            ratio = item["ratio"]
            nbin = item["nbin"]
            if ratio.get("n", 0):
                lines.append(
                    f"| {profile} | `{target:g}` | {item['blocks']} | "
                    f"{ratio['n']} | {item['censored']} | "
                    f"{ratio['win_rate']:.1%} | {ratio['median']:.3f} | "
                    f"{ratio['p10']:.3f} | {ratio['p90']:.3f} | "
                    f"{nbin.get('median', float('nan')):.0f} |"
                )
            else:
                lines.append(
                    f"| {profile} | `{target:g}` | {item['blocks']} | "
                    f"0 | {item['censored']} | no qualified reference rows | "
                    "— | — | — | — |"
                )

    lines += [
        "",
        "## By caustic distance",
        "",
        "| profile | target | region | ratio points | win rate | median R |",
        "|---|---:|---|---:|---:|---:|",
    ]
    bands = (
        ("inner", -1.0e-12, 0.8),
        ("tangent", 0.8, 1.05),
        ("outer-near", 1.05, 2.01),
    )
    for profile in ("uniform", "linear"):
        for target in TARGETS:
            for label, low, high in bands:
                values = [
                    value
                    for row in payload["results"]
                    if row.get("status", "completed") == "completed"
                    and row["profile"] == profile
                    and abs(float(row["target"]) - target) < 1.0e-15
                    and low <= float(row["d_over_rho"]) < high
                    for value in row["ratios_vbm_over_lcbinint"]
                    if value is not None and math.isfinite(float(value))
                ]
                stats = _stats(values)
                if stats.get("n", 0):
                    lines.append(
                        f"| {profile} | `{target:g}` | {label} | "
                        f"{stats['n']} | {stats['win_rate']:.1%} | "
                        f"{stats['median']:.3f} |"
                    )

    lines += [
        "",
        "## Route-conditioned comparison",
        "",
        "The same timing ratios, restricted by the ordinary auto route recorded",
        "for each block. This is the closest available comparison to production:",
        "pure `grid` rows are finite-source grid calls, while mixed and",
        "source-plane rows are shown separately.",
        "",
        "| profile | target | route | ratio points | win rate | median R |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for profile in ("uniform", "linear"):
        for target in TARGETS:
            for route, stats in sorted(conditioned[profile][target].items()):
                if stats.get("n", 0):
                    lines.append(
                        f"| {profile} | `{target:g}` | `{route}` | "
                        f"{stats['n']} | {stats['win_rate']:.1%} | "
                        f"{stats['median']:.3f} |"
                    )
    lines += [
        "",
        "## Production route audit",
        "",
        "These counts are from the ordinary auto route on the same selected",
        "blocks. A mixed route is deliberately not reclassified as a pure grid",
        "or a pure shortcut.",
        "",
        "| profile | target | route counts |",
        "|---|---:|---|",
    ]
    for profile in ("uniform", "linear"):
        for target in TARGETS:
            counts = routes[profile][target]
            text = ", ".join(f"`{key}`: {value}"
                             for key, value in sorted(counts.items())) or "—"
            lines.append(f"| {profile} | `{target:g}` | {text} |")
    lines += [
        "",
        "The shortcut rows are not counted as grid wins. They are a separate",
        "production-routing result, because a point-source or hexadecapole",
        "formula is not the same algorithm as either finite-source integrator.",
        "",
        "Figure: `figures/near_caustic_grid_vs_vbm_ratio.png`.",
    ]
    return "\n".join(lines) + "\n"


def _figure(payload, path):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), sharey=True)
    styles = {
        "uniform": ("#2563eb", "o", "uniform"),
        "linear": ("#dc2626", "^", "linear LD"),
    }
    for axis, target in zip(axes, TARGETS):
        for profile, (colour, marker, label) in styles.items():
            points = [row for row in payload["results"]
                      if row["profile"] == profile
                      and abs(float(row["target"]) - target) < 1.0e-15]
            x, y = [], []
            for row in points:
                for value in row["ratios_vbm_over_lcbinint"]:
                    if value is not None and math.isfinite(float(value)):
                        x.append(float(row["d_over_rho"]))
                        y.append(float(value))
            axis.scatter(x, y, s=13, alpha=0.55, color=colour,
                         marker=marker, label=label, edgecolors="none")
        axis.axhline(1.0, color="#111827", linewidth=0.9, linestyle="--")
        axis.set_yscale("log")
        axis.set_ylim(1.0e-3, 1.0e2)
        axis.set_xlim(-0.08, 2.08)
        axis.set_xticks([0, 0.5, 1, 1.5, 2])
        axis.grid(True, which="major", alpha=0.25)
        axis.set_title(rf"$\epsilon={target:g}$")
        axis.set_xlabel(r"$d/\rho$")
    axes[0].set_ylabel(r"$R=t_{VBM}/t_{grid}$")
    axes[0].legend(frameon=False, loc="lower left")
    title = ("Pure-grid finite-source integrator speed"
             if payload.get("route_filter") == "grid"
             else "Near-caustic finite-source integrator speed")
    figure.suptitle(title, y=1.02)
    figure.tight_layout()
    for suffix in (".pdf", ".png"):
        figure.savefig(str(path) + suffix, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--route-filter", choices=("all", "grid"),
                        default="all")
    args = parser.parse_args()
    payload = json.loads(args.results.read_text())
    route_rows = _load_route_rows(args.routes)
    payload = _filtered_payload(payload, route_rows, args.route_filter)
    summaries, routes, conditioned = _summaries(payload, route_rows)
    args.output.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    report_name = ("REPORT_pure_grid_vs_vbm.md"
                   if args.route_filter == "grid"
                   else "REPORT_near_caustic_grid_vs_vbm.md")
    figure_name = ("pure_grid_vs_vbm_ratio"
                   if args.route_filter == "grid"
                   else "near_caustic_grid_vs_vbm_ratio")
    (args.output / report_name).write_text(
        _markdown(payload, summaries, routes, conditioned)
    )
    _figure(payload, figure_dir / figure_name)
    (args.output / "summary.json").write_text(json.dumps(
        {"route_filter": args.route_filter,
         "finite_source": summaries, "routes": routes,
         "route_conditioned": conditioned}, indent=2
    ))
    print(json.dumps({"route_filter": args.route_filter,
                      "finite_source": summaries, "routes": routes,
                      "route_conditioned": conditioned}, indent=2))


if __name__ == "__main__":
    main()
