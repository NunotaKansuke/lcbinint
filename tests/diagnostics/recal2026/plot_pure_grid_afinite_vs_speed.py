#!/usr/bin/env python3
"""Relate finite-source magnification to the VBM/grid speed ratio.

The input is the merged pure-grid benchmark.  Each point is one certified
reference epoch.  The x coordinate is A_finite, the y coordinate is
R = t_VBM / t_lcbinint, and colour encodes d/rho.  Marker shape separates a
uniform source from linear limb darkening so that colour is reserved for the
geometric variable requested in the diagnostic.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


PROFILES = ("uniform", "linear")
TARGETS = (1.0e-2, 1.0e-3, 1.0e-4)
AFINITE_BINS = (1.0, 2.0, 5.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0)
AFINITE_LABELS = (
    "1–2", "2–5", "5–10", "10–30", "30–100", "100–300",
    "300–1000", "≥1000",
)
RHO_BINS = (3.0e-5, 1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2,
            3.0e-2, 1.0e-1, 1.1)
RHO_LABELS = (
    "3e-5–1e-4", "1e-4–3e-4", "3e-4–1e-3", "1e-3–3e-3",
    "3e-3–1e-2", "1e-2–3e-2", "3e-2–1e-1", "≥1e-1",
)


def _geometry_key(row):
    return (
        int(row["case_id"]),
        float(row["s"]),
        float(row["q"]),
        float(row["rho"]),
        float(row["x"]),
        float(row["y"]),
    )


def _actual_geometry(payload):
    """Recompute lcbinint's refined caustic distance for every geometry.

    The benchmark's ``d_over_rho`` field is the requested sampling factor,
    not the achieved distance.  The latter is what the production route sees,
    and it can differ near cusps or when the normal step lands beside another
    caustic branch.
    """

    import lcbinint

    curve = lcbinint.LightCurve(
        lens="binary",
        options=lcbinint.Options(
            coordinates="vbm",
            nbin=16,
            caustic_bins=1400,
            inverse_ray_grid="cartesian",
            max_source_bins=16,
            point_source_threshold=0.0,
            hexadecapole_threshold=0.0,
            adaptive_hex_threshold=0.0,
        ),
    )
    distances = {}
    for row in payload["results"]:
        key = _geometry_key(row)
        if key in distances:
            continue
        info = curve.info(
            [float(row["x"])],
            t0=0.0,
            tE=1.0,
            u0=float(row["y"]),
            alpha=0.0,
            s=float(row["s"]),
            q=float(row["q"]),
            rho=float(row["rho"]),
        )
        distance = float(info.caustic_distances[0])
        rho = float(row["rho"])
        distances[key] = distance / rho if math.isfinite(distance) else None
    return distances


def _points(payload, distances, profile, target):
    points = []
    unresolved = 0
    for row in payload["results"]:
        if row.get("status", "completed") != "completed":
            continue
        if row["profile"] != profile:
            continue
        if abs(float(row["target"]) - target) > 1.0e-15:
            continue
        for index, (a_finite, ratio, status) in enumerate(zip(
            row.get("reference", ()),
            row.get("ratios_vbm_over_lcbinint", ()),
            row.get("ratio_status", ()),
        )):
            if status != "measured":
                unresolved += 1
                continue
            if a_finite is None or ratio is None:
                unresolved += 1
                continue
            a_finite = float(a_finite)
            ratio = float(ratio)
            if (
                not math.isfinite(a_finite)
                or not math.isfinite(ratio)
                or a_finite <= 0.0
                or ratio <= 0.0
            ):
                unresolved += 1
                continue
            points.append({
                "a_finite": a_finite,
                "ratio": ratio,
                "d_over_rho": distances.get(_geometry_key(row)),
                "rho": float(row["rho"]),
                "case_id": int(row["case_id"]),
                "reference_index": index,
            })
    return points, unresolved


def _stats(points):
    if not points:
        return {"n": 0}
    ratios = np.asarray([point["ratio"] for point in points], dtype=float)
    log_a = np.log10([point["a_finite"] for point in points])
    log_r = np.log10(ratios)
    if len(points) > 1 and np.std(log_a) > 0.0 and np.std(log_r) > 0.0:
        pearson = float(np.corrcoef(log_a, log_r)[0, 1])
    else:
        pearson = None
    order_a = np.argsort(log_a)
    order_r = np.argsort(log_r)
    rank_a = np.empty(len(points), dtype=float)
    rank_r = np.empty(len(points), dtype=float)
    rank_a[order_a] = np.arange(len(points), dtype=float)
    rank_r[order_r] = np.arange(len(points), dtype=float)
    spearman = (
        float(np.corrcoef(rank_a, rank_r)[0, 1])
        if len(points) > 1
        else None
    )
    return {
        "n": int(len(points)),
        "win_count": int(np.sum(ratios > 1.0)),
        "win_rate": float(np.mean(ratios > 1.0)),
        "median_ratio": float(np.median(ratios)),
        "p10_ratio": float(np.percentile(ratios, 10)),
        "p90_ratio": float(np.percentile(ratios, 90)),
        "median_a_finite": float(np.median([p["a_finite"] for p in points])),
        "median_d_over_rho": float(
            np.median([p["d_over_rho"] for p in points])
        ),
        "pearson_log10": pearson,
        "spearman_log10": spearman,
    }


def _binned_stats(points):
    result = []
    for low, high, label in zip(
        AFINITE_BINS[:-1], AFINITE_BINS[1:], AFINITE_LABELS
    ):
        selected = [
            point for point in points
            if low <= point["a_finite"] < high
        ]
        result.append({"label": label, "low": low, "high": high,
                       **_stats(selected)})
    return result


def _summary(payload, distances):
    summary = {}
    for profile in PROFILES:
        summary[profile] = {}
        for target in TARGETS:
            points, unresolved = _points(
                payload, distances, profile, target
            )
            all_stats = _stats(points)
            summary[profile][str(target)] = {
                "unresolved": unresolved,
                "all": all_stats,
                "bins": _binned_stats(points),
                # Retain points for plotting, but strip them before writing
                # the compact summary JSON in main().
                "points": points,
            }
    return summary


def _markdown(summary):
    lines = [
        "# A_finite versus pure-grid speed ratio",
        "",
        "This diagnostic uses the merged pure-grid benchmark. Each point is one",
        "certified reference epoch. `A_finite` is the finite-source magnification",
        "at that epoch and `R = t_VBM / t_lcbinint`; therefore `R > 1` means",
        "the LCB-in grid is faster. Colour in the first figure is the actual",
        "refined caustic distance `d/rho`; marker",
        "shape distinguishes a uniform source from linear limb darkening.",
        "",
        "## Overall relation",
        "",
        "| profile | target | n | unresolved | grid wins | win rate | median R | p10 | p90 | Spearman rho(log A, log R) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for profile in PROFILES:
        for target in TARGETS:
            item = summary[profile][str(target)]
            stats = item["all"]
            corr = stats["spearman_log10"]
            corr_text = "—" if corr is None else f"{corr:.3f}"
            lines.append(
                f"| {profile} | `{target:g}` | {stats['n']} | "
                f"{item['unresolved']} | {stats['win_count']} | "
                f"{stats['win_rate']:.1%} | {stats['median_ratio']:.3f} | "
                f"{stats['p10_ratio']:.3f} | {stats['p90_ratio']:.3f} | "
                f"{corr_text} |"
            )
    lines += [
        "",
        "## By A_finite bin",
        "",
        "| profile | target | A_finite | n | grid wins | win rate | median R | p10 | p90 | median d/rho |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for profile in PROFILES:
        for target in TARGETS:
            for item in summary[profile][str(target)]["bins"]:
                if not item["n"]:
                    continue
                lines.append(
                    f"| {profile} | `{target:g}` | {item['label']} | "
                    f"{item['n']} | {item['win_count']} | "
                    f"{item['win_rate']:.1%} | {item['median_ratio']:.3f} | "
                    f"{item['p10_ratio']:.3f} | {item['p90_ratio']:.3f} | "
                    f"{item['median_d_over_rho']:.2f} |"
                )
    lines += [
        "",
        "## High-magnification cross-check",
        "",
        "For the high-A tail, the same A_finite range is split by d/rho.",
        "This tests whether a high magnification value is sufficient by itself",
        "or whether the caustic geometry remains a necessary condition.",
        "",
        "| profile | target | A_finite | d/rho band | n | grid wins | win rate | median R |",
        "|---|---:|---|---|---:|---:|---:|---:|",
    ]
    distance_bands = (
        ("0–0.1", 0.0, 0.1),
        ("0.1–0.3", 0.1, 0.3),
        ("0.3–0.8", 0.3, 0.8),
        ("≥0.8", 0.8, float("inf")),
    )
    for profile in PROFILES:
        for target in TARGETS:
            points = summary[profile][str(target)]["points"]
            for label, low, high in distance_bands:
                selected = [
                    point for point in points
                    if point["a_finite"] >= 1000.0
                    and low <= point["d_over_rho"] < high
                ]
                stats = _stats(selected)
                if stats["n"]:
                    lines.append(
                        f"| {profile} | `{target:g}` | `≥1000` | {label} | "
                        f"{stats['n']} | {stats['win_count']} | "
                        f"{stats['win_rate']:.1%} | "
                        f"{stats['median_ratio']:.3f} |"
                    )
    lines += [
        "",
        "## By source radius",
        "",
        "The second figure uses the same axes but colours points by source",
        "radius `rho` on a logarithmic scale.",
        "",
        "| profile | target | rho | n | grid wins | win rate | median R | median A_finite |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for profile in PROFILES:
        for target in TARGETS:
            points = summary[profile][str(target)]["points"]
            for low, high, label in zip(RHO_BINS[:-1], RHO_BINS[1:], RHO_LABELS):
                selected = [
                    point for point in points
                    if low <= point["rho"] < high
                ]
                stats = _stats(selected)
                if stats["n"]:
                    lines.append(
                        f"| {profile} | `{target:g}` | {label} | "
                        f"{stats['n']} | {stats['win_count']} | "
                        f"{stats['win_rate']:.1%} | "
                        f"{stats['median_ratio']:.3f} | "
                        f"{stats['median_a_finite']:.3g} |"
                    )
    lines += [
        "",
        "Figures: `figures/Afinite_vs_speed_ratio_colored_d_over_rho.png` and",
        "`figures/Afinite_vs_speed_ratio_colored_rho.png`.",
        "",
        "Interpretation: a monotonic increase of R with A_finite would support",
        "an A_finite-only decision rule. The reported rank correlation and the",
        "d/rho colour separation test that assumption directly.",
    ]
    return "\n".join(lines) + "\n"


def _figure(summary, path, colour_key, colour_label, cmap_name, norm):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure, axes = plt.subplots(
        1, 3, figsize=(15.0, 4.9), sharey=True, constrained_layout=True
    )
    cmap = plt.get_cmap(cmap_name)
    markers = {"uniform": "o", "linear": "^"}
    labels = {"uniform": "uniform", "linear": "linear LD"}
    for axis, target in zip(axes, TARGETS):
        for profile in PROFILES:
            points = summary[profile][str(target)]["points"]
            if not points:
                continue
            axis.scatter(
                [point["a_finite"] for point in points],
                [point["ratio"] for point in points],
                c=[point[colour_key] for point in points],
                cmap=cmap,
                norm=norm,
                marker=markers[profile],
                s=21,
                alpha=0.72,
                linewidths=0.25,
                edgecolors="white",
            )
        axis.axhline(1.0, color="black", linewidth=0.9, linestyle="--")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(1.0, 3000.0)
        axis.set_ylim(1.0e-3, 1.0e2)
        axis.set_xticks([1, 10, 100, 1000])
        axis.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        axis.grid(True, which="major", alpha=0.25)
        axis.set_title(rf"$\epsilon={target:g}$")
        axis.set_xlabel(r"$A_{\rm finite}$")
    axes[0].set_ylabel("R = t_VBM / t_LCB-in")
    legend = [
        Line2D([0], [0], marker=markers[profile], color="none",
               markerfacecolor="#777777", markeredgecolor="white",
               markersize=7, label=labels[profile])
        for profile in PROFILES
    ]
    axes[0].legend(handles=legend, frameon=False, loc="lower left")
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    colorbar = figure.colorbar(scalar, ax=axes, pad=0.02, fraction=0.03)
    colorbar.set_label(colour_label)
    figure.suptitle(
        r"Finite-source magnification versus speed ratio "
        r"($R>1$: LCB-in faster)", y=1.02
    )
    for suffix in (".pdf", ".png"):
        figure.savefig(str(path) + suffix, dpi=240, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.results.read_text())
    distances = _actual_geometry(payload)
    summary = _summary(payload, distances)
    args.output.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    (args.output / "REPORT_Afinite_vs_speed_ratio.md").write_text(
        _markdown(summary)
    )
    from matplotlib.colors import LogNorm, Normalize

    _figure(
        summary,
        figure_dir / "Afinite_vs_speed_ratio_colored_d_over_rho",
        "d_over_rho",
        "actual d/rho",
        "viridis",
        Normalize(vmin=0.0, vmax=2.0),
    )
    finite_radii = [
        point["rho"]
        for profile in PROFILES
        for target in TARGETS
        for point in summary[profile][str(target)]["points"]
        if point["rho"] > 0.0 and math.isfinite(point["rho"])
    ]
    _figure(
        summary,
        figure_dir / "Afinite_vs_speed_ratio_colored_rho",
        "rho",
        r"$\rho$",
        "plasma",
        LogNorm(vmin=min(finite_radii), vmax=max(finite_radii)),
    )
    (args.output / "actual_geometry_summary.json").write_text(
        json.dumps({
            "count": len(distances),
            "min_d_over_rho": min(
                value for value in distances.values() if value is not None
            ),
            "max_d_over_rho": max(
                value for value in distances.values() if value is not None
            ),
        }, indent=2)
    )
    compact = json.loads(json.dumps(summary))
    for profile in PROFILES:
        for target in TARGETS:
            compact[profile][str(target)].pop("points", None)
    (args.output / "Afinite_vs_speed_ratio_summary.json").write_text(
        json.dumps(compact, indent=2)
    )
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
