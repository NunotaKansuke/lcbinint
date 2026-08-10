#!/usr/bin/env python3
"""Visualise pure-kernel speed ratio against rho and actual d/rho."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np


PROFILES = ("uniform", "linear")
TARGETS = (1.0e-3, 1.0e-4)
RHO_BANDS = (
    (3.0e-5, 1.0e-3, r"$3\times10^{-5}\leq\rho<10^{-3}$"),
    (1.0e-3, 1.0e-2, r"$10^{-3}\leq\rho<10^{-2}$"),
    (1.0e-2, 1.0e-1, r"$10^{-2}\leq\rho<10^{-1}$"),
    (1.0e-1, float("inf"), r"$\rho\geq10^{-1}$"),
)
D_BANDS = (
    (0.0, 0.3, "0–0.3"),
    (0.3, 0.8, "0.3–0.8"),
    (0.8, 1.05, "0.8–1.05"),
    (1.05, 2.01, "1.05–2.01"),
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


def _actual_distances(payload):
    """Use the same refined caustic-distance calculation as the A_finite plot."""

    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from bench_grid_vs_vbm_pure_kernel import (  # noqa: PLC0415
        BUILD_DIR,
        _load_build_lcbinint,
    )

    _load_build_lcbinint(BUILD_DIR)
    import lcbinint  # noqa: PLC0415

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


def _points(payload, distances):
    points = []
    for row in payload["results"]:
        if row.get("status", "completed") != "completed":
            continue
        d_over_rho = distances.get(_geometry_key(row))
        if d_over_rho is None or not math.isfinite(float(d_over_rho)):
            continue
        for index, (ratio, status) in enumerate(zip(
            row.get("ratios_vbm_over_lcbinint", ()),
            row.get("ratio_status", ()),
        )):
            if status != "measured" or ratio is None:
                continue
            ratio = float(ratio)
            if ratio <= 0.0 or not math.isfinite(ratio):
                continue
            points.append({
                "profile": row["profile"],
                "target": float(row["target"]),
                "rho": float(row["rho"]),
                "d_over_rho": float(d_over_rho),
                "ratio": ratio,
                "case_id": int(row["case_id"]),
                "reference_index": index,
            })
    return points


def _stats(points):
    if not points:
        return {"n": 0}
    ratios = np.asarray([point["ratio"] for point in points], dtype=float)
    return {
        "n": int(len(points)),
        "grid_wins": int(np.sum(ratios > 1.0)),
        "win_rate": float(np.mean(ratios > 1.0)),
        "median_R": float(np.median(ratios)),
        "p10_R": float(np.percentile(ratios, 10)),
        "p90_R": float(np.percentile(ratios, 90)),
    }


def _select(points, profile, target, rho=None, d=None):
    selected = [
        point for point in points
        if point["profile"] == profile
        and abs(point["target"] - target) < 1.0e-15
    ]
    if rho is not None:
        low, high = rho
        selected = [point for point in selected if low <= point["rho"] < high]
    if d is not None:
        low, high = d
        selected = [
            point for point in selected
            if low <= point["d_over_rho"] < high
        ]
    return selected


def _write_report(payload, points, output):
    lines = [
        "# Pure-kernel rho and d/rho speed relation",
        "",
        "`R = t_VBM / t_LCB-in`; therefore `R > 1` means LCB-in is faster.",
        "The plotted d/rho is the refined actual caustic distance, not merely",
        "the requested sampling factor. Only external epsilon values 1e-3 and",
        "1e-4 are included.",
        "",
        "## Overall",
        "",
        "| profile | epsilon | n | LCB-in wins | win rate | median R |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for profile in PROFILES:
        for target in TARGETS:
            stats = _stats(_select(points, profile, target))
            lines.append(
                f"| {profile} | `{target:g}` | {stats['n']} | "
                f"{stats['grid_wins']} | {stats['win_rate']:.1%} | "
                f"{stats['median_R']:.3f} |"
            )
    lines += [
        "",
        "## Rho bands",
        "",
        "| profile | epsilon | rho band | n | LCB-in wins | win rate | median R |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for profile in PROFILES:
        for target in TARGETS:
            for low, high, label in RHO_BANDS:
                stats = _stats(_select(points, profile, target, rho=(low, high)))
                if stats["n"]:
                    lines.append(
                        f"| {profile} | `{target:g}` | {label} | {stats['n']} | "
                        f"{stats['grid_wins']} | {stats['win_rate']:.1%} | "
                        f"{stats['median_R']:.3f} |"
                    )
    lines += [
        "",
        "## Actual d/rho bands",
        "",
        "| profile | epsilon | actual d/rho | n | LCB-in wins | win rate | median R |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for profile in PROFILES:
        for target in TARGETS:
            for low, high, label in D_BANDS:
                stats = _stats(_select(points, profile, target, d=(low, high)))
                if stats["n"]:
                    lines.append(
                        f"| {profile} | `{target:g}` | {label} | {stats['n']} | "
                        f"{stats['grid_wins']} | {stats['win_rate']:.1%} | "
                        f"{stats['median_R']:.3f} |"
                    )
    lines += [
        "",
        "Figures:",
        "`figures/R_vs_actual_d_over_rho_by_rho.png` and",
        "`figures/R_vs_rho_colored_actual_d_over_rho.png`.",
    ]
    (output / "REPORT_rho_d_over_rho_speed.md").write_text(
        "\n".join(lines) + "\n"
    )


def _figures(points, output):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    profiles = {"uniform": ("#2563eb", "o", "uniform"),
                "linear": ("#dc2626", "^", "linear LD")}
    figure, axes = plt.subplots(
        len(TARGETS), len(RHO_BANDS), figsize=(15.0, 6.5),
        sharex="row", sharey=True, constrained_layout=True,
    )
    for row_index, target in enumerate(TARGETS):
        for col_index, (rho_low, rho_high, rho_label) in enumerate(RHO_BANDS):
            axis = axes[row_index, col_index]
            for profile, (colour, marker, label) in profiles.items():
                selected = _select(
                    points, profile, target, rho=(rho_low, rho_high)
                )
                if not selected:
                    continue
                axis.scatter(
                    [point["d_over_rho"] for point in selected],
                    [point["ratio"] for point in selected],
                    s=13, alpha=0.55, color=colour, marker=marker,
                    edgecolors="none", label=label,
                )
            axis.axhline(1.0, color="#111827", linewidth=0.8, linestyle="--")
            axis.set_yscale("log")
            axis.set_ylim(1.0e-3, 1.0e2)
            axis.set_xlim(-0.03, 2.05)
            axis.grid(True, which="major", alpha=0.25)
            if row_index == 0:
                axis.set_title(rho_label, fontsize=9)
            if col_index == 0:
                axis.set_ylabel(rf"$\epsilon={target:g}$\n$R$")
            if row_index == len(TARGETS) - 1:
                axis.set_xlabel(r"actual $d/\rho$")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
    figure.suptitle(
        r"Pure-kernel speed ratio versus actual $d/\rho$, split by $\rho$",
        y=1.02,
    )
    for suffix in (".png", ".pdf"):
        figure.savefig(
            str(output / "R_vs_actual_d_over_rho_by_rho") + suffix,
            dpi=220, bbox_inches="tight",
        )
    plt.close(figure)

    figure, axes = plt.subplots(
        1, len(TARGETS), figsize=(10.0, 4.2), sharey=True,
        constrained_layout=True,
    )
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=0.0, vmax=2.0)
    for axis, target in zip(axes, TARGETS):
        for profile, (colour, marker, label) in profiles.items():
            selected = _select(points, profile, target)
            axis.scatter(
                [point["rho"] for point in selected],
                [point["ratio"] for point in selected],
                c=[point["d_over_rho"] for point in selected],
                cmap=cmap, norm=norm, s=15, alpha=0.6, marker=marker,
                edgecolors="none", label=label,
            )
        axis.axhline(1.0, color="#111827", linewidth=0.8, linestyle="--")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_ylim(1.0e-3, 1.0e2)
        axis.grid(True, which="major", alpha=0.25)
        axis.set_title(rf"$\epsilon={target:g}$")
        axis.set_xlabel(r"$\rho$")
    axes[0].set_ylabel(r"$R=t_{VBM}/t_{LCB-in}$")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    figure.colorbar(scalar, ax=axes, label=r"actual $d/\rho$")
    figure.suptitle(r"Pure-kernel speed ratio versus $\rho$", y=1.02)
    for suffix in (".png", ".pdf"):
        figure.savefig(
            str(output / "R_vs_rho_colored_actual_d_over_rho") + suffix,
            dpi=220, bbox_inches="tight",
        )
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.results.read_text())
    distances = _actual_distances(payload)
    points = _points(payload, distances)
    args.output.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    _write_report(payload, points, args.output)
    _figures(points, figure_dir)
    summary = {
        profile: {
            str(target): _stats(_select(points, profile, target))
            for target in TARGETS
        }
        for profile in PROFILES
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.output / "actual_geometry_summary.json").write_text(json.dumps({
        "unique_geometries": len(distances),
        "min_actual_d_over_rho": min(
            value for value in distances.values() if value is not None
        ),
        "max_actual_d_over_rho": max(
            value for value in distances.values() if value is not None
        ),
    }, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
