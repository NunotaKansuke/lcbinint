#!/usr/bin/env python3
"""Make the q-agnostic, publication-facing linear-LD crossover figures.

Only the measured domain q >= 1e-4 is retained.  The figure does not use q as
an explanatory axis: it tests the two quantities that matter visually here,
caustic proximity d/rho and point-source magnification A_point.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from plot_vbm_regime_maps import _load_points, _take_target


TARGETS = (1e-2, 1e-3, 1e-4)
TARGET_LABELS = {
    1e-2: r"$\epsilon=10^{-2}$",
    1e-3: r"$\epsilon=10^{-3}$",
    1e-4: r"$\epsilon=10^{-4}$",
}
PROX_BINS = ((0.0, 0.25), (0.25, 0.5), (0.5, 0.8), (0.8, 1.0))
PROX_LABELS = ("0–0.25", "0.25–0.5", "0.5–0.8", "0.8–1.0")
APOINT_BINS = ((1.0, 3.0), (3.0, 10.0), (10.0, 30.0),
               (30.0, 100.0), (100.0, float("inf")))
APOINT_LABELS = ("1–3", "3–10", "10–30", "30–100", "100+")


def _load_focused(args) -> list[dict]:
    points = _load_points(args.focused_1e_4, source="focused")
    points.extend(_load_points(args.focused_1e_2_1e_3, source="focused"))
    return [
        point for point in points
        if point["q"] >= 1e-4
        and 0.01 <= point["rho"] < 0.1
        and 0.0 <= point["d_over_rho"] < 1.0
    ]


def _aggregate_blocks(points: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for point in points:
        grouped[(point["case_id"], point["target"])].append(point)
    records = []
    for group in grouped.values():
        first = group[0]
        record = dict(first)
        record["log_ratio"] = float(np.median(
            [point["log_ratio"] for point in group]))
        record["ratio"] = 10.0 ** record["log_ratio"]
        records.append(record)
    return records


def _style(axis):
    axis.grid(True, which="both", color="#94a3b8", alpha=0.22,
              linewidth=0.55)
    axis.tick_params(labelsize=8.5)
    for spine in axis.spines.values():
        spine.set_color("#475569")
        spine.set_linewidth(0.75)


def _norm(points: list[dict], limit: float = 2.0) -> TwoSlopeNorm:
    values = np.asarray([point["log_ratio"] for point in points], dtype=float)
    observed = float(np.max(np.abs(values))) if values.size else limit
    bound = max(limit, observed)
    return TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)


def _save(figure, output: Path):
    figure.savefig(output, bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def _map(points: list[dict], output: Path):
    block_points = _aggregate_blocks(points)
    norm = _norm(block_points)
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(12.3, 4.55),
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.08},
    )
    for axis, target in zip(axes, TARGETS):
        _style(axis)
        selected = _take_target(block_points, target)
        axis.scatter(
            [point["d_over_rho"] for point in selected],
            [point["apoint"] for point in selected],
            c=[point["log_ratio"] for point in selected],
            cmap="RdBu_r",
            norm=norm,
            s=30,
            alpha=0.84,
            linewidths=0.4,
            edgecolors="#1f2937",
            zorder=3,
        )
        ratios = [point["ratio"] for point in selected]
        wins = sum(value > 1.0 for value in ratios)
        axis.text(
            0.04,
            0.94,
            f"n={len(ratios)}; win={wins / len(ratios):.0%}; "
            f"median R={np.median(ratios):.2g}",
            transform=axis.transAxes,
            va="top",
            fontsize=7.4,
            color="#334155",
            bbox={"facecolor": "white", "alpha": 0.78,
                  "edgecolor": "none", "pad": 1.8},
        )
        axis.axvline(0.8, color="#991b1b", linestyle=(0, (3, 2)),
                     linewidth=0.9, zorder=1)
        axis.axhline(10.0, color="#b45309", linestyle=(0, (3, 2)),
                     linewidth=0.9, zorder=1)
        axis.set_xlim(-0.03, 1.03)
        axis.set_ylim(0.9, 150.0)
        axis.set_yscale("log")
        axis.set_xticks([0, 0.25, 0.5, 0.8, 1.0])
        axis.set_yticks([1, 3, 10, 30, 100])
        axis.set_title(TARGET_LABELS[target], fontsize=11, pad=7)
        axis.set_xlabel(r"caustic proximity $d/\rho$", fontsize=10)
    axes[0].set_ylabel(r"point-source magnification $A_{\rm point}$",
                       fontsize=10)
    axes[0].text(
        0.97,
        0.08,
        "$d/\\rho<0.8$, $A_{\\rm point}>10$",
        transform=axes[0].transAxes,
        ha="right",
        fontsize=7.2,
        color="#991b1b",
    )
    figure.subplots_adjust(left=0.09, right=0.90, bottom=0.16, top=0.84)
    cax = figure.add_axes([0.925, 0.22, 0.018, 0.56])
    colourbar = figure.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap="RdBu_r"), cax=cax)
    colourbar.set_label(r"$\log_{10}(R)$,  $R=t_{\rm VBM}/t_{\rm LCB}$",
                        fontsize=9.5, labelpad=8)
    colourbar.set_ticks([-2, -1, 0, 1, 2])
    colourbar.set_ticklabels([r"$10^{-2}$", r"$10^{-1}$", "1",
                              r"$10$", r"$10^2$"])
    figure.suptitle(
        r"Crossover map: caustic proximity versus magnification "
        r"($q\geq10^{-4}$)",
        fontsize=13.5,
        y=0.97,
    )
    figure.text(
        0.50,
        0.045,
        r"Focused finite-source branch: $0.01\leq\rho<0.1$, $d/\rho<1$; "
        "one point per geometry, colour is the epoch-median speed ratio",
        ha="center",
        fontsize=8.8,
        color="#334155",
    )
    _save(figure, output)


def _box_row(axis, points, target, bins, labels, field, norm, rng, title):
    selected = _take_target(points, target)
    distributions = []
    for low, high in bins:
        distributions.append([
            point["ratio"] for point in selected
            if low <= point[field] < high
        ])
    positions = np.arange(1, len(bins) + 1, dtype=float)
    box = axis.boxplot(
        distributions,
        positions=positions,
        widths=0.52,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "#111827", "linewidth": 1.25},
        whiskerprops={"color": "#64748b", "linewidth": 0.85},
        capprops={"color": "#64748b", "linewidth": 0.85},
        boxprops={"edgecolor": "#475569", "linewidth": 0.8},
    )
    for patch, values in zip(box["boxes"], distributions):
        if values:
            patch.set_facecolor(
                mpl.cm.RdBu_r(norm(math.log10(np.median(values)))))
            patch.set_alpha(0.56)
    for position, values in zip(positions, distributions):
        if not values:
            continue
        jitter = rng.uniform(-0.14, 0.14, len(values))
        axis.scatter(
            position + jitter,
            values,
            c=[math.log10(value) for value in values],
            cmap="RdBu_r",
            norm=norm,
            s=10,
            alpha=0.42,
            linewidths=0,
            zorder=3,
            rasterized=True,
        )
    axis.axhline(1.0, color="#991b1b", linestyle=(0, (3, 2)),
                 linewidth=0.95, zorder=1)
    axis.set_xlim(0.45, len(bins) + 0.55)
    axis.set_ylim(0.003, 50.0)
    axis.set_yscale("log")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=28, ha="right", fontsize=7.5)
    axis.set_yticks([1e-2, 1e-1, 1, 10])
    counts = [len(values) for values in distributions]
    axis.text(
        0.04,
        0.94,
        "n=" + ",".join(str(count) for count in counts),
        transform=axis.transAxes,
        va="top",
        fontsize=7.1,
        color="#334155",
        bbox={"facecolor": "white", "alpha": 0.76,
              "edgecolor": "none", "pad": 1.6},
    )
    if title:
        axis.set_title(title, fontsize=10.5, pad=7)
    return distributions


def _distributions(points: list[dict], output: Path):
    norm = _norm(points, limit=1.5)
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(12.3, 7.0),
        sharex=False,
        sharey=True,
        gridspec_kw={"hspace": 0.16, "wspace": 0.08},
    )
    rng = np.random.default_rng(20260809)
    highmag = [point for point in points if point["apoint"] >= 10.0]
    near = [point for point in points if point["d_over_rho"] < 0.8]
    for col, target in enumerate(TARGETS):
        _style(axes[0, col])
        _style(axes[1, col])
        _box_row(
            axes[0, col], highmag, target, PROX_BINS, PROX_LABELS,
            "d_over_rho", norm, rng, TARGET_LABELS[target])
        _box_row(
            axes[1, col], near, target, APOINT_BINS, APOINT_LABELS,
            "apoint", norm, rng, None)
        axes[0, col].set_xlabel(r"proximity bin $d/\rho$", fontsize=9.5)
        axes[1, col].set_xlabel(r"magnification bin $A_{\rm point}$",
                                fontsize=9.5)
    axes[0, 0].set_ylabel(
        r"$R=t_{\rm VBM}/t_{\rm LCB}$\n($A_{\rm point}\geq10$)",
        fontsize=9.5)
    axes[1, 0].set_ylabel(
        r"$R=t_{\rm VBM}/t_{\rm LCB}$\n($d/\rho<0.8$)",
        fontsize=9.5)
    figure.subplots_adjust(left=0.10, right=0.90, bottom=0.15, top=0.89)
    cax = figure.add_axes([0.925, 0.22, 0.018, 0.56])
    colourbar = figure.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap="RdBu_r"), cax=cax)
    colourbar.set_label(r"$\log_{10}(R)$", fontsize=9.5, labelpad=8)
    colourbar.set_ticks([-1, 0, 1])
    colourbar.set_ticklabels([r"$0.1$", "1", r"$10$"])
    figure.suptitle(
        r"What controls the speed ratio?  Proximity versus magnification "
        r"($q\geq10^{-4}$)",
        fontsize=13.5,
        y=0.97,
    )
    figure.text(
        0.50,
        0.035,
        "All usable reference epochs are shown; dashed line is equal speed (R=1)",
        ha="center",
        fontsize=9.0,
        color="#334155",
    )
    _save(figure, output)


def _write_summary(points: list[dict], output: Path):
    highmag = [point for point in points if point["apoint"] >= 10.0]
    near = [point for point in points if point["d_over_rho"] < 0.8]
    lines = [
        "# q-free final crossover summary",
        "",
        "All records satisfy q >= 1e-4. q is a sample cut, not a plotted "
        "explanatory variable.",
        "",
        "## High-magnification sample, binned by d/rho",
        "",
        "| target | proximity bin | points | LCB win rate | median R |",
        "|---:|---:|---:|---:|---:|",
    ]
    for target in TARGETS:
        for label, (low, high) in zip(PROX_LABELS, PROX_BINS):
            ratios = [
                point["ratio"] for point in _take_target(highmag, target)
                if low <= point["d_over_rho"] < high
            ]
            if ratios:
                lines.append(
                    f"| {target:.0e} | {label} | {len(ratios)} | "
                    f"{sum(value > 1 for value in ratios) / len(ratios):.1%} | "
                    f"{np.median(ratios):.3g} |"
                )
    lines.extend([
        "",
        "## Near-caustic sample, binned by A_point",
        "",
        "| target | A_point bin | points | LCB win rate | median R |",
        "|---:|---:|---:|---:|---:|",
    ])
    for target in TARGETS:
        for label, (low, high) in zip(APOINT_LABELS, APOINT_BINS):
            ratios = [
                point["ratio"] for point in _take_target(near, target)
                if low <= point["apoint"] < high
            ]
            if ratios:
                lines.append(
                    f"| {target:.0e} | {label} | {len(ratios)} | "
                    f"{sum(value > 1 for value in ratios) / len(ratios):.1%} | "
                    f"{np.median(ratios):.3g} |"
                )
    lines.extend([
        "",
        "The map is the primary visual; the two-row distribution figure is "
        "the diagnostic for whether proximity or magnification provides the "
        "cleaner separation.",
    ])
    output.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--focused-1e-4",
        type=Path,
        default=Path(
            "tests/diagnostics/results/recal2026/grid_vs_vbm_dark/"
            "focused_rho01_01_linear_1e-4/results.json"
        ),
    )
    parser.add_argument(
        "--focused-1e-2-1e-3",
        type=Path,
        default=Path(
            "tests/diagnostics/results/recal2026/grid_vs_vbm_dark/"
            "focused_rho01_01_linear_1e-2_1e-3/results.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tests/diagnostics/results/recal2026/figures"),
    )
    args = parser.parse_args()
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })
    points = _load_focused(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _map(points, args.output_dir / "vbm_crossover_map_linear_ld_qge1e-4.pdf")
    _distributions(
        points,
        args.output_dir / "vbm_ratio_distributions_linear_ld_qge1e-4.pdf",
    )
    _write_summary(
        points,
        args.output_dir / "vbm_qge1e-4_crossover_summary.md",
    )
    print(args.output_dir / "vbm_crossover_map_linear_ld_qge1e-4.pdf")
    print(args.output_dir / "vbm_ratio_distributions_linear_ld_qge1e-4.pdf")
    print(args.output_dir / "vbm_qge1e-4_crossover_summary.md")


if __name__ == "__main__":
    main()
