#!/usr/bin/env python3
"""Plot the clearest linear-LD VBM/lcbinint crossover views.

The first figure is a block-median map in (d/rho, A_point), split by q.  The
second figure keeps every usable reference epoch and shows the ratio
distribution in d/rho bins.  Both use R=t_VBM/t_lcbinint, so red means that
lcbinint is faster and blue means that VBM is faster.
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
Q_GROUPS = (
    ("q_ge_1e-5", r"$q\geq10^{-5}$", lambda q: q >= 1e-5),
    ("q_lt_1e-5", r"$q<10^{-5}$", lambda q: q < 1e-5),
)
D_BINS = ((0.0, 0.25), (0.25, 0.5), (0.5, 0.8), (0.8, 1.0))
D_LABELS = ("0–0.25", "0.25–0.5", "0.5–0.8", "0.8–1.0")


def _aggregate_blocks(points: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for point in points:
        grouped[(point["source"], point["case_id"], point["target"])].append(point)
    records = []
    for group in grouped.values():
        first = group[0]
        logs = np.asarray([point["log_ratio"] for point in group], dtype=float)
        record = dict(first)
        record["log_ratio"] = float(np.median(logs))
        record["ratio"] = float(10.0 ** record["log_ratio"])
        record["n_epochs"] = len(group)
        records.append(record)
    return records


def _style(axis):
    axis.grid(True, which="both", color="#94a3b8", alpha=0.22,
              linewidth=0.55)
    axis.tick_params(labelsize=8.5)
    for spine in axis.spines.values():
        spine.set_color("#475569")
        spine.set_linewidth(0.75)


def _load_all(args) -> tuple[list[dict], list[dict]]:
    broad = _load_points(args.broad, source="broad")
    focused = _load_points(args.focused_1e_4, source="focused")
    focused.extend(_load_points(args.focused_1e_2_1e_3, source="focused"))
    return broad, focused


def _common_norm(points: list[dict], limit: float = 2.0) -> TwoSlopeNorm:
    values = np.asarray([point["log_ratio"] for point in points], dtype=float)
    observed = float(np.max(np.abs(values))) if values.size else limit
    bound = max(limit, observed)
    return TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)


def _crossover_map(points: list[dict], output: Path) -> None:
    """Map the focused finite-source branch using one point per geometry."""
    map_points = [
        point for point in points
        if 0.01 <= point["rho"] < 0.1
        and 0.0 <= point["d_over_rho"] < 1.0
    ]
    block_points = _aggregate_blocks(map_points)
    norm = _common_norm(block_points)
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(12.3, 7.3),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.12, "wspace": 0.08},
    )
    for row, (group_name, group_label, group_filter) in enumerate(Q_GROUPS):
        for col, target in enumerate(TARGETS):
            axis = axes[row, col]
            _style(axis)
            selected = [
                point for point in _take_target(block_points, target)
                if group_filter(point["q"])
            ]
            if selected:
                axis.scatter(
                    [point["d_over_rho"] for point in selected],
                    [point["apoint"] for point in selected],
                    c=[point["log_ratio"] for point in selected],
                    cmap="RdBu_r",
                    norm=norm,
                    s=26,
                    alpha=0.84,
                    linewidths=0.4,
                    edgecolors="#1f2937",
                    zorder=3,
                )
                ratios = [point["ratio"] for point in selected]
                wins = sum(ratio > 1.0 for ratio in ratios)
                median = float(np.median(ratios))
                note = f"n={len(ratios)}; win={wins / len(ratios):.0%}; " \
                    f"median R={median:.2g}"
            else:
                note = "n=0"
            axis.axvline(0.8, color="#991b1b", linestyle=(0, (3, 2)),
                         linewidth=0.9, zorder=1)
            axis.axhline(10.0, color="#b45309", linestyle=(0, (3, 2)),
                         linewidth=0.9, zorder=1)
            axis.set_xlim(-0.03, 1.03)
            axis.set_xticks([0, 0.25, 0.5, 0.8, 1.0])
            axis.set_ylim(0.9, 150.0)
            axis.set_yscale("log")
            axis.set_yticks([1, 3, 10, 30, 100])
            if row == 0:
                axis.set_title(TARGET_LABELS[target], fontsize=11, pad=7)
            if col == 0:
                axis.set_ylabel(f"$A_{{\\rm point}}$  ({group_label})",
                                fontsize=9.5)
            if row == 1:
                axis.set_xlabel(r"caustic proximity $d/\rho$", fontsize=10)
            axis.text(
                0.04,
                0.94,
                note,
                transform=axis.transAxes,
                va="top",
                fontsize=7.2,
                color="#334155",
                bbox={"facecolor": "white", "alpha": 0.78,
                      "edgecolor": "none", "pad": 1.8},
            )
    axes[0, 0].text(
        0.97,
        0.08,
        "$d/\\rho<0.8$ and $A_{\\rm point}>10$",
        transform=axes[0, 0].transAxes,
        ha="right",
        fontsize=7.3,
        color="#991b1b",
    )
    figure.subplots_adjust(left=0.10, right=0.90, bottom=0.10, top=0.91)
    cax = figure.add_axes([0.925, 0.22, 0.018, 0.56])
    colourbar = figure.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap="RdBu_r"), cax=cax)
    colourbar.set_label(r"$\log_{10}(R)$,  $R=t_{\rm VBM}/t_{\rm LCB}$",
                        fontsize=9.5, labelpad=8)
    colourbar.set_ticks([-2, -1, 0, 1, 2])
    colourbar.set_ticklabels([r"$10^{-2}$", r"$10^{-1}$", "1",
                              r"$10$", r"$10^2$"])
    figure.suptitle(
        "Where does the speed crossover occur?",
        fontsize=14,
        y=0.975,
    )
    figure.text(
        0.50,
        0.025,
        r"Focused branch: $0.01\leq\rho<0.1$, $d/\rho<1$. One point per geometry; "
        "colour is the median over usable reference epochs; "
        "red = lcbinint faster, blue = VBM faster",
        ha="center",
        fontsize=9.0,
        color="#334155",
    )
    figure.savefig(output, bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def _distribution(points: list[dict], output: Path) -> None:
    """Show all epoch-level ratios in the focused high-magnification sample."""
    selected = [
        point for point in points
        if point["apoint"] >= 10.0
        and 0.01 <= point["rho"] < 0.1
        and 0.0 <= point["d_over_rho"] < 1.0
    ]
    norm = _common_norm(selected, limit=1.5)
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(12.3, 7.0),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.14, "wspace": 0.08},
    )
    rng = np.random.default_rng(20260809)
    for row, (_, group_label, group_filter) in enumerate(Q_GROUPS):
        for col, target in enumerate(TARGETS):
            axis = axes[row, col]
            _style(axis)
            group_points = [
                point for point in _take_target(selected, target)
                if group_filter(point["q"])
            ]
            distributions = []
            for low, high in D_BINS:
                distributions.append([
                    point["ratio"] for point in group_points
                    if low <= point["d_over_rho"] < high
                ])
            positions = np.arange(1, len(D_BINS) + 1, dtype=float)
            valid = [values for values in distributions if values]
            if valid:
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
                            mpl.cm.RdBu_r(norm(math.log10(np.median(values))))
                        )
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
            axis.set_xlim(0.45, 4.55)
            axis.set_ylim(0.003, 50.0)
            axis.set_yscale("log")
            axis.set_xticks(positions)
            axis.set_xticklabels(D_LABELS, rotation=28, ha="right",
                                 fontsize=7.5)
            axis.set_yticks([1e-2, 1e-1, 1, 10])
            if row == 0:
                axis.set_title(TARGET_LABELS[target], fontsize=11, pad=7)
            if col == 0:
                axis.set_ylabel(f"$R=t_{{\\rm VBM}}/t_{{\\rm LCB}}$\n({group_label})",
                                fontsize=9.5)
            if row == 1:
                axis.set_xlabel(r"caustic proximity bin $d/\rho$", fontsize=9.5)
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
    figure.subplots_adjust(left=0.10, right=0.90, bottom=0.15, top=0.91)
    cax = figure.add_axes([0.925, 0.22, 0.018, 0.56])
    colourbar = figure.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap="RdBu_r"), cax=cax)
    colourbar.set_label(r"$\log_{10}(R)$", fontsize=9.5, labelpad=8)
    colourbar.set_ticks([-1, 0, 1])
    colourbar.set_ticklabels([r"$0.1$", "1", r"$10$"])
    figure.suptitle(
        r"Speed-ratio distributions near high-magnification caustic crossings "
        r"($A_{\rm point}\geq10$, $0.01\leq\rho<0.1$)",
        fontsize=13.5,
        y=0.975,
    )
    figure.text(
        0.50,
        0.035,
        "All usable reference epochs are shown; the dashed line is equal speed (R=1)",
        ha="center",
        fontsize=9.0,
        color="#334155",
    )
    figure.savefig(output, bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def _write_summary(focused: list[dict], output: Path) -> None:
    selected = [
        point for point in focused
        if point["apoint"] >= 10.0
        and 0.01 <= point["rho"] < 0.1
        and 0.0 <= point["d_over_rho"] < 1.0
    ]
    lines = [
        "# Crossover-figure summary",
        "",
        "The map uses one block-median point per geometry. The distribution "
        "figure uses every usable reference epoch in the focused sample.",
        "",
        "| target | q group | points | LCB win rate | median R |",
        "|---:|---|---:|---:|---:|",
    ]
    for target in TARGETS:
        for _, label, group_filter in Q_GROUPS:
            ratios = [
                point["ratio"] for point in _take_target(selected, target)
                if group_filter(point["q"])
            ]
            if ratios:
                lines.append(
                    f"| {target:.0e} | {label} | {len(ratios)} | "
                    f"{sum(value > 1 for value in ratios) / len(ratios):.1%} | "
                    f"{np.median(ratios):.3g} |"
                )
    lines.extend([
        "",
        "The proposed primary visual is the (d/rho, A_point) map.  The "
        "one-dimensional distribution is the supporting visual because it "
        "makes the R=1 crossover and the q split explicit.",
    ])
    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--broad",
        type=Path,
        default=Path("tests/diagnostics/results/recal2026/grid_vs_vbm_dark/results.json"),
    )
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
    broad, focused = _load_all(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _crossover_map(focused, args.output_dir / "vbm_crossover_map_linear_ld.pdf")
    _distribution(
        focused,
        args.output_dir / "vbm_ratio_distributions_linear_ld.pdf",
    )
    _write_summary(focused, args.output_dir / "vbm_crossover_summary.md")
    print(args.output_dir / "vbm_crossover_map_linear_ld.pdf")
    print(args.output_dir / "vbm_ratio_distributions_linear_ld.pdf")
    print(args.output_dir / "vbm_crossover_summary.md")


if __name__ == "__main__":
    main()
