#!/usr/bin/env python3
"""Plot speed ratio against d/rho, coloured by A_point.

The analysis cut is q >= 1e-4.  The plot is intentionally one-dimensional in
d/rho: A_point is a colour only, so a systematic colour pattern would reveal
whether magnification adds an independent trend.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from plot_vbm_regime_maps import _load_points, _take_target


TARGETS = (1e-2, 1e-3, 1e-4)
TARGET_LABELS = {
    1e-2: r"$\epsilon=10^{-2}$",
    1e-3: r"$\epsilon=10^{-3}$",
    1e-4: r"$\epsilon=10^{-4}$",
}


def _load_broad(path: Path) -> list[dict]:
    return [
        point for point in _load_points(path, source="broad")
        if point["q"] >= 1e-4
    ]


def _load_focused(path_a: Path, path_b: Path) -> list[dict]:
    points = _load_points(path_a, source="focused")
    points.extend(_load_points(path_b, source="focused"))
    return [
        point for point in points
        if point["q"] >= 1e-4
        and 0.01 <= point["rho"] < 0.1
        and 0.0 <= point["d_over_rho"] < 1.0
    ]


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def _corr(values_x, values_y) -> float:
    if len(values_x) < 3:
        return float("nan")
    x = _rank(np.asarray(values_x, dtype=float))
    y = _rank(np.asarray(values_y, dtype=float))
    return float(np.corrcoef(x, y)[0, 1])


def _style(axis):
    axis.grid(True, which="both", color="#94a3b8", alpha=0.22,
              linewidth=0.55)
    axis.tick_params(labelsize=8.5)
    for spine in axis.spines.values():
        spine.set_color("#475569")
        spine.set_linewidth(0.75)


def _bins_for(scope: str):
    if scope == "broad":
        return ((0.0, 0.25), (0.25, 0.5), (0.5, 0.8), (0.8, 1.0),
                (1.0, 3.0), (3.0, 10.0), (10.0, 31.0))
    return ((0.0, 0.25), (0.25, 0.5), (0.5, 0.8), (0.8, 1.0))


def _plot_scope(axis, points, target, scope, colour_norm):
    selected = _take_target(points, target)
    x = np.asarray([point["d_over_rho"] for point in selected], dtype=float)
    y = np.asarray([point["ratio"] for point in selected], dtype=float)
    colour = np.asarray([point["apoint"] for point in selected], dtype=float)
    axis.scatter(
        x,
        y,
        c=colour,
        cmap="viridis",
        norm=colour_norm,
        s=18,
        alpha=0.64,
        linewidths=0.22,
        edgecolors="#1f2937",
        rasterized=True,
        zorder=3,
    )
    bins = _bins_for(scope)
    centers = []
    medians = []
    lows = []
    highs = []
    for low, high in bins:
        values = y[(x >= low) & (x < high)]
        if not len(values):
            continue
        centers.append((low + high) / 2.0)
        medians.append(float(np.median(values)))
        lows.append(float(np.percentile(values, 25)))
        highs.append(float(np.percentile(values, 75)))
    if centers:
        axis.fill_between(centers, lows, highs, color="#111827",
                          alpha=0.10, zorder=1)
        axis.plot(centers, medians, color="#111827", marker="s",
                  markersize=3.3, linewidth=1.15, zorder=4)
    axis.axhline(1.0, color="#991b1b", linestyle=(0, (3, 2)),
                 linewidth=0.95, zorder=1)
    if scope == "broad":
        axis.set_xscale("symlog", linthresh=0.1, linscale=1.0)
        axis.set_xlim(-0.03, 32.0)
        axis.set_xticks([0, 0.1, 0.25, 0.5, 0.8, 1, 3, 10, 30])
    else:
        axis.set_xlim(-0.03, 1.03)
        axis.set_xticks([0, 0.25, 0.5, 0.8, 1.0])
    axis.set_ylim(0.003, 30.0)
    axis.set_yscale("log")
    axis.set_yticks([1e-2, 1e-1, 1, 10])
    log_a = np.log10(colour)
    log_r = np.log10(y)
    d_corr = _corr(x, log_r)
    a_corr = _corr(log_a, log_r)
    axis.text(
        0.04,
        0.94,
        f"n={len(y)}; Spearman(d/rho, log R)={d_corr:+.2f}\n"
        f"Spearman(log A, log R)={a_corr:+.2f}",
        transform=axis.transAxes,
        va="top",
        fontsize=7.1,
        color="#334155",
        bbox={"facecolor": "white", "alpha": 0.78,
              "edgecolor": "none", "pad": 1.7},
    )
    return {
        "n": len(y),
        "d_corr": d_corr,
        "a_corr": a_corr,
        "selected": selected,
    }


def _figure(broad, focused, output: Path):
    """Make one row: each panel is one relative target, colour is A_point."""
    del focused
    colour_values = np.asarray([point["apoint"] for point in broad],
                               dtype=float)
    colour_norm = LogNorm(vmin=1.0, vmax=max(1000.0, colour_values.max()))
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(12.3, 4.7),
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.08},
    )
    stats = {}
    for axis, target in zip(axes, TARGETS):
        _style(axis)
        stats[("broad", target)] = _plot_scope(
            axis, broad, target, "broad", colour_norm)
        axis.set_title(TARGET_LABELS[target], fontsize=11, pad=7)
        axis.set_xlabel(r"$d/\rho$", fontsize=10.5)
    axes[0].set_ylabel(r"$R=t_{\rm VBM}/t_{\rm LCB}$", fontsize=10.5)
    figure.subplots_adjust(left=0.09, right=0.90, bottom=0.17, top=0.84)
    cax = figure.add_axes([0.925, 0.21, 0.018, 0.56])
    colourbar = figure.colorbar(
        mpl.cm.ScalarMappable(norm=colour_norm, cmap="viridis"), cax=cax)
    colourbar.set_label(r"point-source magnification $A_{\rm point}$",
                        fontsize=9.5, labelpad=8)
    colourbar.set_ticks([1, 3, 10, 30, 100, 300, 1000])
    colourbar.set_ticklabels(["1", "3", "10", "30", "100", "300", "1000"])
    figure.suptitle(
        r"One-dimensional speed-ratio relation: $d/\rho$ versus "
        r"$R$ (all measured patterns, $q\geq10^{-4}$)",
        fontsize=13.5,
        y=0.97,
    )
    figure.text(
        0.50,
        0.045,
        "Colour = $A_{\\rm point}$; black curves/bands = d/rho-bin median/IQR; "
        "red dashed line = R=1",
        ha="center",
        fontsize=8.8,
        color="#334155",
    )
    figure.savefig(output, bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)
    return stats


def _write_summary(stats, output: Path):
    lines = [
        "# d/rho speed-ratio relation summary",
        "",
        "The plotted sample is linear limb darkening with q >= 1e-4. "
        "Colour encodes A_point; it is not used as an explanatory axis.",
        "",
        "| scope | target | points | Spearman(d/rho, log R) | "
        "Spearman(log A, log R) |",
        "|---|---:|---:|---:|---:|",
    ]
    for scope in ("broad",):
        for target in TARGETS:
            item = stats[(scope, target)]
            lines.append(
                f"| {scope} | {target:.0e} | {item['n']} | "
                f"{item['d_corr']:+.3f} | {item['a_corr']:+.3f} |"
            )
    lines.extend([
        "",
        "The figure is a diagnostic, not a fitted law. A stable d/rho-only "
        "rule would require a narrow colour spread at fixed d/rho and a "
        "stable binned trend across targets; the coloured scatter is shown "
        "to test that assumption directly.",
    ])
    output.write_text("\n".join(lines) + "\n")


def main():
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
    broad = _load_broad(args.broad)
    focused = _load_focused(args.focused_1e_4, args.focused_1e_2_1e_3)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "vbm_d_over_rho_speed_relation_qge1e-4.pdf"
    stats = _figure(broad, focused, output)
    _write_summary(stats,
                   args.output_dir / "vbm_d_over_rho_speed_relation_summary.md")
    print(output)
    print(args.output_dir / "vbm_d_over_rho_speed_relation_summary.md")


if __name__ == "__main__":
    main()
