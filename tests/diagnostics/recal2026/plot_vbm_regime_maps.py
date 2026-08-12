#!/usr/bin/env python3
"""Make publication-quality speed-regime maps for the direct VBM benchmark.

The plotted ratio is

    R = t_VBM / t_lcbinint.

Thus R > 1 (red) means that the grid integrator is faster, while R < 1
(blue) means that direct VBMicrolensing is faster.  Each point is one of the
measured reference epochs; no block is reduced to a single representative
timing in the scatter plots.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle


TARGETS = (1e-2, 1e-3, 1e-4)
TARGET_LABELS = {
    1e-2: r"$\epsilon=10^{-2}$",
    1e-3: r"$\epsilon=10^{-3}$",
    1e-4: r"$\epsilon=10^{-4}$",
}

DATA_FIELDS = {
    "apoint": "point_magnification",
    "rho": "rho",
    "q": "q",
    "d_over_rho": "d_over_rho",
}


def _load_points(path: Path, *, source: str) -> list[dict]:
    """Flatten one benchmark JSON file into one record per timing point."""
    payload = json.loads(path.read_text())
    points: list[dict] = []
    for row in payload["results"]:
        if row.get("profile") != "linear":
            continue
        ratios = row.get("ratios_vbm_over_lcbinint", [])
        for epoch, ratio in enumerate(ratios):
            if ratio is None or not math.isfinite(ratio) or ratio <= 0:
                continue
            point = {
                "source": source,
                "case_id": row["case_id"],
                "target": float(row["target"]),
                "apoint": max(float(row["point_magnification"]), 1.0),
                "rho": float(row["rho"]),
                "q": float(row["q"]),
                "d_over_rho": float(row["d_over_rho"]),
                "epoch": epoch,
                "ratio": float(ratio),
                "log_ratio": math.log10(float(ratio)),
            }
            points.append(point)
    return points


def _take_target(points: list[dict], target: float) -> list[dict]:
    return [point for point in points if math.isclose(point["target"], target)]


def _norm(points: list[dict], *, limit: float = 2.0) -> TwoSlopeNorm:
    """Use one symmetric scale so zero always means equal speed."""
    values = np.asarray([point["log_ratio"] for point in points], dtype=float)
    observed = float(np.nanmax(np.abs(values))) if values.size else 1.0
    return TwoSlopeNorm(vmin=-max(limit, observed), vcenter=0.0,
                        vmax=max(limit, observed))


def _scatter(
    axis,
    points: list[dict],
    x_field: str,
    y_field: str,
    norm: TwoSlopeNorm,
    *,
    size_scale: bool = False,
    size_field: str = "apoint",
):
    if not points:
        return None
    x = np.asarray([point[x_field] for point in points], dtype=float)
    y = np.asarray([point[y_field] for point in points], dtype=float)
    colour = np.asarray([point["log_ratio"] for point in points], dtype=float)
    if size_scale:
        size_coordinate = np.asarray(
            [point[size_field] for point in points], dtype=float)
        size_fraction = np.clip((np.log10(size_coordinate) - 1.0) / 1.2,
                                0.0, 1.0)
        size = 16.0 + 44.0 * size_fraction
    else:
        size = 18.0
    return axis.scatter(
        x,
        y,
        c=colour,
        cmap="RdBu_r",
        norm=norm,
        s=size,
        alpha=0.70,
        linewidths=0.28,
        edgecolors="#1f2937",
        rasterized=True,
        zorder=3,
    )


def _style_axis(axis):
    axis.grid(True, which="both", color="#94a3b8", alpha=0.22,
              linewidth=0.55)
    axis.tick_params(labelsize=8.5)
    for spine in axis.spines.values():
        spine.set_color("#475569")
        spine.set_linewidth(0.75)


def _add_high_mag_band(axis, *, xmax: float):
    axis.axvspan(10.0, xmax, color="#f59e0b", alpha=0.08, zorder=0)
    axis.axvline(10.0, color="#b45309", linestyle=(0, (3, 2)),
                 linewidth=0.9, zorder=1)


def _overview(points: list[dict], output: Path) -> None:
    """Overview projection: Apoint against the main physical parameters."""
    norm = _norm(points)
    figure, axes = plt.subplots(
        3,
        3,
        figsize=(12.4, 10.0),
        sharex="col",
        gridspec_kw={"hspace": 0.14, "wspace": 0.08},
    )
    x_max = max(point["apoint"] for point in points) * 1.08
    y_specs = [
        ("d_over_rho", r"$d/\rho$", "symlog"),
        ("rho", r"$\rho$", "log"),
        ("q", r"$q$", "log"),
    ]
    for col, target in enumerate(TARGETS):
        target_points = _take_target(points, target)
        for row, (y_field, y_label, y_scale) in enumerate(y_specs):
            axis = axes[row, col]
            _style_axis(axis)
            _add_high_mag_band(axis, xmax=x_max)
            _scatter(axis, target_points, "apoint", y_field, norm)
            axis.set_xscale("log")
            axis.set_xlim(0.92, x_max)
            axis.set_yscale(y_scale)
            if y_field == "d_over_rho":
                axis.set_ylim(-0.03, 34.0)
                axis.set_yticks([0, 0.1, 0.25, 0.5, 0.8, 1, 3, 10, 30])
            elif y_field == "rho":
                axis.set_ylim(2e-5, 1.25)
                axis.set_yticks([3e-5, 1e-4, 3e-4, 1e-3, 3e-3,
                                 1e-2, 3e-2, 1e-1, 3e-1, 1.0])
            else:
                axis.set_ylim(7e-7, 1.6)
                axis.set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2,
                                 1e-1, 1.0])
            if row == 0:
                axis.set_title(TARGET_LABELS[target], fontsize=11, pad=7)
            if col == 0:
                axis.set_ylabel(y_label, fontsize=10)
            if row == len(y_specs) - 1:
                axis.set_xlabel(r"point-source magnification $A_{\rm point}$",
                                fontsize=10)
            finite = [point for point in target_points
                      if point[y_field] >= 0]
            ratios = [point["ratio"] for point in finite]
            wins = sum(ratio > 1.0 for ratio in ratios)
            axis.text(
                0.035,
                0.95,
                f"n={len(ratios)}; LCB win={wins / len(ratios):.0%}",
                transform=axis.transAxes,
                va="top",
                fontsize=7.3,
                color="#334155",
                bbox={"facecolor": "white", "alpha": 0.72,
                      "edgecolor": "none", "pad": 1.8},
            )
    axes[0, 0].text(
        0.985,
        0.08,
        r"shaded: $A_{\rm point}\geq10$",
        transform=axes[0, 0].transAxes,
        ha="right",
        fontsize=7.3,
        color="#92400e",
    )
    figure.subplots_adjust(left=0.08, right=0.90, bottom=0.08, top=0.95)
    colourbar_axis = figure.add_axes([0.925, 0.20, 0.018, 0.58])
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap="RdBu_r")
    colourbar = figure.colorbar(mappable, cax=colourbar_axis)
    colourbar.set_label(
        r"$log_{10}(t_{\rm VBM}/t_{\rm LCB})$",
        fontsize=10,
        labelpad=8,
    )
    colourbar.set_ticks([-2, -1, 0, 1, 2])
    colourbar.set_ticklabels([r"$10^{-2}$", r"$10^{-1}$", "1",
                              r"$10$", r"$10^2$"])
    figure.text(
        0.50,
        0.985,
        "Direct VBMicrolensing versus required-Nbin lcbinint: linear LD",
        ha="center",
        va="top",
        fontsize=14,
    )
    figure.text(
        0.50,
        0.015,
        "Red (ratio > 1): lcbinint faster   |   Blue (ratio < 1): VBM faster",
        ha="center",
        fontsize=9.5,
        color="#334155",
    )
    figure.savefig(output, bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def _high_magnification(points: list[dict], output: Path) -> None:
    """Zoom into the measured finite-source/high-magnification branch."""
    selected = [point for point in points if point["apoint"] >= 10.0]
    norm = _norm(selected, limit=1.5)
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(12.4, 4.35),
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.08},
    )
    for axis, target in zip(axes, TARGETS):
        target_points = _take_target(selected, target)
        _style_axis(axis)
        axis.add_patch(Rectangle(
            (0.0, 1e-5),
            0.8,
            1.0 - 1e-5,
            facecolor="#dc2626",
            edgecolor="none",
            alpha=0.055,
            zorder=0,
        ))
        axis.axvline(0.8, color="#991b1b", linestyle=(0, (3, 2)),
                     linewidth=0.9, zorder=1)
        axis.axhline(1e-5, color="#991b1b", linestyle=(0, (3, 2)),
                     linewidth=0.9, zorder=1)
        _scatter(axis, target_points, "d_over_rho", "q", norm,
                 size_scale=True, size_field="apoint")
        axis.set_xlim(-0.03, 1.03)
        axis.set_ylim(7e-7, 1.6)
        axis.set_yscale("log")
        axis.set_xlabel(r"$d/\rho$", fontsize=10)
        axis.set_title(TARGET_LABELS[target], fontsize=11, pad=7)
        ratios = [point["ratio"] for point in target_points]
        if ratios:
            wins = sum(ratio > 1.0 for ratio in ratios)
            median = float(np.median(ratios))
            axis.text(
                0.04,
                0.95,
                f"n={len(ratios)}; win={wins / len(ratios):.0%}; "
                f"median R={median:.2g}",
                transform=axis.transAxes,
                va="top",
                fontsize=7.5,
                color="#334155",
                bbox={"facecolor": "white", "alpha": 0.78,
                      "edgecolor": "none", "pad": 1.8},
            )
    axes[0].set_ylabel(r"mass ratio $q$", fontsize=10)
    axes[0].set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2,
                        1e-1, 1.0])
    axes[0].text(
        0.04,
        0.05,
        "candidate fast region\n"
        r"$d/\rho<0.8$, $q\geq10^{-5}$",
        transform=axes[0].transAxes,
        fontsize=7.5,
        color="#991b1b",
        va="bottom",
    )
    size_handles = [
        plt.scatter([], [], s=16, color="#64748b", alpha=0.65,
                    edgecolors="#1f2937", linewidths=0.3,
                    label=r"$A_{\rm point}=10$"),
        plt.scatter([], [], s=40, color="#64748b", alpha=0.65,
                    edgecolors="#1f2937", linewidths=0.3,
                    label=r"$A_{\rm point}=30$"),
        plt.scatter([], [], s=70, color="#64748b", alpha=0.65,
                    edgecolors="#1f2937", linewidths=0.3,
                    label=r"$A_{\rm point}\gtrsim100$"),
    ]
    figure.legend(
        handles=size_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.50, -0.01),
        fontsize=8,
    )
    figure.subplots_adjust(left=0.08, right=0.90, bottom=0.19, top=0.88)
    colourbar_axis = figure.add_axes([0.925, 0.22, 0.018, 0.56])
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap="RdBu_r")
    colourbar = figure.colorbar(mappable, cax=colourbar_axis)
    colourbar.set_label(
        r"$log_{10}(t_{\rm VBM}/t_{\rm LCB})$",
        fontsize=10,
        labelpad=8,
    )
    colourbar.set_ticks([-1.5, -1, 0, 1, 1.5])
    colourbar.set_ticklabels([r"$10^{-1.5}$", r"$10^{-1}$", "1",
                              r"$10$", r"$10^{1.5}$"])
    figure.suptitle(
        r"High-magnification linear-LD branch ($A_{\rm point}\geq10$)",
        fontsize=14,
        y=0.98,
    )
    figure.text(
        0.50,
        0.06,
        r"Point area encodes $A_{\rm point}$; red means the grid integrator is faster",
        ha="center",
        fontsize=9.0,
        color="#334155",
    )
    figure.savefig(output, bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def _write_summary(points: list[dict], output: Path) -> None:
    lines = [
        "# Speed-regime map summary",
        "",
        "The plotted quantity is `R = t_VBM / t_lcbinint`; `R > 1` is a "
        "lcbinint win and `R < 1` is a VBM win. The scatter figures use "
        "every usable reference epoch from the direct timing measurements.",
        "",
        "| target | Apoint cut | points | LCB win rate | median R |",
        "|---:|---:|---:|---:|---:|",
    ]
    for target in TARGETS:
        target_points = [point for point in _take_target(points, target)
                         if point["apoint"] >= 10.0]
        ratios = [point["ratio"] for point in target_points]
        if not ratios:
            continue
        wins = sum(ratio > 1.0 for ratio in ratios)
        lines.append(
            f"| {target:.0e} | `>=10` | {len(ratios)} | "
            f"{wins / len(ratios):.1%} | {np.median(ratios):.3g} |"
        )
    lines.extend([
        "",
        "The high-magnification panel marks the candidate branch "
        "`d/rho < 0.8`, `q >= 1e-5` with guide lines. This is a visual "
        "description of the measured sample, not yet a production decision "
        "boundary; an independent holdout is required before hard-coding it.",
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
    broad = _load_points(args.broad, source="broad")
    focused = _load_points(args.focused_1e_4, source="focused")
    focused.extend(_load_points(args.focused_1e_2_1e_3, source="focused"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _overview(broad, args.output_dir / "vbm_regime_map_linear_ld.pdf")
    _high_magnification(focused, args.output_dir /
                        "vbm_regime_map_highmag_linear_ld.pdf")
    _write_summary(focused, args.output_dir / "vbm_regime_map_summary.md")
    print(args.output_dir / "vbm_regime_map_linear_ld.pdf")
    print(args.output_dir / "vbm_regime_map_highmag_linear_ld.pdf")
    print(args.output_dir / "vbm_regime_map_summary.md")


if __name__ == "__main__":
    main()
