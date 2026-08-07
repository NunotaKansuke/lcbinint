#!/usr/bin/env python3
"""Plot absolute-branch required Nbin against Apoint bins."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from .absolute_error_law import ABSOLUTE_LEVELS


APOINT_EDGES = (1.0, 4.0, 16.0, 64.0, 256.0, 1024.0, float("inf"))
APOINT_LABELS = ("1--4", "4--16", "16--64", "64--256",
                 "256--1024", r"$\geq1024$")
COLOURS = {"cartesian": "#2563eb", "polar": "#c2410c"}


def _load(path):
    with path.open(newline="") as handle:
        records = list(csv.DictReader(handle))
    for row in records:
        row["absolute_tolerance"] = float(row["absolute_tolerance"])
        row["point_magnification"] = max(float(row["point_magnification"]), 1.0)
        row["required_resolution"] = float(row["required_resolution"])
        row["censored"] = row.get("censored", "False").lower() == "true"
    return records


def _bin_index(point):
    for index, edge in enumerate(APOINT_EDGES[1:]):
        if point < edge:
            return index
    return len(APOINT_LABELS) - 1


def _page(records, grid):
    figure, axes = plt.subplots(3, 3, figsize=(14.0, 11.0), sharey=True)
    axes = axes.ravel()
    colour = COLOURS[grid]
    for axis, atol in zip(axes, ABSOLUTE_LEVELS):
        rows = [row for row in records
                if row["dataset"] == "holdout"
                and row["grid"] == grid
                and row["absolute_tolerance"] == atol]
        grouped = [[] for _ in APOINT_LABELS]
        censored = [[] for _ in APOINT_LABELS]
        for row in rows:
            index = _bin_index(row["point_magnification"])
            grouped[index].append(row["required_resolution"])
            if row["censored"]:
                censored[index].append(row["required_resolution"])

        positions = np.arange(len(APOINT_LABELS))
        stats = []
        for values in grouped:
            values = np.asarray(values, dtype=float)
            if values.size:
                stats.append({
                    "label": "",
                    "whislo": float(np.percentile(values, 5)),
                    "q1": float(np.percentile(values, 25)),
                    "med": float(np.percentile(values, 50)),
                    "q3": float(np.percentile(values, 75)),
                    "whishi": float(np.percentile(values, 95)),
                    "fliers": [],
                })
            else:
                stats.append({
                    "label": "", "whislo": np.nan, "q1": np.nan,
                    "med": np.nan, "q3": np.nan, "whishi": np.nan,
                    "fliers": [],
                })
        axis.bxp(
            stats, positions=positions, widths=0.55, showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": colour, "alpha": 0.24,
                      "edgecolor": colour, "linewidth": 1.1},
            whiskerprops={"color": colour, "linewidth": 1.0},
            capprops={"color": colour, "linewidth": 1.0},
            medianprops={"color": "black", "linewidth": 1.6},
        )
        for position, values, censored_values in zip(
                positions, grouped, censored):
            values = np.asarray(values, dtype=float)
            if not values.size:
                continue
            axis.scatter(position, np.percentile(values, 99), color="#b42318",
                         marker="D", s=18, zorder=4)
            if censored_values:
                axis.scatter(
                    position,
                    np.percentile(censored_values, 99),
                    color="#7c3aed",
                    edgecolors="white", marker="^", s=28, zorder=5)
        axis.set_title(f"$a_{{\\rm tol}}={atol:.0e}$", fontsize=10)
        axis.set_xticks(positions)
        axis.set_xticklabels(APOINT_LABELS, rotation=35, ha="right",
                             fontsize=7.5)
        axis.set_yscale("log")
        axis.set_ylim(3.5, 500)
        axis.set_yticks([4, 8, 16, 32, 64, 128, 256, 400])
        axis.get_yaxis().set_major_formatter(plt.ScalarFormatter())
        axis.grid(True, which="both", alpha=0.2)
        axis.text(
            0.04, 0.04,
            "n=" + str(len(rows)) + "  "
            + "c=" + str(sum(len(values) for values in censored)),
            transform=axis.transAxes, fontsize=7.5,
            bbox={"facecolor": "white", "alpha": 0.75,
                  "edgecolor": "#c9d1d9"},
        )
    axes[0].set_ylabel("lower-bound required Nbin")
    axes[3].set_ylabel("lower-bound required Nbin")
    axes[6].set_ylabel("lower-bound required Nbin")
    handles = [
        plt.Line2D([], [], color="black", linewidth=1.6, label="median"),
        plt.Line2D([], [], color="#b42318", marker="D", linestyle="none",
                   markersize=4.5, label="p99 lower bound"),
        plt.Line2D([], [], color="#7c3aed", marker="^", linestyle="none",
                   markeredgecolor="white", markersize=5,
                   label="censored-row p99 lower bound"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=3,
                  frameon=False, bbox_to_anchor=(0.5, 0.015))
    figure.suptitle(
        f"Absolute branch: required Nbin versus Apoint ({grid})\n"
        "holdout; boxes are p25--p75, whiskers p5--p95",
        y=0.995, fontsize=14,
    )
    figure.tight_layout(rect=(0, 0.055, 1, 0.96))
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        default=(
            "tests/diagnostics/results/recal2026/absolute_error_law/"
            "absolute_error_records.csv"),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = _load(Path(args.records))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as pdf:
        for grid in ("cartesian", "polar"):
            figure = _page(records, grid)
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
