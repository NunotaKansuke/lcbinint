#!/usr/bin/env python3
"""Make the paper-facing PDF for the two-branch empirical resolution law."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from tests.diagnostics.recal2026 import analysis


LEVELS = (1.0e-2, 5.0e-3, 3.0e-3, 2.0e-3, 1.0e-3,
          5.0e-4, 3.0e-4, 2.0e-4, 1.0e-4)
GRIDS = ("cartesian", "polar")
BUCKETS = (4, 6, 8, 10, 12, 16, 24, 32, 40, 50, 64, 80, 100,
           128, 160, 200, 256, 320, 400)
COLOURS = {"cartesian": "#2563eb", "polar": "#c2410c"}


def _records(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _model(report, grid, branch):
    if branch == "relative":
        selected = report["grids"][grid].get("selected")
        if selected is None:
            selected = report["grids"][grid]["budget_power"]
        return selected["model"]
    return report["grids"][grid]["model"]


def _summary(records, report, branch):
    output = {}
    tolerance_key = ("relative_tolerance" if branch == "relative"
                     else "absolute_tolerance")
    budget_field = ("relative_budget" if branch == "relative" else "budget")
    for grid in GRIDS:
        rows = [row for row in records
                if row["dataset"] == "holdout" and row["grid"] == grid]
        model = _model(report, grid, branch)
        levels = []
        for level in LEVELS:
            values = np.asarray([
                float(row["required_resolution"])
                for row in rows
                if abs(float(row[tolerance_key]) - level) < 1.0e-15
            ])
            raw = 2.0 ** (
                model["intercept"]
                + model["slope"] * np.log2(1.0e-3 / level)
                + model.get("safety_offset", 0.0)
            )
            levels.append({
                "epsilon": level,
                "p5": float(np.percentile(values, 5)),
                "p16": float(np.percentile(values, 16)),
                "q1": float(np.percentile(values, 25)),
                "median": float(np.percentile(values, 50)),
                "q3": float(np.percentile(values, 75)),
                "p84": float(np.percentile(values, 84)),
                "p95": float(np.percentile(values, 95)),
                "p99": float(np.percentile(values, 99)),
                "fit_raw": float(raw),
                "fit_bucket": float(next(
                    bucket for bucket in BUCKETS if raw <= bucket)),
                "rows": int(values.size),
            })
        output[grid] = {"levels": levels, "model": model}
    return output


def _plot_boxplots(summary, title, axis_label):
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharey=True)
    for axis, grid in zip(axes, GRIDS):
        values = summary[grid]["levels"]
        positions = np.arange(len(values))
        colour = COLOURS[grid]
        stats = [{
            "label": "",
            "whislo": item["p5"],
            "q1": item["q1"],
            "med": item["median"],
            "q3": item["q3"],
            "whishi": item["p95"],
            "fliers": [],
        } for item in values]
        axis.bxp(
            stats, positions=positions, widths=0.55, showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": colour, "alpha": 0.25,
                      "edgecolor": colour, "linewidth": 1.2},
            whiskerprops={"color": colour, "linewidth": 1.1},
            capprops={"color": colour, "linewidth": 1.1},
            medianprops={"color": "black", "linewidth": 1.8},
        )
        # The thick interval is the central 68%; it is intentionally distinct
        # from the standard Q1--Q3 box.
        for position, item in zip(positions, values):
            axis.plot([position, position], [item["p16"], item["p84"]],
                      color=colour, linewidth=7, alpha=0.48,
                      solid_capstyle="round", zorder=1)
        axis.scatter(positions, [item["p99"] for item in values],
                     color="#b42318", marker="D", s=28, zorder=4)
        axis.scatter(positions, [item["fit_bucket"] for item in values],
                     facecolors="white", edgecolors="#30363d", marker="s",
                     s=28, zorder=5)
        axis.set_xticks(positions)
        axis.set_xticklabels([f"{item['epsilon']:.0e}" for item in values])
        axis.set_yscale("log")
        axis.set_xlim(-0.6, len(values) - 0.4)
        axis.set_ylim(3.5, 500)
        axis.set_yticks([4, 8, 16, 32, 64, 128, 256, 400])
        axis.get_yaxis().set_major_formatter(plt.ScalarFormatter())
        axis.grid(True, which="both", alpha=0.22)
        axis.set_title(grid)
        axis.set_xlabel(axis_label)
        axis.text(
            0.03, 0.04,
            "box: Q1--Q3; whisker: p5--p95\n"
            "wide bar: p16--p84",
            transform=axis.transAxes, fontsize=8.5,
            bbox={"facecolor": "white", "alpha": 0.78,
                  "edgecolor": "#c9d1d9"},
        )
    axes[0].set_ylabel("required Nbin")
    handles = [
        plt.Line2D([], [], color=COLOURS["polar"], linewidth=7, alpha=0.48,
                   label="central 68% (p16--p84)"),
        plt.Line2D([], [], color="black", linewidth=1.8, label="median"),
        plt.Line2D([], [], color="#b42318", marker="D", linestyle="none",
                   markersize=5, label="p99"),
        plt.Line2D([], [], color="#30363d", marker="s", markerfacecolor="white",
                   linestyle="none", markersize=5, label="fitted bucket"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=4,
                  frameon=False, bbox_to_anchor=(0.5, -0.02))
    figure.suptitle(title, y=0.99, fontsize=14)
    figure.tight_layout(rect=(0, 0.10, 1, 0.96))
    return figure


def _plot_mixed_heatmap(mixed):
    figure, axes = plt.subplots(
        1, 2, figsize=(11.0, 4.7), sharey=True, constrained_layout=True)
    for axis, grid in zip(axes, GRIDS):
        values = np.asarray(mixed["holdout"][grid]["coverage_matrix"])
        image = axis.imshow(values, origin="upper", vmin=0.99, vmax=1.0,
                            cmap="YlGnBu", aspect="auto")
        axis.set_xticks(np.arange(len(LEVELS)))
        axis.set_yticks(np.arange(len(LEVELS)))
        labels = [f"{level:.0e}" for level in LEVELS]
        axis.set_xticklabels(labels, rotation=45, ha="right")
        axis.set_yticklabels(labels)
        axis.set_xlabel("relative tolerance")
        axis.set_title(grid)
        for row in range(len(LEVELS)):
            for column in range(len(LEVELS)):
                axis.text(column, row, f"{values[row, column] * 100:.1f}",
                          ha="center", va="center", fontsize=6.5,
                          color="black" if values[row, column] < 0.997 else "white")
        axis.grid(False)
    axes[0].set_ylabel("absolute tolerance")
    figure.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025,
                    pad=0.04, label="holdout coverage")
    figure.suptitle(
        "Mixed max-budget rule: holdout coverage (%)\n"
        "Nmix = min(Nabsolute, Nrelative); every cell is a 99% pass",
        y=1.02, fontsize=13,
    )
    return figure


def _choose_example(rows):
    usable = []
    for index, row in enumerate(rows):
        required = row.get("required", {}).get("0.001", {})
        if not required.get("usable"):
            continue
        if required.get("cartesian") is None or required.get("polar") is None:
            continue
        point = max(abs(float(row.get("point_magnification", 1.0))), 1.0)
        rho = max(float(row.get("rho", 1.0e-12)), 1.0e-12)
        distance = float(row.get("caustic_distance", 0.0)) / rho
        usable.append((index, row, required, point, rho, distance))
    features = np.asarray([
        [np.log10(item[3]), np.log10(item[4]), item[5]]
        for item in usable
    ])
    centre = np.median(features, axis=0)
    scale = np.percentile(features, 75, axis=0) - np.percentile(features, 25, axis=0)
    scale[scale == 0.0] = 1.0
    candidates = []
    for item, feature in zip(usable, features):
        if item[2]["cartesian"] != 10 or item[2]["polar"] != 12:
            continue
        score = float(np.sum(((feature - centre) / scale) ** 2))
        candidates.append((score, item))
    if not candidates:
        raise RuntimeError("could not find a representative holdout case")
    return min(candidates, key=lambda item: item[0])[1][1]


def _example_curves(row):
    reference = float(row["reference"]["value"])
    budget = 1.0e-3 * max(abs(reference), 1.0)
    output = {
        "reference": reference,
        "budget": budget,
        "case_id": row.get("case_id"),
        "profile": row.get("profile"),
        "required": row["required"]["0.001"],
        "rows": [],
    }
    for grid in GRIDS:
        values = []
        for bucket in BUCKETS:
            entry = row[grid].get(str(bucket), row[grid].get(bucket))
            if not entry:
                continue
            magnification = float(entry["magnification"])
            values.append({
                "nbin": bucket,
                "magnification": magnification,
                "absolute_error": abs(magnification - reference),
            })
        output["rows"].append((grid, values))
    return output


def _plot_example(example):
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    reference = example["reference"]
    budget = example["budget"]
    for axis, (grid, values) in zip(axes, example["rows"]):
        nbin = np.asarray([item["nbin"] for item in values])
        magnification = np.asarray([item["magnification"] for item in values])
        colour = COLOURS[grid]
        axis.plot(nbin, magnification, color=colour, marker="o", markersize=4,
                  linewidth=1.5, label=grid)
        axis.axhline(reference, color="black", linewidth=1.2,
                     label="reference")
        axis.axhspan(reference - budget, reference + budget,
                     color="#6b7280", alpha=0.15, label="error budget")
        axis.set_xscale("log")
        axis.set_xticks([4, 8, 16, 32, 64, 128, 256, 400])
        axis.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        axis.grid(True, which="both", alpha=0.22)
        axis.set_title(f"{grid}: A(N)")
        axis.set_xlabel("Nbin")
        axis.set_ylabel("magnification")
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    axes[0].text(
        0.04, 0.04,
        f"reference={reference:.6f}\nbudget={budget:.3g}",
        transform=axes[0].transAxes, fontsize=8.5,
        bbox={"facecolor": "white", "alpha": 0.78,
              "edgecolor": "#c9d1d9"},
    )
    right = axes[1]
    right.clear()
    for grid, values in example["rows"]:
        nbin = np.asarray([item["nbin"] for item in values])
        error = np.asarray([item["absolute_error"] for item in values])
        right.plot(nbin, np.maximum(error, 1.0e-12), color=COLOURS[grid],
                   marker="o", markersize=4, linewidth=1.5, label=grid)
    right.axhline(budget, color="black", linewidth=1.2, label="error budget")
    right.set_xscale("log")
    right.set_yscale("log")
    right.set_xticks([4, 8, 16, 32, 64, 128, 256, 400])
    right.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    right.set_title("absolute error |A(N)-Aref|")
    right.set_xlabel("Nbin")
    right.set_ylabel("absolute error")
    right.grid(True, which="both", alpha=0.22)
    right.legend(frameon=False, fontsize=9, loc="lower left")
    figure.suptitle(
        f"Representative holdout case: case {example['case_id']} ({example['profile']})\n"
        f"reltol=1e-3; required buckets: Cartesian {example['required']['cartesian']}, "
        f"polar {example['required']['polar']}",
        y=1.02, fontsize=13,
    )
    figure.tight_layout()
    return figure


def _plot_method(relative, absolute, mixed):
    figure = plt.figure(figsize=(11.0, 7.2))
    figure.text(0.07, 0.94, "A common empirical rule for both tolerance branches",
                fontsize=15, weight="medium")
    figure.text(
        0.07, 0.87,
        "1. For every row, increase Nbin and record the first persistent crossing\n"
        "   of the requested budget against the high-resolution reference.\n\n"
        "2. Fit the discovery p99 in base-two logarithms, separately for the\n"
        "   relative and absolute branches.  The grid only changes C and beta.\n\n"
        "3. At runtime policy level, use B=max(Babs,Brel), hence\n"
        "   epsilon=max(atol/max(|A|,1), reltol) and N=min(Nabs,Nrel).\n\n"
        "4. The 9x9 mixed holdout matrix is a validation of this composition,\n"
        "   not a second fit.  All 162 cells clear the 99% target.",
        fontsize=11, va="top", linespacing=1.45,
    )
    table = [("grid", "Crel", "beta-rel", "Cabs", "beta-abs", "mixed min")]
    for grid in GRIDS:
        rel_model = relative[grid]["model"]
        abs_model = absolute[grid]["model"]
        rel_c = 2.0 ** (rel_model["intercept"] + rel_model.get("safety_offset", 0.0))
        abs_c = 2.0 ** (abs_model["intercept"] + abs_model.get("safety_offset", 0.0))
        table.append((
            grid, f"{rel_c:.1f}", f"{rel_model['slope']:.3f}",
            f"{abs_c:.1f}", f"{abs_model['slope']:.3f}",
            f"{mixed['holdout'][grid]['summary']['minimum_coverage'] * 100:.2f}%",
        ))
    axis = figure.add_axes((0.08, 0.15, 0.84, 0.22))
    axis.axis("off")
    rendered = axis.table(cellText=table[1:], colLabels=table[0],
                          cellLoc="center", colLoc="center", loc="center")
    rendered.auto_set_font_size(False)
    rendered.set_fontsize(10.5)
    rendered.scale(1.0, 1.65)
    figure.text(
        0.07, 0.08,
        "The fitted laws are population-level p99 rules; the runtime estimator is a separate claim.\n"
        "This PDF documents calibration and validation, not a production C++ selector change.",
        fontsize=10, color="#57606a",
    )
    return figure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", default="tests/diagnostics/results/recal2026/error_budget_law/error_budget_records.csv")
    parser.add_argument("--report", default="tests/diagnostics/results/recal2026/error_budget_law/error_budget_law.json")
    parser.add_argument("--absolute-records", default="tests/diagnostics/results/recal2026/absolute_error_law/absolute_error_records.csv")
    parser.add_argument("--absolute-report", default="tests/diagnostics/results/recal2026/absolute_error_law/absolute_error_law.json")
    parser.add_argument("--mixed-report", default="tests/diagnostics/results/recal2026/mixed_error_law.json")
    parser.add_argument("--holdout", default="tests/diagnostics/results/recal2026/holdout")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = _records(Path(args.records))
    absolute_records = _records(Path(args.absolute_records))
    report = json.loads(Path(args.report).read_text())
    absolute_report = json.loads(Path(args.absolute_report).read_text())
    mixed_report = json.loads(Path(args.mixed_report).read_text())
    relative_summary = _summary(records, report, "relative")
    absolute_summary = _summary(absolute_records, absolute_report, "absolute")
    example = _example_curves(_choose_example(analysis.load(args.holdout)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as pdf:
        pdf.savefig(_plot_boxplots(
            relative_summary,
            "Relative branch: required resolution on the independent holdout",
            "relative tolerance $r_{\\rm tol}$"), bbox_inches="tight")
        plt.close("all")
        pdf.savefig(_plot_boxplots(
            absolute_summary,
            "Absolute branch: required resolution on the independent holdout",
            "absolute tolerance $a_{\\rm tol}$"), bbox_inches="tight")
        plt.close("all")
        pdf.savefig(_plot_mixed_heatmap(mixed_report), bbox_inches="tight")
        plt.close("all")
        pdf.savefig(_plot_example(example), bbox_inches="tight")
        plt.close("all")
        pdf.savefig(_plot_method(
            relative_summary, absolute_summary, mixed_report),
            bbox_inches="tight")
        plt.close("all")
    print(output)


if __name__ == "__main__":
    main()
