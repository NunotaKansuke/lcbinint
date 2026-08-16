#!/usr/bin/env python3
"""Make paper-style C0 warm-up light-curve comparison figures.

The plotted API comparison keeps only the finite-source cases with visible
structure.  VBM is shown as a line, the warm-up lcbinint result as large
scatter, and each row includes a relative-error panel plus caustic-focused
insets.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_SCRIPT = Path(__file__).with_name("benchmark_synthetic_warmup.py")
OUTPUT_DIR = (
    ROOT
    / "tests"
    / "diagnostics"
    / "results"
    / "recal2026"
    / "synthetic_lightcurve_benchmark_narrow_windows_20260816"
)

SELECTED_CASES = (
    "resonant_high_mag",
    "resonant_large_source",
    "close_binary",
    "high_q",
    "wide_planet",
    "close_secondary_caustics",
)
CASE_TITLES = {
    "resonant_high_mag": "resonant high magnification",
    "resonant_large_source": "resonant large source",
    "close_binary": "close binary",
    "high_q": "high mass ratio",
    "wide_planet": "wide planetary caustic",
    "close_secondary_caustics": "close-binary secondary caustics",
}
GEOMETRY_EXTRA_TIME_PADDING = {
    "high_q": 1.5,
}
def load_benchmark_module():
    spec = importlib.util.spec_from_file_location("synthetic_benchmark", BENCHMARK_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {BENCHMARK_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_lcbinint(benchmark):
    return benchmark.load_lcbinint()


def selected_cases(benchmark):
    cases = {case.name: case for case in benchmark.BINARY_CASES}
    return [cases[name] for name in SELECTED_CASES]


def vbm_parameters(benchmark, params):
    return benchmark.vbm_parameters("binary", params)


def evaluate(benchmark, lcbinint, case, n_times=240):
    import VBMicrolensing

    case_n_times = int(case.n_times or n_times)
    times = np.linspace(case.t_min, case.t_max, case_n_times)
    params = dict(case.params)
    curve = benchmark.make_curve(lcbinint, "binary", ld=False)
    report = curve.warmup(times, params, grid_timing_repeats=1)
    lc_values = np.asarray(curve(times, params), dtype=float)

    vbm = VBMicrolensing.VBMicrolensing()
    vbm.Tol = benchmark.TOLERANCE
    vbm.RelTol = benchmark.TOLERANCE
    vbm.a1 = 0.0
    vbm.a2 = 0.0
    result = vbm.BinaryLightCurve(
        vbm_parameters(benchmark, params), times.tolist()
    )
    vbm_values = np.asarray(result[0], dtype=float)

    caustics = curve.caustics(s=params["s"], q=params["q"])
    # Geometry gets its own dense trajectory grid.  In particular, the
    # high-q inset extends beyond the light-curve interval; these points are
    # used only for the trajectory drawing, never for magnification values.
    geometry_padding = GEOMETRY_EXTRA_TIME_PADDING.get(case.name, 0.0)
    geometry_times = np.linspace(
        case.t_min - geometry_padding,
        case.t_max + geometry_padding,
        max(1200, case_n_times * 5),
    )
    trajectory = curve.source_trajectory(geometry_times, params)
    source_x = np.asarray(trajectory.x, dtype=float)
    source_y = np.asarray(trajectory.y, dtype=float)
    # Use the standard dimensionless event time in every panel.  The
    # magnifications and errors above are still evaluated on physical t.
    display_times = (times - float(params["t0"])) / float(params["tE"])
    time_label = r"$(t-t_0)/t_E$"
    relative_error = np.abs(lc_values - vbm_values) / np.maximum(
        np.abs(vbm_values), 1.0e-12
    )
    return {
        "case": case,
        "times": display_times,
        "time_label": time_label,
        "params": params,
        "lc": lc_values,
        "vbm": vbm_values,
        "relative_error": relative_error,
        "report": report,
        "caustics": [
            (
                np.asarray(xs, dtype=float),
                np.asarray(ys, dtype=float),
            )
            for xs, ys in zip(caustics.x, caustics.y)
        ],
        "source_x": source_x,
        "source_y": source_y,
    }


def geometry_data(record, *, focus="nearest"):
    source = np.column_stack((record["source_x"], record["source_y"]))
    branches = []
    for xs, ys in record["caustics"]:
        finite = np.isfinite(xs) & np.isfinite(ys)
        branches.append(np.column_stack((xs[finite], ys[finite])))
    if focus == "upper_secondary" and len(branches) >= 3:
        # In the close topology the upper off-axis branch is the first
        # secondary caustic met by the selected event trajectory.
        branch_index = max(
            range(len(branches)), key=lambda index: float(np.mean(branches[index][:, 1]))
        )
    else:
        distances = []
        for branch in branches:
            delta = source[:, None, :] - branch[None, :, :]
            distances.append(float(np.min(np.sum(delta * delta, axis=2))))
        branch_index = int(np.argmin(distances))
    branch = branches[branch_index]
    delta = source[:, None, :] - branch[None, :, :]
    source_distances = np.min(np.sum(delta * delta, axis=2), axis=1)
    closest_index = int(np.argmin(source_distances))
    return source, branches, branch_index, branch, closest_index


def style_geometry_axis(inset, title, *, square):
    inset.set_facecolor("white")
    if square:
        inset.set_box_aspect(1.0)
        inset.set_aspect("equal", adjustable="box")
    else:
        inset.set_aspect("equal", adjustable="box")
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_color("#525a64")
        spine.set_linewidth(0.7)
    if title:
        inset.set_title(title, fontsize=6.7, pad=1.5)


def source_linewidth(record):
    """Encode the source radius in the trajectory stroke width."""
    rho = float(record["params"]["rho"])
    return 0.75 + 80.0 * rho


def square_limits(inset, points, padding=0.10):
    x_values = np.concatenate([values[:, 0] for values in points])
    y_values = np.concatenate([values[:, 1] for values in points])
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[finite]
    y_values = y_values[finite]
    span = max(float(np.ptp(x_values)), float(np.ptp(y_values)), 1.0e-6)
    span *= 1.0 + 2.0 * padding
    x_center = 0.5 * (float(np.min(x_values)) + float(np.max(x_values)))
    y_center = 0.5 * (float(np.min(y_values)) + float(np.max(y_values)))
    inset.set_xlim(x_center - 0.5 * span, x_center + 0.5 * span)
    inset.set_ylim(y_center - 0.5 * span, y_center + 0.5 * span)


def caustic_zoom_limits(inset, branch, closest_point, scale=1.3, padding=0.08):
    points = np.vstack((branch, np.asarray(closest_point, dtype=float)))
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    span = max(float(x_max - x_min), float(y_max - y_min), 1.0e-6)
    span *= scale
    span *= 1.0 + 2.0 * padding
    x_center = 0.5 * (float(x_min) + float(x_max))
    y_center = 0.5 * (float(y_min) + float(y_max))
    inset.set_xlim(
        x_center - 0.5 * span,
        x_center + 0.5 * span,
    )
    inset.set_ylim(
        y_center - 0.5 * span,
        y_center + 0.5 * span,
    )


def local_geometry_inset(
    ax,
    record,
    *,
    loc="upper right",
    width="62%",
    height="62%",
    bbox_to_anchor=(0.0, -0.05, 1.25, 1.00),
    title="caustic + full source path",
    focus="nearest",
    zoom_scale=1.3,
):
    inset = inset_axes(
        ax,
        width=width,
        height=height,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        bbox_transform=ax.transAxes,
        borderpad=0.0,
    )
    inset.set_zorder(10)
    source, _, _, branch, closest_index = geometry_data(
        record, focus=focus
    )
    # Use the outermost available trajectory endpoints; the axes remain
    # caustic-focused, so only the visible portion is clipped by the inset.
    path_source = source
    branch_span = max(float(np.ptp(branch[:, 0])), float(np.ptp(branch[:, 1])))
    branch_linewidth = 2.2 if branch_span < 0.08 else 1.25
    inset.plot(
        path_source[:, 0],
        path_source[:, 1],
        color="#0072B2",
        lw=source_linewidth(record),
        alpha=0.95,
        zorder=2,
    )
    inset.plot(
        branch[:, 0],
        branch[:, 1],
        color="#30343b",
        lw=branch_linewidth,
        alpha=0.95,
        zorder=4,
    )
    caustic_zoom_limits(inset, branch, source[closest_index], scale=zoom_scale)
    style_geometry_axis(inset, "", square=True)
    return inset


def full_geometry_inset(ax, record):
    inset = inset_axes(
        ax,
        width="62%",
        height="62%",
        loc="upper right",
        bbox_to_anchor=(0.0, -0.05, 1.25, 1.00),
        bbox_transform=ax.transAxes,
        borderpad=0.0,
    )
    inset.set_zorder(10)
    source, branches, _, _, _ = geometry_data(
        record, focus="upper_secondary"
    )
    for branch in branches:
        inset.plot(
            branch[:, 0],
            branch[:, 1],
            color="#30343b",
            lw=1.1,
            alpha=0.95,
            zorder=4,
        )
    inset.plot(
        source[:, 0],
        source[:, 1],
        color="#0072B2",
        lw=source_linewidth(record),
        alpha=0.9,
        zorder=2,
    )
    # The source array is the full trajectory, but the viewing window is set
    # by the caustic topology so the three small branches remain legible.
    square_limits(inset, branches, padding=0.12)
    style_geometry_axis(inset, "", square=True)
    return inset


def geometry_insets(ax, record):
    if record["case"].name == "close_secondary_caustics":
        # Keep the close-binary event consistent with the other rows while
        # showing all three caustics in its one geometry inset.
        full_geometry_inset(ax, record)
        return
    if record["case"].name == "wide_planet":
        # The expanded wide-event light curve has structure at both ends of
        # the panel; place the inset center at t ~= -1 in the light-curve
        # axes so it does not sit on top of the right-hand planetary peak.
        return local_geometry_inset(
            ax,
            record,
            loc="upper center",
            bbox_to_anchor=(-0.125, -0.05, 1.25, 1.00),
        )
    if record["case"].name == "close_binary":
        return local_geometry_inset(
            ax,
            record,
            width="64%",
            height="64%",
            title="central caustic + full source path",
        )
    return local_geometry_inset(ax, record)


def add_case_row(axes, record, show_x_label=True, panel_number=None):
    ax_curve, ax_error = axes
    case = record["case"]
    params = record["params"]
    times = record["times"]
    time_label = record["time_label"]
    lc_values = record["lc"]
    vbm_values = record["vbm"]
    errors = record["relative_error"]

    ax_curve.plot(
        times,
        vbm_values,
        color="#D55E00",
        lw=1.35,
        label="VBMicrolensing",
        zorder=2,
    )
    ax_curve.scatter(
        times,
        lc_values,
        color="#0072B2",
        s=16,
        alpha=0.82,
        linewidths=0.2,
        edgecolors="white",
        label=r"$\mathtt{lcbinint}$ · with warm-up",
        zorder=3,
    )
    ax_curve.set_title(
        f"$s={params['s']:g}$, $q={params['q']:g}$, "
        f"$\\rho={params['rho']:g}$",
        fontsize=10.2,
        loc="left",
        pad=4,
    )
    ax_curve.set_ylabel("magnification")
    ax_curve.grid(True, color="#d9dee5", linewidth=0.55, alpha=0.7)
    ax_curve.set_axisbelow(True)
    if panel_number is not None:
        ax_curve.text(
            0.02,
            0.96,
            str(panel_number),
            transform=ax_curve.transAxes,
            ha="left",
            va="top",
            fontsize=10.5,
            fontweight="bold",
            color="#111827",
            zorder=12,
        )
    geometry_insets(ax_curve, record)

    ax_error.scatter(
        times,
        errors,
        color="#0072B2",
        s=10,
        alpha=0.82,
        linewidths=0.15,
        edgecolors="white",
    )
    ax_error.axhline(
        1.0e-3,
        color="#4b5563",
        lw=0.9,
        ls=(0, (3, 2)),
    )
    ax_error.set_yscale("log")
    positive = errors[np.isfinite(errors) & (errors > 0.0)]
    lower = max(float(np.min(positive)) * 0.45, 1.0e-7)
    upper = max(float(np.max(positive)) * 2.2, 2.0e-3)
    ax_error.set_ylim(lower, upper)
    ax_error.set_ylabel("relative error")
    ax_error.grid(True, which="both", color="#d9dee5", linewidth=0.55, alpha=0.7)
    ax_error.set_axisbelow(True)
    ax_error.text(
        0.98,
        0.08,
        f"max = {np.max(errors):.2e}",
        transform=ax_error.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.2,
        color="#0072B2",
    )
    if show_x_label:
        ax_curve.set_xlabel(time_label)
        ax_error.set_xlabel(time_label, labelpad=1)
        ax_error.tick_params(axis="x", pad=1)


def make_grid_figure(records, output_path):
    """Render six cases as equal-sized panels in a two-column grid."""
    fig = plt.figure(figsize=(12.0, 9.6))
    fig.patch.set_facecolor("white")
    outer = fig.add_gridspec(
        3,
        2,
        height_ratios=(1.0, 1.0, 1.0),
        hspace=0.28,
        wspace=0.14,
    )
    axes = []

    def add_cell(slot, record):
        inner = outer[slot].subgridspec(
            2,
            1,
            height_ratios=(2.35, 0.72),
            hspace=0.06,
        )
        ax_curve = fig.add_subplot(inner[0])
        ax_error = fig.add_subplot(inner[1], sharex=ax_curve)
        add_case_row(
            (ax_curve, ax_error),
            record,
            show_x_label=True,
            panel_number=slot[0] * 2 + slot[1] + 1,
        )
        ax_curve.set_xlabel("")
        ax_curve.tick_params(axis="x", labelbottom=False)
        ax_error.set_xlabel(record["time_label"], labelpad=1)
        ax_error.tick_params(axis="x", pad=1)
        axes.append((ax_curve, ax_error))

    for index, record in enumerate(records):
        add_cell((index // 2, index % 2), record)

    axes[0][0].legend(
        loc="upper left",
        bbox_to_anchor=(0.015, 0.90),
        frameon=False,
        fontsize=8.5,
        handlelength=2.0,
        labelspacing=0.3,
        borderaxespad=0.45,
    )
    fig.subplots_adjust(top=0.985, bottom=0.04, left=0.06, right=0.985)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_speed_figure(output_path):
    with (OUTPUT_DIR / "benchmark.json").open(encoding="utf-8") as handle:
        payload = json.load(handle)
    selected = set(SELECTED_CASES)
    records = {
        (row["case"], row["profile"]): row
        for row in payload["records"]
        if row["lens"] == "binary" and row["case"] in selected
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), sharey=True)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(top=0.90, bottom=0.20, left=0.085, right=0.985, wspace=0.16)
    colors = {"no_warmup": "#7b8794", "warmup": "#0072B2", "vbm": "#D55E00"}
    names = list(SELECTED_CASES)
    positions = np.arange(len(names), dtype=float)
    width = 0.24
    for axis, profile, title in zip(
        axes,
        ("C0_uniform", "C1_linear_ld"),
        ("uniform source", "linear limb-darkened source"),
    ):
        for offset, key, label in (
            (-width, "no_warmup", r"$\mathtt{lcbinint}$ · without warm-up"),
            (0.0, "warmup", r"$\mathtt{lcbinint}$ · with warm-up"),
            (width, "vbm", "VBMicrolensing"),
        ):
            values = [
                records[(name, profile)][key]["steady_median_ms_per_epoch"]
                for name in names
            ]
            axis.bar(
                positions + offset,
                values,
                width=width * 0.92,
                color=colors[key],
                label=label,
                alpha=0.92,
                edgecolor="white",
                linewidth=0.35,
            )
        axis.set_yscale("log")
        axis.set_xticks(positions)
        axis.set_xticklabels([str(index) for index in range(1, len(names) + 1)], fontsize=14)
        axis.tick_params(axis="y", which="major", labelsize=14, width=0.9, length=5)
        axis.tick_params(axis="x", which="major", labelsize=14, width=0.9, length=5)
        axis.set_title(title, fontsize=16, pad=10)
        axis.grid(True, which="both", axis="y", color="#d9dee5", linewidth=0.55, alpha=0.75)
        axis.set_axisbelow(True)
        axis.set_xlabel("")
        for spine in axis.spines.values():
            spine.set_color("#4b5563")
    axes[0].set_ylabel("milliseconds per epoch", fontsize=16, labelpad=12)
    axes[0].legend(
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        ncol=1,
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#4b5563",
        fontsize=12,
        handlelength=2.0,
        labelspacing=0.45,
        columnspacing=1.0,
        borderpad=0.55,
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    benchmark = load_benchmark_module()
    lcbinint = load_lcbinint(benchmark)
    records = []
    for case in selected_cases(benchmark):
        print(f"evaluating {case.name}", flush=True)
        records.append(evaluate(benchmark, lcbinint, case))
    grid_output = OUTPUT_DIR / "paper_binary_c0_warmup_grid.png"
    make_grid_figure(records, grid_output)
    print(grid_output)
    speed_output = OUTPUT_DIR / "paper_binary_speed_selected.png"
    make_speed_figure(speed_output)
    print(speed_output)


if __name__ == "__main__":
    main()
