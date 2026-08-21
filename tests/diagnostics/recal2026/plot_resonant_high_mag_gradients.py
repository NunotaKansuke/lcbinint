#!/usr/bin/env python3
"""Render a paper-style light-curve/Jacobian figure for one binary event.

The figure deliberately compares only the compiled JAX warm-up and microLUX
lanes.  The upper row shows the magnification and the remaining rows show the
physical-parameter Jacobian curves stored by the final benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS = (
    ROOT
    / "tests"
    / "diagnostics"
    / "results"
    / "recal2026"
    / "synthetic_lightcurve_jax_microlux_value_and_grad_final_20260820"
    / "results.json"
)

CASE_NAME = "resonant_high_mag"
PROFILE_ORDER = ("C1_linear_ld",)
PROFILE_TITLES = {
    "C0_uniform": r"Uniform source ($c=0$)",
    "C1_linear_ld": r"Linear limb darkening ($c=0.5$)",
}
PARAMETER_LABELS = {
    "s": r"$\partial A/\partial s$",
    "q": r"$\partial A/\partial q$",
    "rho": r"$\partial A/\partial\rho$",
    "u0": r"$\partial A/\partial u_0$",
    "alpha": r"$\partial A/\partial\alpha$",
    "t0": r"$\partial A/\partial t_0$",
    "tE": r"$\partial A/\partial t_E$",
}
# Match the physical-parameter order used in the paper-style reference panel.
# The benchmark has seven parameters; a repeated ``rho`` in a verbal list is
# therefore represented only once.
PLOT_PARAMETER_ORDER = ("t0", "tE", "u0", "rho", "q", "s", "alpha")

JAX_COLOR = "#0072B2"
MICROLUX_COLOR = "#D55E00"
ZERO_COLOR = "#7A8088"
GRID_COLOR = "#D9DEE5"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS,
        help="final benchmark results.json",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="output path without .pdf/.png suffix",
    )
    return parser.parse_args()


def load_records(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = {
        record["profile"]: record
        for record in payload["records"]
        if record["case"] == CASE_NAME
    }
    missing = [profile for profile in PROFILE_ORDER if profile not in records]
    if missing:
        raise ValueError(f"missing profiles for {CASE_NAME}: {missing}")
    return payload, records


def lane_arrays(record, lane_name):
    lane = record[lane_name]
    times = np.asarray(record["times"], dtype=float)
    parameters = record["parameters"]
    tau = (times - float(parameters["t0"])) / float(parameters["tE"])
    values = np.asarray(lane["value"]["values"], dtype=float)
    gradients = np.asarray(lane["gradient"]["values"], dtype=float)
    if values.shape != tau.shape:
        raise ValueError(f"unexpected value shape for {lane_name}: {values.shape}")
    if gradients.shape != (tau.size, 7):
        raise ValueError(
            f"unexpected gradient shape for {lane_name}: {gradients.shape}"
        )
    if not np.isfinite(values).all() or not np.isfinite(gradients).all():
        raise ValueError(f"non-finite curve in {record['profile']} / {lane_name}")
    return tau, values, gradients


def ratio(record, metric):
    jax_lane = record["jax_warmup"][metric]["steady_seconds"]
    micro_lane = record["microlux"][metric]["steady_seconds"]
    return float(micro_lane) / float(jax_lane)


def style_axis(axis, *, show_x=False):
    axis.grid(True, which="major", color=GRID_COLOR, linewidth=0.5, alpha=0.55)
    axis.set_axisbelow(True)
    axis.tick_params(axis="both", which="major", labelsize=8.0, length=3.2)
    for spine in axis.spines.values():
        spine.set_color("#343A40")
        spine.set_linewidth(0.7)
    if not show_x:
        axis.tick_params(axis="x", labelbottom=False)


def plot_lane(axis, tau, values, *, label=None, color=JAX_COLOR, linestyle="-"):
    axis.plot(
        tau,
        values,
        color=color,
        linewidth=1.45,
        linestyle=linestyle,
        label=label,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=3,
    )


def make_figure(payload, records, output_prefix: Path):
    available_parameter_keys = tuple(payload["gradient_parameter_keys"])
    parameter_keys = PLOT_PARAMETER_ORDER
    missing_parameters = [key for key in parameter_keys if key not in available_parameter_keys]
    if missing_parameters:
        raise ValueError(
            f"missing requested physical parameters: {missing_parameters}; "
            f"available={available_parameter_keys}"
        )
    gradient_indices = tuple(available_parameter_keys.index(key) for key in parameter_keys)
    missing_labels = [key for key in parameter_keys if key not in PARAMETER_LABELS]
    if missing_labels:
        raise ValueError(f"missing labels for parameters: {missing_labels}")

    profile = PROFILE_ORDER[0]
    record = records[profile]
    jax_tau, jax_values, jax_gradients = lane_arrays(record, "jax_warmup")
    micro_tau, micro_values, micro_gradients = lane_arrays(record, "microlux")
    if not np.array_equal(jax_tau, micro_tau):
        raise ValueError("JAX and microLUX use different time grids")
    tau = jax_tau

    all_values = (jax_values, micro_values)
    value_min = min(float(np.min(values)) for values in all_values)
    value_max = max(float(np.max(values)) for values in all_values)
    value_pad = max((value_max - value_min) * 0.045, 1.0e-6)
    value_limits = (value_min - value_pad, value_max + value_pad)
    gradient_limits = []
    gradient_scales = []
    for gradient_index in gradient_indices:
        peak = max(
            float(np.max(np.abs(jax_gradients[:, gradient_index]))),
            float(np.max(np.abs(micro_gradients[:, gradient_index]))),
        )
        pad = max(peak * 0.08, 1.0e-12)
        gradient_limits.append((-peak - pad, peak + pad))
        if peak >= 1.0e3:
            exponent = 3
        elif peak >= 1.0e2:
            exponent = 2
        else:
            exponent = 0
        gradient_scales.append(10.0**exponent)

    figure = plt.figure(figsize=(8.6, 7.9), facecolor="white")
    grid = figure.add_gridspec(
        8,
        1,
        height_ratios=(2.45, 0.78, 0.78, 0.78, 0.78, 0.78, 0.78, 0.78),
        hspace=0.055,
    )
    axes = [figure.add_subplot(grid[row, 0]) for row in range(8)]
    top = axes[0]
    style_axis(top)
    top.set_xlim(float(tau[0]), float(tau[-1]))
    top.set_ylim(*value_limits)
    plot_lane(
        top,
        tau,
        jax_values,
        label=r"$\mathtt{lcbinint}$",
        color=JAX_COLOR,
    )
    plot_lane(
        top,
        tau,
        micro_values,
        label=r"$\mathtt{microlux}$",
        color=MICROLUX_COLOR,
        linestyle=(0, (4.0, 2.0)),
    )
    top.set_ylabel("Magnification", fontsize=9.2)
    top.legend(
        loc="upper left",
        bbox_to_anchor=(0.012, 0.995),
        frameon=False,
        fontsize=10.5,
        handlelength=3.1,
        handletextpad=0.55,
        borderaxespad=0.0,
    )

    for index, (parameter_key, gradient_index) in enumerate(
        zip(parameter_keys, gradient_indices, strict=True), start=1
    ):
        axis = axes[index]
        style_axis(axis, show_x=index == len(axes) - 1)
        scale = gradient_scales[index - 1]
        exponent = int(np.log10(scale))
        axis.set_xlim(float(tau[0]), float(tau[-1]))
        axis.set_ylim(
            gradient_limits[index - 1][0] / scale,
            gradient_limits[index - 1][1] / scale,
        )
        axis.axhline(0.0, color=ZERO_COLOR, linewidth=0.55, alpha=0.75, zorder=1)
        plot_lane(
            axis,
            tau,
            jax_gradients[:, gradient_index] / scale,
            color=JAX_COLOR,
        )
        plot_lane(
            axis,
            tau,
            micro_gradients[:, gradient_index] / scale,
            color=MICROLUX_COLOR,
            linestyle=(0, (4.0, 2.0)),
        )
        axis.set_ylabel(PARAMETER_LABELS[parameter_key], fontsize=8.6, labelpad=4.0)
        if exponent:
            axis.text(
                0.985,
                0.86,
                rf"$\times 10^{{{exponent}}}$",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=7.4,
                color="#4B535B",
        )
        if index == len(axes) - 1:
            axis.set_xlabel(r"$(t-t_0)/t_E$", fontsize=9.0, labelpad=3.5)

    figure.subplots_adjust(left=0.105, right=0.99, top=0.99, bottom=0.065)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    figure.savefig(
        output_prefix.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(figure)


def main():
    args = parse_args()
    results_path = args.results.resolve()
    output_prefix = args.output_prefix
    if output_prefix is None:
        output_prefix = results_path.parent / "resonant_high_mag_gradient_panels"
    payload, records = load_records(results_path)
    make_figure(payload, records, output_prefix.resolve())
    print(output_prefix.with_suffix(".pdf"))
    print(output_prefix.with_suffix(".png"))


if __name__ == "__main__":
    main()
