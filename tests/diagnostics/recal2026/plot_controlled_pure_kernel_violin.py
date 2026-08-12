#!/usr/bin/env python3
"""Make a 2-by-3 binned violin view of the controlled speed benchmark.

The columns and axes match the paper-facing scatter figure: rho, d/rho, and
A_finite. The scatter points are replaced by binned, side-by-side violins,
one for each source profile. Densities are estimated in log10(R), while the
y-axis is labelled in the original speed-ratio units.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


PROFILES = (
    ("uniform", "Uniform", "#2563eb"),
    ("linear", "Linear LD", "#dc2626"),
)
TARGETS = (1.0e-3, 1.0e-4)
RHO_BINS = (
    (3.0e-5, 1.0e-3, r"$3\times10^{-5}$--$10^{-3}$"),
    (1.0e-3, 1.0e-2, r"$10^{-3}$--$10^{-2}$"),
    (1.0e-2, 1.0e-1, r"$10^{-2}$--$10^{-1}$"),
    (1.0e-1, 1.0 + 1.0e-12, r"$\geq10^{-1}$"),
)
D_BINS = (
    (0.0, 0.4, r"$0$--$0.4$"),
    (0.4, 0.8, r"$0.4$--$0.8$"),
    (0.8, 1.2, r"$0.8$--$1.2$"),
    (1.2, 1.6, r"$1.2$--$1.6$"),
    (1.6, 2.0 + 1.0e-12, r"$1.6$--$2$"),
)
A_BINS = (
    (1.0, 10.0, r"$1$--$10$"),
    (10.0, 100.0, r"$10$--$10^2$"),
    (100.0, 1000.0, r"$10^2$--$10^3$"),
    (1000.0, 1.0e5, r"$\geq10^3$"),
)
Y_LIMITS_LOG10 = (math.log10(1.0e-3), math.log10(1.0e2))
Y_TICKS_LOG10 = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0)


def _target_label(target):
    exponent = int(round(math.log10(target)))
    return rf"$\epsilon_{{\rm rel}}=10^{{{exponent}}}$"


def _ratio_values(payload, profile, target):
    values = []
    for row in payload["results"]:
        if row.get("profile") != profile:
            continue
        if abs(float(row.get("target")) - target) > 1.0e-15:
            continue
        actual_d_over_rho = row.get("actual_d_over_rho")
        point_magnification = row.get("reference")
        if actual_d_over_rho is None or point_magnification is None:
            continue
        for a_finite, ratio, status in zip(
            point_magnification,
            row.get("ratios_vbm_over_lcbinint", ()),
            row.get("ratio_status", ()),
        ):
            if status != "measured" or ratio is None:
                continue
            a_finite = float(a_finite)
            ratio = float(ratio)
            if (
                math.isfinite(a_finite)
                and a_finite > 0.0
                and math.isfinite(ratio)
                and ratio > 0.0
            ):
                values.append(
                    {
                        "rho": float(row["rho"]),
                        "d_over_rho": float(actual_d_over_rho),
                        "a_finite": a_finite,
                        "ratio": ratio,
                    }
                )
    return values


def _stats(values):
    values = np.asarray(values, dtype=float)
    if not values.size:
        return {"count": 0}
    return {
        "count": int(values.size),
        "win_rate": float(np.mean(values > 1.0)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
    }


def _select(values, column, low, high):
    return np.asarray(
        [item["ratio"] for item in values if low <= item[column] < high],
        dtype=float,
    )


def _format_log_tick(value):
    exponent = int(round(value))
    return rf"$10^{{{exponent}}}$"


def _plot_binned_violin(
    axis,
    values_by_profile,
    bins,
    column,
    log_x,
    offset_fraction,
    width_fraction,
):
    """Draw paired violins using the scale of this column's x variable."""

    for profile_index, (profile, _, colour) in enumerate(PROFILES):
        sign = -1.0 if profile_index == 0 else 1.0
        for low, high, _ in bins:
            values = _select(values_by_profile[profile], column, low, high)
            if not values.size:
                continue
            log_values = np.log10(values)
            if log_x:
                # ``violinplot(widths=...)`` expects a linear-data width even
                # when the axis is logarithmic.  Define the pair in log space
                # first, then convert its endpoints back to linear data.
                log_low = math.log10(low)
                log_high = math.log10(high)
                log_span = log_high - log_low
                log_position = (
                    0.5 * (log_low + log_high)
                    + sign * offset_fraction * log_span
                )
                half_log_width = 0.5 * width_fraction * log_span
                left = 10.0 ** (log_position - half_log_width)
                right = 10.0 ** (log_position + half_log_width)
                position = 10.0 ** log_position
                width = right - left
            else:
                center = 0.5 * (low + high)
                bin_width = high - low
                position = center + sign * offset_fraction * bin_width
                width = width_fraction * bin_width
            violin = axis.violinplot(
                log_values,
                positions=[position],
                widths=[width],
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            for body in violin["bodies"]:
                body.set_facecolor(colour)
                body.set_edgecolor(colour)
                body.set_alpha(0.58)
                body.set_linewidth(0.6)
            summary = _stats(values)
            axis.vlines(
                position,
                math.log10(summary["p10"]),
                math.log10(summary["p90"]),
                color="black",
                linewidth=0.65,
                zorder=4,
            )
            axis.vlines(
                position,
                math.log10(summary["p25"]),
                math.log10(summary["p75"]),
                color="black",
                linewidth=2.8,
                zorder=5,
            )
            axis.scatter(
                [position],
                [math.log10(summary["p50"])],
                color="white",
                edgecolor="black",
                linewidth=0.5,
                s=10,
                zorder=6,
            )
    axis.axhline(0.0, color="black", linewidth=0.7, zorder=1)
    axis.set_ylim(*Y_LIMITS_LOG10)
    axis.set_yticks(Y_TICKS_LOG10)
    axis.set_yticklabels([_format_log_tick(value) for value in Y_TICKS_LOG10])
    axis.grid(False)


def _figure(distributions, output):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    columns = (
        # The three x variables have deliberately independent geometry.
        ("rho", RHO_BINS, r"$\rho$", True, (3.0e-5, 1.0), 0.20, 0.34),
        # d/rho is linear: keep the pair separated inside each 0.4-wide bin.
        ("d_over_rho", D_BINS, r"$d/\rho$", False, (-0.03, 2.03), 0.22, 0.30),
        ("a_finite", A_BINS, r"$A_{\rm finite}$", True, (1.0, 2.0e4), 0.20, 0.34),
    )
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(12.3, 6.7),
        sharey=True,
        gridspec_kw={"wspace": 0.08, "hspace": 0.12},
    )
    for row_index, target in enumerate(TARGETS):
        for col_index, (
            column,
            bins,
            xlabel,
            log_x,
            x_limits,
            offset_fraction,
            width_fraction,
        ) in enumerate(columns):
            axis = axes[row_index, col_index]
            _plot_binned_violin(
                axis,
                {profile: distributions[(profile, target)] for profile, _, _ in PROFILES},
                bins,
                column,
                log_x,
                offset_fraction,
                width_fraction,
            )
            axis.set_xlim(*x_limits)
            if column == "rho":
                axis.set_xscale("log")
                axis.set_xticks([1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0])
                axis.set_xticklabels([r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$",
                                      r"$10^{-1}$", r"$10^{0}$"])
            elif column == "a_finite":
                axis.set_xscale("log")
                axis.set_xticks([1.0, 10.0, 100.0, 1000.0, 10000.0])
                axis.set_xticklabels([r"$10^0$", r"$10^1$", r"$10^2$",
                                      r"$10^3$", r"$10^4$"])
            else:
                axis.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
                axis.set_xticklabels(["0", "0.5", "1", "1.5", "2"])
            if row_index == len(TARGETS) - 1:
                axis.set_xlabel(xlabel)
            else:
                axis.set_xticklabels([])
            if col_index == 0:
                exponent = int(round(math.log10(target)))
                axis.set_ylabel(
                    rf"$t_{{\rm VBM}}/t_{{\rm lcbinint}}$ "
                    rf"($\epsilon_{{\rm rel}}=10^{{{exponent}}}$)"
                )
            else:
                axis.tick_params(labelleft=False)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
    legend = [
        Line2D([0], [0], color=colour, linewidth=8, alpha=0.58, label=label)
        for _, label, colour in PROFILES
    ]
    axes[0, 0].legend(
        handles=legend,
        loc="upper left",
        frameon=True,
        ncol=2,
        fontsize=14,
        handlelength=1.2,
        borderpad=0.5,
    )
    figure.subplots_adjust(left=0.10, right=0.995, bottom=0.15, top=0.98)
    figure.savefig(str(output) + ".pdf", dpi=240, bbox_inches="tight")
    figure.savefig(str(output) + ".png", dpi=240, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.results.read_text())
    distributions = {
        (profile, target): _ratio_values(payload, profile, target)
        for profile, _, _ in PROFILES
        for target in TARGETS
    }
    summary = {}
    for profile, _, _ in PROFILES:
        for target in TARGETS:
            for column, bins, _ in (
                ("rho", RHO_BINS, r"$\rho$"),
                ("d_over_rho", D_BINS, r"$d/\rho$"),
                ("a_finite", A_BINS, r"$A_{\rm finite}$"),
            ):
                summary[f"{profile}:{target:g}:{column}"] = [
                    {"label": label, **_stats(_select(distributions[(profile, target)], column, low, high))}
                    for low, high, label in bins
                ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _figure(distributions, args.output)
    Path(str(args.output) + ".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
