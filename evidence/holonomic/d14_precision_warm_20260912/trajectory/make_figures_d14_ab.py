#!/usr/bin/env python3
"""Make the standard 2x3 maps for the D14-inclusive V2 A/B.

The runtime row compares the preceding full-epoch V2 run with the D14
candidate on the same trajectory snapshot.  Both sides include the D14 and
topology stage, so this figure does not mix a pure-kernel VBM timer into the
speed ratio.  The lower row keeps the established VBM-1e-6 accuracy map.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parent
CANDIDATE_RESULTS = ROOT / "results.tsv"
BASELINE_RESULTS = (ROOT.parent.parent.parent.parent /
                    "evidence/holonomic/v2_vbm_adaptive_trajectory_20260911/results.tsv")
FIGURES = ROOT / "figures_d14_ab"
FIGURES.mkdir(parents=True, exist_ok=True)

MIN_CELL_POPULATION = 8
PROFILES = ("uniform", "linear")
TARGETS = (1e-3, 1e-4)
LANES = ("cold", "warm")
SPECS = (("q", "rho", r"$q \times \rho$"),
         ("A", "rho", r"$A_{\rm fs} \times \rho$"),
         ("A", "q", r"$A_{\rm fs} \times q$"))

SPEED_BOUNDS = [0, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 65]
SPEED_COLORS = ["#2166ac", "#67a9cf", "#d1e5f0", "#f7f7f7", "#fddbc7",
                "#ef8a62", "#d73027", "#b2182b", "#7f0000", "#4d0010"]
SPEED_LABELS = ["< 0.25 ×", "0.25 ×", "0.5 ×", "1 ×", "2 ×", "4 ×",
                "8 ×", "16 ×", "32 ×", "≥ 64 ×"]
ERROR_BOUNDS = [0, 1e-8, 1e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4]
ERROR_COLORS = ["#006d2c", "#31a354", "#74c476", "#d9f0a3", "#fee08b",
                "#fdae61", "#f46d43", "#c5002f"]
ERROR_LABELS = ["≤ 10⁻⁸", "10⁻⁷", "10⁻⁶", "3 × 10⁻⁶",
                "10⁻⁵", "3 × 10⁻⁵", "10⁻⁴", "≥ 3 × 10⁻⁴"]


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def load_rows(path: Path):
    header = None
    rows = []
    with path.open() as stream:
        for line in stream:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            if header is None:
                header = fields
                continue
            if len(fields) != len(header):
                raise RuntimeError(f"{path}: row has {len(fields)} fields, expected {len(header)}")
            row = dict(zip(header, fields))
            for key, value in list(row.items()):
                try:
                    row[key] = float(value)
                except ValueError:
                    pass
            rows.append(row)
    if header is None:
        raise RuntimeError(f"empty result file: {path}")
    return header, rows


def row_key(row):
    return (int(row["case_id"]), int(row["configuration_id"]), row["profile"],
            int(row["d_bin_index"]), int(row["epoch_index"]), float(row["target"]))


def join_rows():
    _, candidate = load_rows(CANDIDATE_RESULTS)
    _, baseline = load_rows(BASELINE_RESULTS)
    if len(candidate) != 14432 or len(baseline) != 14432:
        raise RuntimeError(f"expected 14432 rows, got candidate={len(candidate)} baseline={len(baseline)}")
    baseline_by_key = {row_key(row): row for row in baseline}
    if len(baseline_by_key) != len(baseline):
        raise RuntimeError("duplicate baseline row key")
    out = []
    for row in candidate:
        old = baseline_by_key.get(row_key(row))
        if old is None:
            raise RuntimeError(f"missing baseline row: {row_key(row)}")
        for lane in LANES:
            current_time = row[f"{lane}_ms"]
            old_time = old[f"{lane}_ms"]
            row[f"{lane}_ratio"] = (old_time / current_time
                                      if finite(old_time) and finite(current_time) and current_time > 0.0
                                      else float("nan"))
            ref = row["reference"]
            value = row[f"{lane}_mu"]
            row[f"{lane}_error"] = (abs(value - ref) / abs(ref)
                                     if finite(value) and finite(ref) and ref != 0.0
                                     else float("nan"))
            row[f"{lane}_ok"] = int(row[f"{lane}_value_converged"]) == 1
        out.append(row)
    return out


def edges_for(kind):
    if kind == "q":
        return np.geomspace(1e-4, 1.0, 13), r"$q$"
    if kind == "rho":
        return np.geomspace(3e-5, 1.0, 13), r"$\rho$"
    if kind == "A":
        return np.logspace(0.0, 5.0, 13), r"$A_{\rm fs}$"
    raise ValueError(kind)


def axis_data(row, kind):
    if kind == "q":
        return row["q"]
    if kind == "rho":
        return row["rho"]
    return row["reference"]


def cell_stat(group, xkind, ykind, value_key, reducer):
    xe, _ = edges_for(xkind)
    ye, _ = edges_for(ykind)
    values = [[[] for _ in range(len(xe) - 1)] for _ in range(len(ye) - 1)]
    for row in group:
        x = axis_data(row, xkind)
        y = axis_data(row, ykind)
        value = row[value_key]
        if not (finite(x) and finite(y) and finite(value) and x > 0.0 and y > 0.0):
            continue
        ix = int(np.searchsorted(xe, x, side="right") - 1)
        iy = int(np.searchsorted(ye, y, side="right") - 1)
        if 0 <= ix < len(xe) - 1 and 0 <= iy < len(ye) - 1:
            values[iy][ix].append(float(value))
    result = np.full((len(ye) - 1, len(xe) - 1), np.nan)
    for iy, row in enumerate(values):
        for ix, cell in enumerate(row):
            if len(cell) >= MIN_CELL_POPULATION:
                result[iy][ix] = reducer(cell)
    return result


def all_counts(group, xkind, ykind, lane):
    xe, _ = edges_for(xkind)
    ye, _ = edges_for(ykind)
    total = np.zeros((len(ye) - 1, len(xe) - 1), dtype=int)
    good = np.zeros_like(total)
    for row in group:
        x = axis_data(row, xkind)
        y = axis_data(row, ykind)
        if not (finite(x) and finite(y) and x > 0.0 and y > 0.0):
            continue
        ix = int(np.searchsorted(xe, x, side="right") - 1)
        iy = int(np.searchsorted(ye, y, side="right") - 1)
        if 0 <= ix < total.shape[1] and 0 <= iy < total.shape[0]:
            total[iy][ix] += 1
            if row[f"{lane}_ok"]:
                good[iy][ix] += 1
    return total, good


def draw_axis(ax, xkind, ykind, values, title, cmap, norm, total, good):
    xe, xlabel = edges_for(xkind)
    ye, ylabel = edges_for(ykind)
    cm = cmap.copy()
    cm.set_bad("#d9d9d9")
    image = ax.pcolormesh(xe, ye, np.ma.masked_invalid(values), cmap=cm, norm=norm,
                          shading="flat", edgecolors="#9a9a9a", linewidth=0.35)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xe[0], xe[-1])
    ax.set_ylim(ye[0], ye[-1])
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=15, pad=10)
    ax.tick_params(labelsize=10)
    for iy in range(values.shape[0]):
        for ix in range(values.shape[1]):
            if total[iy, ix] >= MIN_CELL_POPULATION and good[iy, ix] < total[iy, ix]:
                x0, x1 = xe[ix], xe[ix + 1]
                y0, y1 = ye[iy], ye[iy + 1]
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                       fill=False, hatch="//", edgecolor="#555555",
                                       linewidth=0.0, alpha=0.23))
    return image


def format_target(target):
    return "10^{-3}" if abs(target - 1e-3) < 1e-12 else "10^{-4}"


def make_figure(rows, profile, target, lane):
    group = [row for row in rows if row["profile"] == profile and
             abs(row["target"] - target) < 1e-15]
    if len(group) != 3608:
        raise RuntimeError(f"unexpected condition size: {profile} {target}: {len(group)}")
    ok_count = sum(row[f"{lane}_ok"] for row in group)
    speed_group = [row for row in group
                   if finite(row[f"{lane}_ratio"]) and row[f"{lane}_ratio"] > 0.0]
    error_group = [row for row in group
                   if row[f"{lane}_ok"] and finite(row[f"{lane}_error"])]
    speed_cmap = ListedColormap(SPEED_COLORS, name="speed")
    error_cmap = ListedColormap(ERROR_COLORS, name="error")
    speed_norm = BoundaryNorm(SPEED_BOUNDS, speed_cmap.N)
    error_norm = BoundaryNorm(ERROR_BOUNDS, error_cmap.N)

    fig = plt.figure(figsize=(18, 10.5), dpi=180)
    grid = fig.add_gridspec(2, 3, left=0.055, right=0.985, top=0.84, bottom=0.20,
                            hspace=0.98, wspace=0.28)
    speed_images = []
    for j, (xkind, ykind, label) in enumerate(SPECS):
        values = cell_stat(speed_group, xkind, ykind, f"{lane}_ratio", np.median)
        total, good = all_counts(group, xkind, ykind, lane)
        speed_images.append(draw_axis(fig.add_subplot(grid[0, j]), xkind, ykind, values,
                                      label + " — runtime", speed_cmap, speed_norm,
                                      total, good))
    cax1 = fig.add_axes([0.055, 0.51, 0.93, 0.037])
    cb1 = fig.colorbar(speed_images[0], cax=cax1, orientation="horizontal",
                       boundaries=SPEED_BOUNDS,
                       ticks=[0.125, 0.375, 0.75, 1.5, 3, 6, 12, 24, 48, 64.5])
    cb1.set_ticklabels(SPEED_LABELS)
    cb1.ax.tick_params(labelsize=11, length=3)
    cb1.set_label(r"median $R=t_{\rm old\ V2}/t_{\rm D14\ V2}$  ( $R>1$: D14 V2 faster )",
                  fontsize=13, labelpad=7)

    error_images = []
    for j, (xkind, ykind, label) in enumerate(SPECS):
        values = cell_stat(error_group, xkind, ykind, f"{lane}_error",
                           lambda cell: np.percentile(cell, 95))
        total, good = all_counts(group, xkind, ykind, lane)
        error_images.append(draw_axis(fig.add_subplot(grid[1, j]), xkind, ykind, values,
                                      label + " — p95 difference", error_cmap, error_norm,
                                      total, good))
    cax2 = fig.add_axes([0.055, 0.065, 0.93, 0.037])
    cb2 = fig.colorbar(error_images[0], cax=cax2, orientation="horizontal",
                       boundaries=ERROR_BOUNDS,
                       ticks=[5e-9, 5.5e-8, 5.5e-7, 2e-6,
                              6.5e-6, 2e-5, 6.5e-5, 2e-4])
    cb2.set_ticklabels(ERROR_LABELS)
    cb2.ax.tick_params(labelsize=11, length=3)
    cb2.set_label(r"cell p95 $|A_{\rm D14\ V2}-A_{\rm VBM,1e-6}|/|A_{\rm VBM,1e-6}|$",
                  fontsize=13, labelpad=0)

    ld = "LD off (uniform)" if profile == "uniform" else "LD on (linear, c=0.5)"
    lane_label = "full-cold" if lane == "cold" else "full-warm"
    fig.suptitle(f"{ld}, D14-inclusive V2 A/B {lane_label}, target "
                 f"$\\epsilon_{{rel}}={format_target(target)}$",
                 fontsize=20, y=0.965)
    fig.text(0.5, 0.905,
             f"Both old and candidate V2 {lane_label} timings include D14/topology; "
             f"candidate value-converged {ok_count}/{len(group)} (100.0%)",
             ha="center", va="center", fontsize=11)
    fig.text(0.5, 0.135,
             "Grey cells: fewer than 8 rows. Error uses candidate value against the existing VBM RelTol=1e-6 reference; "
             "hatching marks incomplete value coverage.",
             ha="center", va="center", fontsize=10)
    stem = f"q_rho_{profile}_{'1e-3' if target == 1e-3 else '1e-4'}_d14_ab_{lane_label}"
    png = FIGURES / f"{stem}.png"
    pdf = FIGURES / f"{stem}.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    return {"profile": profile, "target": target, "lane": lane,
            "png": str(png), "pdf": str(pdf), "rows": len(group),
            "value_converged": ok_count, "speed_rows": len(speed_group),
            "error_rows": len(error_group)}


def main():
    rows = join_rows()
    figures = [make_figure(rows, profile, target, lane)
               for profile in PROFILES for target in TARGETS for lane in LANES]
    (ROOT / "figures_d14_ab.json").write_text(json.dumps(figures, indent=2) + "\n")
    print(json.dumps(figures, indent=2))


if __name__ == "__main__":
    main()
