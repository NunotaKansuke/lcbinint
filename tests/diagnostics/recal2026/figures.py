#!/usr/bin/env python3
"""The campaign's figures, drawn from the sweep directories rather than by hand.

Every panel is split by source profile.  Pooling uniform and limb-darkened
sources would be the wrong default here: they are different integrands, the
grids carry a different number of limb samples for each, and the one regime
where lcbinint beats a contour integrator appears only under limb darkening --
pooled, it averages away into a loss.

Two conventions are fixed for every figure that follows.

*Cost is read at delivered accuracy, not at a requested tolerance.*  For each
block and each engine the cheapest setting that actually reached the accuracy is
used.  A knob is a request; the error column is what happened, and a setting
that misses on half the geometries is not a point on anyone's front.

*The routing pipeline is drawn apart from the grids.*  ``lcbinint_auto`` chooses
its own route and will answer with a point-source or hexadecapole formula where
those are valid, which is most of its advantage at low magnification.  Drawing
it on the same line style as a forced grid would present a routing decision as a
quadrature result -- the comparison would be between a pipeline and a kernel.
It is drawn dashed, and the share of its blocks that never ran a grid at all is
annotated on the panel, so the reader can see where the speed came from.

One consequence of reading cost at delivered accuracy has to be stated rather
than quietly enjoyed: for the forced grids it selects, per block, the cheapest
of nine bin counts *after* seeing which of them met the tolerance.  No caller
has that oracle.  The grid curves are therefore a lower bound on what a fixed
choice of bins could achieve, while the routed curve is what a caller actually
gets from one decision made in advance -- which is why the routed line can sit
above the grids it dispatches to, and why the gap between them is a measure of
what the routing still leaves on the table rather than a defect in either.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .speed_analysis import ACCURACY_TARGETS, _usable, cheapest_meeting, load

PROFILES = ("uniform", "linear")

# Drawn in this order; the styling separates forced grids (solid) from the
# routing pipeline (dashed) and from the external engines (dotted).
SERIES = (
    ("lcbinint_cartesian", "lcbinint Cartesian", "#1f4e79", "-", "o"),
    ("lcbinint_polar", "lcbinint polar", "#2e8b57", "-", "s"),
    ("vbm", "VBM", "#b22222", ":", "^"),
    ("microlux", "microlux", "#8b5a00", ":", "v"),
    ("lcbinint_jax", "lcbinint JAX", "#6a3d9a", ":", "D"),
    ("lcbinint_auto", "lcbinint auto (routed)", "#444444", "--", None),
)

# Routes that answer without ever building a grid.  Their share is what makes
# the auto curve incomparable to a forced grid, so it is measured, not assumed.
CHEAP_ROUTES = frozenset(("point_source", "hexadecapole"))


def _figure(rows, columns, width=4.1, height=3.3):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        rows, columns, figsize=(width * columns, height * rows),
        squeeze=False, sharey=True)
    return figure, axes


def _binned(x, y, edges):
    """Median cost in each bin, with the inter-quartile band."""
    centres, medians, lows, highs = [], [], [], []
    x = np.asarray(x)
    y = np.asarray(y)
    for lo, hi in zip(edges[:-1], edges[1:]):
        inside = (x >= lo) & (x < hi)
        if inside.sum() < 3:
            continue
        centres.append(float(np.sqrt(lo * hi)))
        medians.append(float(np.median(y[inside])))
        lows.append(float(np.percentile(y[inside], 25)))
        highs.append(float(np.percentile(y[inside], 75)))
    return (np.array(centres), np.array(medians), np.array(lows),
            np.array(highs))


def _series_points(rows, engine, target, axis_key):
    """(x, cost) for every block where this engine reached ``target``."""
    x, cost = [], []
    for row in rows:
        if not _usable(row, target):
            continue
        value = row.get(axis_key)
        if value is None or not np.isfinite(value) or value <= 0:
            continue
        best = cheapest_meeting(row, engine, target)
        if best is None:
            continue
        x.append(float(value))
        cost.append(best[0] * 1.0e3)
    return np.array(x), np.array(cost)


def _cheap_route_share(rows, target):
    """How often ``lcbinint_auto`` answered without building a grid."""
    total = cheap = 0
    for row in rows:
        if not _usable(row, target):
            continue
        for entry in row.get("engines", []):
            if entry.get("engine") != "lcbinint_auto":
                continue
            if entry.get("knob") != target:
                continue
            methods = set(entry.get("methods") or ())
            if not methods:
                continue
            total += 1
            cheap += methods.issubset(CHEAP_ROUTES)
    return (cheap / total) if total else None


def cost_axis_figure(all_rows, axis_key, axis_label, path, *, bins=9):
    """One figure: cost against ``axis_key``, profiles down, accuracy across."""
    import matplotlib.pyplot as plt

    figure, axes = _figure(len(PROFILES), len(ACCURACY_TARGETS))
    for r, profile in enumerate(PROFILES):
        rows = [row for row in all_rows if row.get("profile") == profile]
        for c, target in enumerate(ACCURACY_TARGETS):
            axis = axes[r][c]
            values = [row[axis_key] for row in rows
                      if row.get(axis_key) and np.isfinite(row[axis_key])
                      and row[axis_key] > 0]
            if not values:
                continue
            edges = np.geomspace(min(values), max(values), bins + 1)
            for engine, label, colour, style, marker in SERIES:
                x, cost = _series_points(rows, engine, target, axis_key)
                if x.size < 4:
                    continue
                centres, medians, lows, highs = _binned(x, cost, edges)
                if not centres.size:
                    continue
                axis.plot(centres, medians, style, color=colour, marker=marker,
                          markersize=3.5, linewidth=1.4, label=label)
                if style == "-":
                    axis.fill_between(centres, lows, highs, color=colour,
                                      alpha=0.12, linewidth=0)
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.grid(True, which="major", alpha=0.25, linewidth=0.5)
            axis.grid(True, which="minor", alpha=0.10, linewidth=0.4)
            if r == 0:
                axis.set_title(f"delivered accuracy {target:g}", fontsize=10)
            if r == len(PROFILES) - 1:
                axis.set_xlabel(axis_label, fontsize=9)
            if c == 0:
                axis.set_ylabel(f"{profile}\nms per epoch", fontsize=9)
            share = _cheap_route_share(rows, target)
            if share is not None:
                axis.annotate(
                    f"auto: {share:.0%} of blocks answered\nwithout a grid",
                    xy=(0.03, 0.97), xycoords="axes fraction", fontsize=6.5,
                    va="top", color="#444444")
            axis.tick_params(labelsize=8)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="lower center", ncol=len(handles),
                      fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.02))
    figure.text(
        0.5, -0.01,
        "Solid and dotted curves take, per block, the cheapest setting that "
        "met the accuracy after the fact; they are an oracle lower bound. "
        "The dashed curve is one decision made in advance.",
        ha="center", fontsize=7, color="#444444")
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    for suffix in (".pdf", ".png"):
        figure.savefig(str(path) + suffix, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return f"{path}.pdf"


def grid_switch_figure(rule_path, path):
    """Corpus cost against the switching threshold, per profile and accuracy.

    The point of drawing it is that the optimum is flat: the corpus supports a
    decade, not two significant figures, and a curve shows that where a single
    chosen number cannot.
    """
    import matplotlib.pyplot as plt

    rule = json.loads(Path(rule_path).read_text())
    figure, axes = _figure(1, len(PROFILES), height=3.4)
    for c, profile in enumerate(PROFILES):
        axis = axes[0][c]
        drawn = False
        for index, target in enumerate(ACCURACY_TARGETS):
            cell = (rule.get(profile) or {}).get(f"{target:g}")
            if not cell or not cell.get("thresholds"):
                continue
            colour = f"C{index}"
            points = sorted(cell["thresholds"],
                            key=lambda item: item["magnification"])
            axis.plot([item["magnification"] for item in points],
                      [item["point_source"] for item in points],
                      "-o", color=colour, markersize=3.5, linewidth=1.4,
                      label=f"{target:g}")
            # The two constants the rule has to beat.  Without them the curve
            # looks like an optimisation; with them it shows what is bought.
            axis.axhline(cell["always_cartesian"], color=colour,
                         linestyle=":", linewidth=0.9, alpha=0.7)
            drawn = True
        if not drawn:
            continue
        axis.axhline(1.0, color="#000000", linewidth=0.8, alpha=0.6)
        axis.axvline(200.0, color="#b22222", linestyle="--", linewidth=1.0)
        axis.annotate("A > 200\n(dotted: always Cartesian)",
                      xy=(0.97, 0.97), xycoords="axes fraction", fontsize=6.5,
                      color="#b22222", ha="right", va="top")
        axis.set_xscale("log")
        axis.set_xlabel("point-source magnification threshold", fontsize=9)
        axis.set_title(profile, fontsize=10)
        axis.grid(True, alpha=0.25, linewidth=0.5)
        axis.tick_params(labelsize=8)
        if c == 0:
            axis.set_ylabel("corpus time / oracle", fontsize=9)
        axis.legend(fontsize=7, frameon=False, title="accuracy",
                    title_fontsize=7)
    figure.tight_layout()
    for suffix in (".pdf", ".png"):
        figure.savefig(str(path) + suffix, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return f"{path}.pdf"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", required=True,
                        help="sweep_speed output directory")
    parser.add_argument("--ext", default="",
                        help="sweep_ext output directory, if it has finished")
    parser.add_argument("--grid-switch-rule", default="")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()

    rows = load(arguments.blocks)
    if arguments.ext:
        from .ext_analysis import _key, load_ext, row_scale

        extra, _ = load_ext(arguments.ext)
        # The two runs did not share a clock: sweep_ext ran at lower
        # concurrency and its seconds are worth less than the stored ones.
        # Drawing both on one axis without correcting would show microlux and
        # the JAX backend as faster than they are by that factor, so the ext
        # timings are put on the stored run's clock using each row's own
        # control measurement.  A row with no control pairing is dropped
        # rather than drawn uncorrected.
        scales = row_scale(extra, rows)
        merged = {_key(row): row for row in rows}
        for row in extra:
            target = merged.get(_key(row))
            scale = scales.get(_key(row))
            if target is None or not scale:
                continue
            for entry in row.get("engines", []):
                seconds = entry.get("seconds_per_epoch")
                if seconds is None:
                    continue
                entry = dict(entry)
                entry["seconds_per_epoch"] = seconds * scale
                target.setdefault("engines", []).append(entry)
        rows = list(merged.values())

    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    written = [
        cost_axis_figure(rows, "magnification", "magnification $A$",
                         output / "magnification-vs-speed"),
        cost_axis_figure(rows, "rho", r"source radius $\rho$",
                         output / "rho-vs-speed"),
    ]
    if arguments.grid_switch_rule:
        written.append(grid_switch_figure(arguments.grid_switch_rule,
                                          output / "grid-switch"))
    for name in written:
        if name:
            print(f"wrote {name}")


if __name__ == "__main__":
    main()
