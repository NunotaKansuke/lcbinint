#!/usr/bin/env python3
"""Make a paper-style schematic of Cartesian run-based region recovery.

The lattice and the mapped-inside mask are deliberately schematic: this figure
is meant to explain the row-wise recovery logic, rather than to show one
particular lens mapping.  The two panels use exactly the same mapped-inside
region: the seed row is isolated on the left, while the right panel follows
the search three rows upward and downward and shows the one-cell-padded
8-neighbour candidate bands.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle


N_COLS = 19
N_ROWS = 16
SEED_ROW = 7
SEED_COL = 4

COLOURS = {
    "ink": "#20262e",
    "grid": "#b9c4ce",
    "inside": "#dbeaf3",
    "recovered": "#6fa8c4",
    "recovered_edge": "#2f789d",
    "run": "#f28e2b",
    "run_edge": "#c66b12",
    "seed": "#2563eb",
    "propagation": "#7b4aa8",
    "white": "#ffffff",
}

CANDIDATE_LINESTYLE = (0, (2.4, 1.4))
PROPAGATION_ARROW_LENGTH = 1.35


def build_inside_mask() -> np.ndarray:
    """Return a rounded, connected image-region mask on the lattice.

    The two narrow lobes leave a seven-cell gap on ``SEED_ROW``.  A one-row
    bridge makes the lobes one connected component, followed by a tapered cap
    above it.  The lower side retains separated runs, so the right panel can
    show both a reached run and the gap-repair candidate below the bridge.
    """

    x, y = np.meshgrid(np.arange(N_COLS), np.arange(N_ROWS))
    left_lobe = ((x - 4.0) / 1.8) ** 2 + ((y - 6.5) / 4.5) ** 2 <= 1.0
    right_lobe = ((x - 14.0) / 1.8) ** 2 + ((y - 6.5) / 4.5) ** 2 <= 1.0
    upper_bridge = (
        ((y == 10) & (x >= 6) & (x <= 12))
        | ((y == 11) & (x >= 5) & (x <= 13))
        | ((y == 12) & (x >= 4) & (x <= 14))
        | ((y == 13) & (x >= 5) & (x <= 13))
    )
    # Add two stepped inner shoulders so that the gap closes progressively
    # toward the bridge rather than ending in a rectangular notch.
    inner_shoulders = (
        ((y == 8) & ((x == 6) | (x == 12)))
        | ((y == 9) & ((x == 6) | (x == 7) | (x == 11) | (x == 12)))
    )
    mask = left_lobe | right_lobe | upper_bridge | inner_shoulders

    # Keep the construction honest: the seed must lie in the mask and the
    # selected row must visibly contain two separated runs.
    runs = find_row_runs(mask, SEED_ROW)
    if not mask[SEED_ROW, SEED_COL] or len(runs) != 2:
        raise RuntimeError("The schematic mask no longer has the intended geometry")
    return mask


def find_row_runs(mask: np.ndarray, row: int) -> list[tuple[int, int]]:
    """Return inclusive contiguous true intervals in one lattice row."""

    runs: list[tuple[int, int]] = []
    start: int | None = None
    for col, occupied in enumerate(mask[row]):
        if occupied and start is None:
            start = col
        elif not occupied and start is not None:
            runs.append((start, col - 1))
            start = None
    if start is not None:
        runs.append((start, mask.shape[1] - 1))
    return runs


def padded_candidate_band(
    run: tuple[int, int],
    n_cols: int,
) -> tuple[int, int]:
    """Return the one-cell-padded band searched in a neighbouring row."""

    return max(0, run[0] - 1), min(n_cols - 1, run[1] + 1)


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def draw_lattice(
    axis: mpl.axes.Axes,
    mask: np.ndarray,
    *,
    inside_colour: str,
    right_padding: float = 0.0,
) -> None:
    """Draw one lattice with a light grid and no axes."""

    for row in range(N_ROWS):
        for col in range(N_COLS):
            face_colour = inside_colour if mask[row, col] else COLOURS["white"]
            axis.add_patch(
                Rectangle(
                    (col - 0.5, row - 0.5),
                    1.0,
                    1.0,
                    facecolor=face_colour,
                    edgecolor=COLOURS["grid"],
                    linewidth=0.55,
                    zorder=1,
                )
            )

    axis.set_aspect("equal")
    axis.set_xlim(-1.15, N_COLS - 0.05 + right_padding)
    axis.set_ylim(-1.1, N_ROWS - 0.05)
    axis.axis("off")


def highlight_run(
    axis: mpl.axes.Axes,
    row: int,
    run: tuple[int, int],
    *,
    face_colour: str = "none",
    edge_colour: str = COLOURS["run_edge"],
    alpha: float = 1.0,
    linewidth: float = 1.7,
    linestyle: str = "-",
) -> None:
    """Highlight every cell in one horizontal run."""

    start, end = run
    for col in range(start, end + 1):
        axis.add_patch(
            Rectangle(
                (col - 0.5, row - 0.5),
                1.0,
                1.0,
                facecolor=face_colour,
                edgecolor=edge_colour,
                alpha=alpha,
                linewidth=linewidth,
                linestyle=linestyle,
                zorder=3,
            )
        )


def add_seed(axis: mpl.axes.Axes) -> None:
    axis.scatter(
        [SEED_COL],
        [SEED_ROW],
        s=58,
        color=COLOURS["seed"],
        edgecolors=COLOURS["white"],
        linewidths=0.9,
        zorder=8,
    )


def add_propagation_arrow(
    axis: mpl.axes.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
) -> None:
    """Draw a compact propagation arrow between cell centres."""

    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.5,
            color=COLOURS["propagation"],
            shrinkA=0.0,
            shrinkB=0.0,
            zorder=7,
        )
    )


def add_propagation_highlights(
    axis: mpl.axes.Axes,
    mask: np.ndarray,
    *,
    steps: int = 3,
) -> None:
    """Highlight reached runs and the 8-connected candidate bands.

    The leftmost run is the one reached by continuing from the seed-side
    endpoint.  The dashed frontier is the full one-cell-padded band searched
    in the next row: it includes outside cells and diagonal endpoint cells,
    not only cells that already map inside the source.  The upper bridge also
    produces a gap-repair scan in the row below it, where the padded band
    spans the empty gap and reaches the right-hand run.
    """

    center_runs = find_row_runs(mask, SEED_ROW)
    seed_run = next(run for run in center_runs if run[0] <= SEED_COL <= run[1])
    highlight_run(
        axis,
        SEED_ROW,
        seed_run,
        face_colour=COLOURS["run"],
        alpha=1.0,
        linewidth=1.85,
    )

    for direction in (1, -1):
        for step in range(1, steps + 1):
            row = SEED_ROW + direction * step
            runs = find_row_runs(mask, row)
            if not runs:
                continue
            # Continue from the seed-side endpoint.  Runs on earlier rows are
            # no longer marked as candidates once the frontier has advanced.
            run = runs[0]
            highlight_run(
                axis,
                row,
                run,
                face_colour=COLOURS["run"],
                edge_colour=COLOURS["run_edge"],
                alpha=1.0,
                linewidth=1.55 if step < steps else 1.9,
            )

        next_row = SEED_ROW + direction * (steps + 1)
        reached_runs = find_row_runs(mask, SEED_ROW + direction * steps)
        if not reached_runs:
            continue
        # The multi-run fill examines run.lo - 1 through run.hi + 1 in the
        # neighbouring row.  Draw that complete candidate band, including
        # cells that are outside the recovered image region.
        candidate_band = padded_candidate_band(
            reached_runs[0], mask.shape[1]
        )
        highlight_run(
            axis,
            next_row,
            candidate_band,
            edge_colour=COLOURS["run_edge"],
            alpha=1.0,
            linewidth=1.35,
            linestyle=CANDIDATE_LINESTYLE,
        )

    # The right-hand run below the bridge is the illustrative next candidate
    # in this panel.  The leftmost gap cell was already evaluated as the
    # outside boundary while the earlier left run was expanded.  Only the
    # remaining gap cells, the right run, and its outer endpoint are shown as
    # new candidates.
    bridge_row = SEED_ROW + 3
    right_candidate_row = bridge_row - 1
    bridge_runs = find_row_runs(mask, bridge_row)
    known_runs = find_row_runs(mask, right_candidate_row)
    if len(bridge_runs) == 1 and len(known_runs) == 2:
        right_candidate_band = (
            known_runs[0][1] + 2,
            padded_candidate_band(known_runs[1], mask.shape[1])[1],
        )
        highlight_run(
            axis,
            right_candidate_row,
            right_candidate_band,
                edge_colour=COLOURS["run_edge"],
                alpha=1.0,
                linewidth=1.45,
                linestyle=CANDIDATE_LINESTYLE,
        )
        add_propagation_arrow(
            axis,
            (right_candidate_band[1] + 1, right_candidate_row),
            (
                right_candidate_band[1] + 1,
                right_candidate_row - PROPAGATION_ARROW_LENGTH,
            ),
        )

    arrow_x = seed_run[0] - 2
    add_propagation_arrow(
        axis,
        (arrow_x, SEED_ROW + steps + 1),
        (arrow_x, SEED_ROW + steps + 1 + PROPAGATION_ARROW_LENGTH),
    )
    add_propagation_arrow(
        axis,
        (arrow_x, SEED_ROW - steps - 1),
        (arrow_x, SEED_ROW - steps - 1 - PROPAGATION_ARROW_LENGTH),
    )


def make_figure() -> mpl.figure.Figure:
    mask = build_inside_mask()
    active_run = next(
        run
        for run in find_row_runs(mask, SEED_ROW)
        if run[0] <= SEED_COL <= run[1]
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(9.2, 3.55),
        gridspec_kw={"wspace": -0.10},
    )
    first_axis, second_axis = axes

    # Panel (a): identify the maximal run containing the seed and expose the
    # first two adjacent rows as the full 8-connected candidate bands.
    draw_lattice(
        first_axis,
        mask,
        inside_colour=COLOURS["inside"],
    )
    highlight_run(
        first_axis,
        SEED_ROW,
        active_run,
        face_colour=COLOURS["run"],
        linewidth=1.85,
    )
    candidate_band = padded_candidate_band(active_run, mask.shape[1])
    for direction in (1, -1):
        candidate_row = SEED_ROW + direction
        highlight_run(
            first_axis,
            candidate_row,
            candidate_band,
            edge_colour=COLOURS["run_edge"],
            alpha=1.0,
            linewidth=1.35,
            linestyle=CANDIDATE_LINESTYLE,
        )
    arrow_x = active_run[0] - 2
    add_propagation_arrow(
        first_axis,
        (arrow_x, SEED_ROW + 1),
        (arrow_x, SEED_ROW + 1 + PROPAGATION_ARROW_LENGTH),
    )
    add_propagation_arrow(
        first_axis,
        (arrow_x, SEED_ROW - 1),
        (arrow_x, SEED_ROW - 1 - PROPAGATION_ARROW_LENGTH),
    )
    add_seed(first_axis)

    # Panel (b): follow the same seed through three rows in both directions.
    draw_lattice(
        second_axis,
        mask,
        inside_colour=COLOURS["inside"],
    )
    add_propagation_highlights(second_axis, mask, steps=3)
    add_seed(second_axis)

    legend_handles = [
        Patch(
            facecolor=COLOURS["inside"],
            edgecolor=COLOURS["grid"],
            label="inside source",
        ),
        Patch(
            facecolor=COLOURS["run"],
            edgecolor=COLOURS["run_edge"],
            label="reached run",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor=COLOURS["seed"],
            markeredgecolor=COLOURS["white"],
            markersize=7,
            label="seed",
        ),
        Line2D(
            [],
            [],
            color=COLOURS["run_edge"],
            linestyle=CANDIDATE_LINESTYLE,
            linewidth=1.2,
            label="next candidate",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=4,
        frameon=True,
        framealpha=1.0,
        facecolor=COLOURS["white"],
        edgecolor=COLOURS["grid"],
        fontsize=11.5,
        handlelength=1.25,
        handletextpad=0.4,
        borderpad=0.45,
        columnspacing=0.9,
    )

    figure.subplots_adjust(
        left=0.025,
        right=0.975,
        bottom=0.04,
        top=0.92,
        wspace=-0.10,
    )
    return figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/cartesian_region_recovery"),
        help="Output path without an extension (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure = make_figure()
    figure.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    figure.savefig(
        args.output.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(figure)
    print(args.output.with_suffix(".pdf"))
    print(args.output.with_suffix(".png"))


if __name__ == "__main__":
    main()
