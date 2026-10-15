#!/usr/bin/env python3
"""Make a six-panel map from the controlled pure-kernel benchmark.

The input is the 20-part controlled rerun used for the paper-facing
54.4/59.4 percent result.  Parts are read in numerical order and one part is
discarded before the next one is loaded.  This keeps the plotting pass
serial and avoids retaining the large nested timing payloads in memory.

The top row shows the median

    R = t_VBM / t_lcbinint

in logarithmic parameter bins.  The bottom row shows the p95 of the stored
cross-code relative difference against the VBM value.  That lower quantity is
an agreement diagnostic, not an independent accuracy certificate, because
the two kernels use independent stopping rules in this benchmark.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap


SPEED_BOUNDARIES = np.asarray(
    (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0),
    dtype=float,
)
SPEED_COLOURS = (
    "#2166ac", "#67a9cf", "#d1e5f0", "#f7f7f7", "#fddbc7",
    "#f4a582", "#ef8a62", "#d73027", "#b2182b", "#67001f",
)
SPEED_LABELS = (
    r"$<0.25\times$", r"$0.25\times$", r"$0.5\times$",
    r"$1\times$", r"$2\times$", r"$4\times$", r"$8\times$",
    r"$16\times$", r"$32\times$", r"$\geq64\times$",
)

ERROR_BOUNDARIES = np.asarray(
    (1.0e-4, 2.0e-4, 5.0e-4, 1.0e-3, 2.0e-3, 5.0e-3, 1.0e-2),
    dtype=float,
)
ERROR_COLOURS = (
    "#006d2c", "#31a354", "#74c476", "#d9ef8b", "#fee08b",
    "#fdae61", "#f46d43", "#bd0026",
)
ERROR_LABELS = (
    r"$\leq10^{-4}$", r"$2\times10^{-4}$", r"$5\times10^{-4}$",
    r"$10^{-3}$", r"$2\times10^{-3}$", r"$5\times10^{-3}$",
    r"$10^{-2}$", r"$\geq10^{-2}$",
)


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _format_target(target: float) -> str:
    exponent = int(round(math.log10(target)))
    if math.isclose(target, 10.0**exponent, rel_tol=1.0e-12, abs_tol=0.0):
        return rf"10^{{{exponent}}}"
    return f"{target:g}"


def _part_sort_key(path: Path) -> tuple[tuple[int, ...], str]:
    """Sort d02_c0, ..., d18_c3 by their numeric labels."""

    numbers = tuple(int(item) for item in re.findall(r"\d+", path.parent.name))
    return numbers, path.parent.name


def _part_paths(parts_dir: Path) -> list[Path]:
    paths = sorted(parts_dir.glob("*/results.json"), key=_part_sort_key)
    if not paths:
        raise FileNotFoundError(f"no */results.json parts found under {parts_dir}")
    return paths


def _load_records(
    parts_dir: Path,
    profile: str,
    target: float,
) -> tuple[list[dict], list[str]]:
    """Read one part at a time and retain only compact plotting records."""

    records: list[dict] = []
    provenance: list[str] = []
    paths = _part_paths(parts_dir)
    for part_index, path in enumerate(paths, 1):
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if payload.get("timing_mode") != "pure_kernel_cache_warm_direct_xy":
            raise ValueError(
                f"{path} is not the controlled direct-XY pure-kernel result: "
                f"timing_mode={payload.get('timing_mode')!r}"
            )
        if payload.get("coordinate_mode") != "direct_internal_source_xy":
            raise ValueError(f"{path} has an unexpected coordinate mode")

        part_rows = sorted(
            payload.get("results", ()),
            key=lambda row: (
                int(row.get("case_id", -1)),
                str(row.get("profile", "")),
                float(row.get("target", float("inf"))),
            ),
        )
        for row in part_rows:
            if row.get("profile") != profile:
                continue
            if not _finite(row.get("target")):
                continue
            if not math.isclose(
                float(row["target"]), target, rel_tol=0.0, abs_tol=1.0e-15
            ):
                continue
            references = row.get("reference", ())
            ratios = row.get("ratios_vbm_over_lcbinint", ())
            statuses = row.get("ratio_status", ())
            errors = row.get("chosen_vbm_errors", ())
            if not (
                len(references) == len(ratios) == len(statuses)
            ):
                raise ValueError(
                    f"inconsistent reference/ratio lengths in {path}, "
                    f"case {row.get('case_id')}"
                )
            for epoch_index, (reference, ratio, status) in enumerate(
                zip(references, ratios, statuses)
            ):
                if status != "measured":
                    continue
                if not all(
                    _finite(value) and float(value) > 0.0
                    for value in (row.get("q"), row.get("rho"), reference, ratio)
                ):
                    continue
                error = (
                    errors[epoch_index]
                    if epoch_index < len(errors)
                    else None
                )
                records.append(
                    {
                        "case_id": int(row["case_id"]),
                        "epoch_index": int(epoch_index),
                        "q": float(row["q"]),
                        "rho": float(row["rho"]),
                        "a_finite": float(reference),
                        "ratio": float(ratio),
                        "error": float(error) if _finite(error) else None,
                    }
                )
        provenance.append(str(path))
        print(
            f"[{part_index}/{len(paths)}] {path.parent.name}: "
            f"retained {len(records)} compact records; releasing payload",
            flush=True,
        )
        # The raw part contains nested grid, reference, and engine metadata.
        # Only the compact records above survive the part boundary.
        del part_rows, payload
        gc.collect()

    records.sort(key=lambda item: (item["case_id"], item["epoch_index"]))
    return records, provenance


def _log_edges(
    records: list[dict],
    field: str,
    default_low: float,
    default_high: float,
    bins: int = 12,
) -> np.ndarray:
    values = np.asarray([record[field] for record in records], dtype=float)
    observed_low = float(np.min(values))
    observed_high = float(np.max(values))
    # Keep the requested plotting domain when the data are already inside it.
    # Flooring an in-range minimum to the preceding decade creates an extra
    # mostly empty bin (notably rho: 3e-5 was expanded down to 1e-5).
    low = (
        10.0 ** math.floor(math.log10(observed_low))
        if observed_low < default_low
        else default_low
    )
    high = (
        10.0 ** math.ceil(math.log10(observed_high))
        if observed_high > default_high
        else default_high
    )
    return np.geomspace(low, high, bins + 1)


def _cell_statistics(
    records: list[dict],
    x_field: str,
    y_field: str,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    minimum_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (len(y_edges) - 1, len(x_edges) - 1)
    ratio_buckets: list[list[list[float]]] = [
        [[] for _ in range(shape[1])] for _ in range(shape[0])
    ]
    error_buckets: list[list[list[float]]] = [
        [[] for _ in range(shape[1])] for _ in range(shape[0])
    ]
    for record in records:
        x_index = int(np.searchsorted(x_edges, record[x_field], side="right") - 1)
        y_index = int(np.searchsorted(y_edges, record[y_field], side="right") - 1)
        if record[x_field] == x_edges[-1]:
            x_index -= 1
        if record[y_field] == y_edges[-1]:
            y_index -= 1
        if not (0 <= x_index < shape[1] and 0 <= y_index < shape[0]):
            continue
        ratio_buckets[y_index][x_index].append(record["ratio"])
        if record["error"] is not None and _finite(record["error"]):
            error_buckets[y_index][x_index].append(record["error"])

    counts = np.zeros(shape, dtype=int)
    speed = np.full(shape, np.nan, dtype=float)
    error = np.full(shape, np.nan, dtype=float)
    for y_index in range(shape[0]):
        for x_index in range(shape[1]):
            ratios = ratio_buckets[y_index][x_index]
            counts[y_index, x_index] = len(ratios)
            if len(ratios) < minimum_count:
                continue
            speed[y_index, x_index] = float(np.median(ratios))
            errors = error_buckets[y_index][x_index]
            if errors:
                error[y_index, x_index] = float(np.percentile(errors, 95.0))
    return counts, speed, error


def _class_norm(colour_count: int) -> BoundaryNorm:
    return BoundaryNorm(
        np.arange(-0.5, colour_count + 0.5, 1.0),
        colour_count,
    )


def _speed_class(values: np.ndarray) -> np.ndarray:
    classes = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    classes[finite] = np.digitize(values[finite], SPEED_BOUNDARIES, right=False)
    return classes


def _error_class(values: np.ndarray) -> np.ndarray:
    classes = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    classes[finite] = np.digitize(values[finite], ERROR_BOUNDARIES, right=False)
    return classes


def _style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor("#e5e5e5")
    axis.grid(True, which="major", color="#9ca3af", alpha=0.28, linewidth=0.55)
    axis.tick_params(axis="both", labelsize=10)
    for spine in axis.spines.values():
        spine.set_color("#374151")
        spine.set_linewidth(0.8)


def _draw_map(
    axis: plt.Axes,
    records: list[dict],
    x_field: str,
    y_field: str,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    minimum_count: int,
    kind: str,
) -> tuple[np.ndarray, np.ndarray]:
    counts, speed, error = _cell_statistics(
        records, x_field, y_field, x_edges, y_edges, minimum_count
    )
    _style_axis(axis)
    if kind == "speed":
        classes = _speed_class(speed)
        cmap = ListedColormap(SPEED_COLOURS)
        norm = _class_norm(len(SPEED_COLOURS))
    else:
        classes = _error_class(error)
        cmap = ListedColormap(ERROR_COLOURS)
        norm = _class_norm(len(ERROR_COLOURS))
    axis.pcolormesh(
        x_edges,
        y_edges,
        np.ma.masked_invalid(classes),
        cmap=cmap,
        norm=norm,
        shading="flat",
        edgecolors="#a3a3a3",
        linewidth=0.28,
        rasterized=True,
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(x_edges[0], x_edges[-1])
    axis.set_ylim(y_edges[0], y_edges[-1])
    return counts, speed if kind == "speed" else error


def _format_axis(axis: plt.Axes, x_label: str, y_label: str) -> None:
    axis.set_xlabel(x_label, fontsize=12)
    axis.set_ylabel(y_label, fontsize=12)


def _add_colourbar(
    figure: plt.Figure,
    grid_axis: plt.Axes,
    colours: tuple[str, ...],
    labels: tuple[str, ...],
    label: str,
) -> None:
    cmap = ListedColormap(colours)
    norm = _class_norm(len(colours))
    colourbar = figure.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=grid_axis,
        orientation="horizontal",
    )
    colourbar.set_ticks(np.arange(len(labels), dtype=float))
    colourbar.set_ticklabels(labels)
    colourbar.ax.tick_params(labelsize=10, length=3, pad=4)
    colourbar.set_label(label, fontsize=11, labelpad=7)


def _write_summary(
    output: Path,
    records: list[dict],
    provenance: list[str],
    profile: str,
    target: float,
    minimum_count: int,
) -> None:
    ratios = np.asarray(
        [record["ratio"] for record in records if _finite(record["ratio"])],
        dtype=float,
    )
    errors = np.asarray(
        [record["error"] for record in records if _finite(record["error"])],
        dtype=float,
    )
    lines = [
        "# Controlled pure-kernel speed/diagnostic map",
        "",
        f"Profile: `{profile}`",
        f"Target: `{target:g}`",
        f"Measured epoch points: `{len(records)}`",
        f"Input parts: `{len(provenance)}` (read serially in numeric order)",
        f"Minimum cell population: `{minimum_count}`",
        f"Gray cells are masked: the upper map has fewer than `{minimum_count}` finite timing ratios; the lower map has fewer than `{minimum_count}` finite cross-code errors.",
        "",
        "The top row is the cell median of "
        "`R=t_VBM/t_lcbinint`; `R>1` favours lcbinint.",
        "The bottom row is the cell p95 of the stored cross-code relative "
        "difference against VBM. It is a disagreement diagnostic, not an "
        "independent accuracy certificate.",
        "",
        "| statistic | value |",
        "|---|---:|",
        f"| lcbinint win rate (`R>1`) | `{np.mean(ratios > 1.0):.3%}` |",
        f"| median `R` | `{np.median(ratios):.6g}` |",
        f"| p10 `R` | `{np.percentile(ratios, 10):.6g}` |",
        f"| p90 `R` | `{np.percentile(ratios, 90):.6g}` |",
    ]
    if errors.size:
        lines.extend([
            f"| cross-code error records | `{len(errors)}` |",
            f"| cross-code error p95 | `{np.percentile(errors, 95):.6g}` |",
        ])
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_figure(
    records: list[dict],
    provenance: list[str],
    output: Path,
    profile: str,
    target: float,
    minimum_count: int,
    speed_ratio_label: str = r"median $R=t_{\rm VBM}/t_{\rm lcbinint}$",
    error_label: str = "cell p95 cross-code relative difference",
) -> None:
    q_edges = _log_edges(records, "q", 1.0e-4, 1.0)
    rho_edges = _log_edges(records, "rho", 3.0e-5, 1.0)
    a_edges = _log_edges(records, "a_finite", 1.0, 1.0e5)
    panels = (
        ("q", "rho", r"$q$", r"$\rho$", r"$q\times\rho$"),
        ("a_finite", "rho", r"$A_{\rm fs}$", r"$\rho$",
         r"$A_{\rm fs}\times\rho$"),
        ("a_finite", "q", r"$A_{\rm fs}$", r"$q$",
         r"$A_{\rm fs}\times q$"),
    )
    edges = {"q": q_edges, "rho": rho_edges, "a_finite": a_edges}

    figure = plt.figure(figsize=(15.5, 9.2), facecolor="white")
    grid = figure.add_gridspec(
        4,
        3,
        height_ratios=(1.0, 0.15, 1.0, 0.15),
        hspace=0.82,
        wspace=0.20,
        left=0.075,
        right=0.975,
        top=0.88,
        bottom=0.095,
    )
    top_axes = [figure.add_subplot(grid[0, index]) for index in range(3)]
    bottom_axes = [figure.add_subplot(grid[2, index]) for index in range(3)]
    top_cbar_axis = figure.add_subplot(grid[1, :])
    bottom_cbar_axis = figure.add_subplot(grid[3, :])

    for index, (x_field, y_field, x_label, y_label, title) in enumerate(panels):
        x_edges = edges[x_field]
        y_edges = edges[y_field]
        _draw_map(
            top_axes[index], records, x_field, y_field, x_edges, y_edges,
            minimum_count, "speed",
        )
        _draw_map(
            bottom_axes[index], records, x_field, y_field, x_edges, y_edges,
            minimum_count, "error",
        )
        top_axes[index].set_title(f"{title} — runtime", fontsize=13, pad=8)
        bottom_axes[index].set_title(
            f"{title} — p95 difference", fontsize=13, pad=8
        )
        _format_axis(top_axes[index], x_label, y_label)
        _format_axis(bottom_axes[index], x_label, y_label)

    _add_colourbar(
        figure,
        top_cbar_axis,
        SPEED_COLOURS,
        SPEED_LABELS,
        speed_ratio_label,
    )
    _add_colourbar(
        figure,
        bottom_cbar_axis,
        ERROR_COLOURS,
        ERROR_LABELS,
        error_label,
    )

    profile_title = (
        "linear limb darkening" if profile == "linear" else "uniform source"
    )
    figure.suptitle(
        rf"{profile_title}, $\epsilon_{{\rm rel}}={_format_target(target)}$",
        fontsize=16,
        y=0.965,
    )
    figure.text(
        0.5,
        0.018,
        "Gray cells: <8 finite ratios (top) or cross-code errors (bottom)",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#374151",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)
    gc.collect()
    _write_summary(output, records, provenance, profile, target, minimum_count)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("uniform", "linear"), default="linear")
    parser.add_argument("--target", type=float, default=1.0e-3)
    parser.add_argument("--min-count", type=int, default=10)
    args = parser.parse_args()
    if not _finite(args.target) or args.target <= 0.0:
        raise SystemExit("--target must be positive and finite")
    if args.min_count < 1:
        raise SystemExit("--min-count must be positive")

    records, provenance = _load_records(args.parts_dir, args.profile, args.target)
    if not records:
        raise SystemExit("no measured records found for the requested profile/target")
    make_figure(
        records,
        provenance,
        args.output,
        args.profile,
        args.target,
        args.min_count,
    )
    print(args.output.with_suffix(".pdf"))
    print(args.output.with_suffix(".png"))
    print(args.output.with_suffix(".md"))


if __name__ == "__main__":
    main()
