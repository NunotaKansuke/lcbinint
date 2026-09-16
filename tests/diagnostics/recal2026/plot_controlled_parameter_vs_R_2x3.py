#!/usr/bin/env python3
"""Render the paper-facing 2-by-3 controlled speed-ratio scatter figure.

The figure layout follows ``controlled_parameter_vs_R_2x3_profiles_20260813``:
rows are the requested relative tolerances and columns show ``rho``, measured
``d/rho``, and finite-source magnification.  Each point is one measured
reference epoch and ``R = t_VBM / t_lcbinint``.

The 2026-09 coverage run contains both the canonical 160-configuration base
corpus and an intentionally non-iid q--rho coverage extension.  The default
scope is the canonical base so that the paper-facing figure keeps the same
statistical meaning as the 2026-08 figure.  Use ``--scope all`` for the
separate coverage diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


PROFILES = (
    ("uniform", "Uniform", "#2563eb", "o"),
    ("linear", "Linear LD", "#dc2626", "^"),
)
TARGETS = (1.0e-3, 1.0e-4)

Y_MIN = 1.0e-3
DEFAULT_Y_MAX = 1.0e3


def _finite_positive(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value <= 0.0:
        return None
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _row_key(row):
    return (
        int(row["case_id"]),
        str(row["profile"]),
        float(row["target"]),
        float(row["x"]),
        float(row["y"]),
    )


def _overlay_reference_parts(payload, parts_dir: Path):
    """Replace raw VBM values with the accuracy-only recheck values.

    The recheck parts intentionally preserve timing and ``R`` fields but do
    not carry the joined ``actual_d_over_rho`` metadata.  Keep the raw merged
    rows as the metadata/timing source and overlay only the VBM reference
    arrays, checking that the speed ratios are unchanged.
    """

    part_paths = sorted(parts_dir.glob("*/results.json"))
    if not part_paths:
        raise FileNotFoundError(f"no derived results.json under {parts_dir}")
    updates = {}
    for path in part_paths:
        part = json.loads(path.read_text())
        for row in part.get("results", ()):
            key = _row_key(row)
            if key in updates:
                raise RuntimeError(f"duplicate derived reference row: {key}")
            updates[key] = row

    merged_rows = []
    missing = []
    for raw_row in payload.get("results", ()):
        key = _row_key(raw_row)
        update = updates.get(key)
        if update is None:
            missing.append(key)
            continue
        if list(raw_row.get("ratios_vbm_over_lcbinint", ())) != list(
            update.get("ratios_vbm_over_lcbinint", ())
        ):
            raise RuntimeError(f"speed ratio changed in derived reference: {key}")
        row = dict(raw_row)
        row["reference"] = update.get("reference")
        row["reference_status"] = update.get("reference_status")
        row["reference_mode"] = update.get("reference_mode")
        row["accuracy_reference_reltol"] = update.get(
            "accuracy_reference_reltol"
        )
        merged_rows.append(row)
    if missing:
        raise RuntimeError(
            f"{len(missing)} raw rows have no derived reference; first={missing[0]}"
        )
    if len(merged_rows) != len(updates):
        raise RuntimeError(
            f"derived/raw row count mismatch: raw={len(merged_rows)} "
            f"derived={len(updates)}"
        )
    updated = dict(payload)
    updated["results"] = merged_rows
    updated["reference_mode"] = "accuracy_only_full_recheck"
    updated["accuracy_reference_reltol"] = 1.0e-6
    updated["reference_parts_dir"] = str(parts_dir.resolve())
    return updated, part_paths


def _selected_row(row, scope: str, base_case_count: int) -> bool:
    if row.get("status", "completed") != "completed":
        return False
    if scope == "all":
        return True
    return int(row["case_id"]) < base_case_count


def _points(payload, scope: str, base_case_count: int):
    points = {
        (profile, target): []
        for profile, _, _, _ in PROFILES
        for target in TARGETS
    }
    selected_rows = 0
    skipped_rows = 0
    for row in payload.get("results", ()):
        if not _selected_row(row, scope, base_case_count):
            continue
        selected_rows += 1
        profile = row.get("profile")
        if profile not in {item[0] for item in PROFILES}:
            skipped_rows += 1
            continue
        try:
            target = float(row["target"])
        except (KeyError, TypeError, ValueError):
            skipped_rows += 1
            continue
        target_key = next(
            (candidate for candidate in TARGETS
             if abs(target - candidate) < 1.0e-15),
            None,
        )
        if target_key is None:
            skipped_rows += 1
            continue
        actual_d_over_rho = _finite_positive(row.get("actual_d_over_rho"))
        rho = _finite_positive(row.get("rho"))
        if actual_d_over_rho is None or rho is None:
            skipped_rows += 1
            continue
        references = row.get("reference", ())
        ratios = row.get("ratios_vbm_over_lcbinint", ())
        statuses = row.get("ratio_status", ())
        for a_finite, ratio, status in zip(references, ratios, statuses):
            if status != "measured":
                continue
            a_finite = _finite_positive(a_finite)
            ratio = _finite_positive(ratio)
            if a_finite is None or ratio is None:
                continue
            points[(profile, target_key)].append(
                {
                    "rho": rho,
                    "d_over_rho": actual_d_over_rho,
                    "a_finite": a_finite,
                    "ratio": ratio,
                }
            )
    return points, selected_rows, skipped_rows


def _stats(values):
    import numpy as np

    values = np.asarray(values, dtype=float)
    if not values.size:
        return {"count": 0}
    return {
        "count": int(values.size),
        "win_count": int((values > 1.0).sum()),
        "win_rate": float((values > 1.0).mean()),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
    }


def _render(points, output: Path, y_limits):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    columns = (
        ("rho", r"$\rho$", True, (3.0e-5, 1.0),
         [1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0],
         [r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$"]),
        ("d_over_rho", r"$d/\rho$", False, (-0.03, 2.03),
         [0.0, 0.5, 1.0, 1.5, 2.0], ["0", "0.5", "1", "1.5", "2"]),
        ("a_finite", r"$A_{\rm finite}$", True, (1.0, 2.0e4),
         [1.0, 10.0, 100.0, 1000.0, 10000.0],
         [r"$10^0$", r"$10^1$", r"$10^2$", r"$10^3$", r"$10^4$"]),
    )

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(12.3, 6.7),
        sharey=True,
        gridspec_kw={"wspace": 0.08, "hspace": 0.12},
    )

    for row_index, target in enumerate(TARGETS):
        exponent = int(round(math.log10(target)))
        for column_index, (field, xlabel, log_x, x_limits, ticks, labels) in enumerate(columns):
            axis = axes[row_index, column_index]
            for profile, _, colour, marker in PROFILES:
                selected = points[(profile, target)]
                if not selected:
                    continue
                axis.scatter(
                    [item[field] for item in selected],
                    [item["ratio"] for item in selected],
                    s=18,
                    marker=marker,
                    color=colour,
                    alpha=0.38,
                    linewidths=0.0,
                    rasterized=False,
                )
            axis.axhline(1.0, color="black", linewidth=0.8, zorder=1)
            axis.set_xlim(*x_limits)
            axis.set_ylim(*y_limits)
            axis.set_yscale("log")
            axis.set_xticks(ticks)
            axis.set_xticklabels(labels)
            if log_x:
                axis.set_xscale("log")
            if row_index == len(TARGETS) - 1:
                axis.set_xlabel(xlabel)
            if column_index == 0:
                axis.set_ylabel(
                    rf"$t_{{\rm VBM}}/t_{{\rm lcbinint}}$ "
                    rf"($\epsilon_{{\rm rel}}=10^{{{exponent}}}$)"
                )
            else:
                axis.tick_params(labelleft=False)

    legend = [
        Line2D(
            [0], [0], marker=marker, linestyle="None", color=colour,
            markerfacecolor=colour, markeredgecolor=colour,
            markersize=6.5, alpha=0.55, label=label,
        )
        for _, label, colour, marker in PROFILES
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
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(output) + ".pdf", dpi=240, bbox_inches="tight")
    figure.savefig(str(output) + ".png", dpi=240, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="output prefix without .pdf/.png")
    parser.add_argument("--scope", choices=("canonical", "all"),
                        default="canonical")
    parser.add_argument("--base-case-count", type=int, default=160)
    parser.add_argument(
        "--y-max",
        type=float,
        default=DEFAULT_Y_MAX,
        help="upper limit of the shared logarithmic R axis (default: 1e3)",
    )
    parser.add_argument(
        "--reference-parts-dir",
        type=Path,
        default=None,
        help=(
            "directory containing accuracy-only derived */results.json parts; "
            "overlay their VBM reference values while preserving raw timing/R"
        ),
    )
    args = parser.parse_args()
    if not math.isfinite(args.y_max) or args.y_max <= Y_MIN:
        parser.error(f"--y-max must be finite and greater than {Y_MIN:g}")

    payload = json.loads(args.results.read_text())
    reference_parts = []
    if args.reference_parts_dir is not None:
        payload, reference_parts = _overlay_reference_parts(
            payload, args.reference_parts_dir
        )
    points, selected_rows, skipped_rows = _points(
        payload, args.scope, args.base_case_count
    )
    y_limits = (Y_MIN, args.y_max)
    _render(points, args.output, y_limits)

    summary = {
        "figure": "controlled_parameter_vs_R_2x3_profiles",
        "scope": args.scope,
        "base_case_count": args.base_case_count,
        "results": str(args.results.resolve()),
        "results_sha256": _sha256(args.results),
        "reference_parts_dir": (
            str(args.reference_parts_dir.resolve())
            if args.reference_parts_dir is not None else None
        ),
        "reference_parts": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in reference_parts
        ],
        "build_extension": payload.get("build_extension"),
        "reference_mode": payload.get("reference_mode"),
        "accuracy_reference_reltol": payload.get("accuracy_reference_reltol"),
        "reference_indices": payload.get("reference_indices"),
        "y_limits": list(y_limits),
        "total_result_rows": len(payload.get("results", ())),
        "selected_result_rows": selected_rows,
        "skipped_rows": skipped_rows,
        "points": {
            f"{profile}:{target:g}": _stats(
                [item["ratio"] for item in points[(profile, target)]]
            )
            for profile, _, _, _ in PROFILES
            for target in TARGETS
        },
    }
    summary_path = Path(str(args.output) + ".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
