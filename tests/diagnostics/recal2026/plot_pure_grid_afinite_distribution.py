#!/usr/bin/env python3
"""Plot finite-source magnification coverage for the pure-grid sample."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from report_near_caustic_grid_vs_vbm import _load_route_rows, _result_route


PROFILES = ("uniform", "linear")
TARGETS = (1.0e-2, 1.0e-3, 1.0e-4)
AFINITE_BINS = (1.0, 2.0, 5.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0)
AFINITE_LABELS = ("1–2", "2–5", "5–10", "10–30", "30–100",
                  "100–300", "300–1000", "≥1000")


def _rows(payload, route_rows, profile, target):
    return [
        row for row in payload["results"]
        if row.get("status", "completed") == "completed"
        and row["profile"] == profile
        and abs(float(row["target"]) - target) < 1.0e-15
        and _result_route(row, route_rows) == "grid"
    ]


def _values(rows):
    return np.asarray(
        [float(value) for row in rows for value in row["reference"]
         if math.isfinite(float(value))],
        dtype=float,
    )


def _summary(values):
    if not values.size:
        return {"n": 0}
    counts, _ = np.histogram(values, bins=AFINITE_BINS)
    return {
        "n": int(values.size),
        "min": float(np.min(values)),
        "p10": float(np.percentile(values, 10)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
        "under_10": int(np.sum(values < 10.0)),
        "10_to_100": int(np.sum((values >= 10.0) & (values < 100.0))),
        "100_to_1000": int(np.sum((values >= 100.0) & (values < 1000.0))),
        "at_least_1000": int(np.sum(values >= 1000.0)),
        "bins": counts.tolist(),
        "values": values.tolist(),
    }


def _markdown(summary):
    lines = [
        "# A_finite distribution of the pure-grid comparison sample",
        "",
        "This is the finite-source magnification distribution of the current",
        "near-caustic sample, restricted to blocks whose ordinary",
        "`lcbinint_auto` route was exactly `grid`. Each of the four certified",
        "reference epochs contributes one `A_finite` value.",
        "",
        "| profile | target | n | min | p10 | median | p90 | max | <10 | 10–100 | 100–1000 | ≥1000 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for profile in PROFILES:
        for target in TARGETS:
            item = summary[profile][str(target)]
            lines.append(
                f"| {profile} | `{target:g}` | {item['n']} | "
                f"{item['min']:.3g} | {item['p10']:.3g} | "
                f"{item['median']:.3g} | {item['p90']:.3g} | "
                f"{item['max']:.3g} | {item['under_10']} | "
                f"{item['10_to_100']} | {item['100_to_1000']} | "
                f"{item['at_least_1000']} |"
            )
    lines += [
        "",
        "## Counts in logarithmic A_finite bins",
        "",
        "| profile | target | " + " | ".join(AFINITE_LABELS) + " |",
        "|---|---:|" + "---:|" * len(AFINITE_LABELS),
    ]
    for profile in PROFILES:
        for target in TARGETS:
            counts = summary[profile][str(target)]["bins"]
            lines.append(
                f"| {profile} | `{target:g}` | "
                + " | ".join(str(value) for value in counts)
                + " |"
            )
    lines += ["", "Figure: `figures/Afinite_distribution_pure_grid.png`."]
    return "\n".join(lines) + "\n"


def _figure(summary, path):
    import matplotlib.pyplot as plt

    styles = {
        "uniform": ("#2563eb", "uniform"),
        "linear": ("#dc2626", "linear LD"),
    }
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), sharey=True)
    for axis, target in zip(axes, TARGETS):
        for profile, (colour, label) in styles.items():
            item = summary[profile][str(target)]
            values = np.asarray(item["values"], dtype=float)
            axis.hist(values, bins=AFINITE_BINS, histtype="step",
                      linewidth=2.0, color=colour, label=label)
            axis.axvline(item["median"], color=colour, linewidth=0.9,
                         linestyle=":", alpha=0.8)
        axis.set_xscale("log")
        axis.set_xlim(1.0, 3000.0)
        axis.set_xticks([1, 10, 100, 1000])
        axis.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        axis.grid(True, which="major", alpha=0.25)
        axis.set_title(rf"$\epsilon={target:g}$")
        axis.set_xlabel(r"$A_{\rm finite}$")
    axes[0].set_ylabel("number of reference epochs")
    axes[0].legend(frameon=False, loc="upper left")
    figure.suptitle("A_finite distribution of the pure-grid sample", y=1.02)
    figure.tight_layout()
    for suffix in (".pdf", ".png"):
        figure.savefig(str(path) + suffix, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.results.read_text())
    route_rows = _load_route_rows(args.routes)
    summary = {profile: {} for profile in PROFILES}
    for profile in PROFILES:
        for target in TARGETS:
            summary[profile][str(target)] = _summary(
                _values(_rows(payload, route_rows, profile, target))
            )

    args.output.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    (args.output / "REPORT_Afinite_distribution.md").write_text(
        _markdown(summary)
    )
    _figure(summary, figure_dir / "Afinite_distribution_pure_grid")
    serializable = json.loads(json.dumps(summary))
    for profile in PROFILES:
        for target in TARGETS:
            serializable[profile][str(target)].pop("values", None)
    (args.output / "Afinite_distribution_summary.json").write_text(
        json.dumps(serializable, indent=2)
    )
    print(json.dumps(serializable, indent=2))


if __name__ == "__main__":
    main()
