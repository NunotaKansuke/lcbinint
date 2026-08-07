#!/usr/bin/env python3
"""Render the absolute-branch lower-censored holdout figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from matplotlib.backends.backend_pdf import PdfPages

from .error_budget_percentiles import (
    _plot_boxplots,
    _records,
    _summary,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        default=(
            "tests/diagnostics/results/recal2026/absolute_error_law/"
            "absolute_error_records.csv"),
    )
    parser.add_argument(
        "--report",
        default=(
            "tests/diagnostics/results/recal2026/absolute_error_law/"
            "absolute_error_law.json"),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    records = _records(Path(args.records))
    report = json.loads(Path(args.report).read_text())
    summary = _summary(records, report, "absolute")
    figure = _plot_boxplots(
        summary,
        "Absolute branch: holdout required Nbin lower bounds",
        "absolute tolerance $a_{\\rm tol}$",
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as pdf:
        pdf.savefig(figure, bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
