#!/usr/bin/env python3
"""Replay the controlled 12,800-epoch corpus for seeding ablations.

The controlled pure-kernel report contains 3,200 rows with four source-plane
positions each.  This harness reuses its selected Cartesian/polar route and
resolution, its VBM reference, and the underlying corpus geometry.  Rows whose
original native job timed out have no selected plan; those 104 epochs use a
    clearly labelled fixed 50-bin Cartesian fallback so the nominal 12,800
    positions are still exercised without letting an arbitrary replacement for
    a missing production plan dominate the timing comparison.

Each lens cache is warmed at a source position outside the measured four-point
block.  The measured positions themselves are evaluated exactly once, so the
native timing includes point-image seeding but excludes caustic-cache
construction and never measures a memoized duplicate result.

The current binary is certificate-only.  Historical legacy JSON can still be
passed to ``compare``; it need not be reproducible by the current API::

    LCBININT_PROBE_STATS=1 \
      LCBININT_PROBE_BUILD=build python -m \
      tests.diagnostics.recal2026.probe_12800 run --input REPORT --output cert.json

    python -m tests.diagnostics.recal2026.probe_12800 compare \
      historical-legacy.json cert.json --output comparison.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


REFERENCE_INDICES = (0, 7, 15, 23)
BLOCK_EPOCHS = 24
BLOCK_SPAN_IN_RADII = 0.4
STRUCTURAL_DIFFERENCE = 1.0e-9
COUNTER_KEYS = (
    "certified_solves",
    "certified_offered",
    "certified_extrema",
    "certifications",
    "unproven",
    "certified_seconds",
    "certify_seconds",
)


def _finite(value):
    return value is not None and math.isfinite(float(value))


def _relative_error(value, reference):
    if not (_finite(value) and _finite(reference)):
        return None
    return float(abs(float(value) - float(reference)) /
                 max(abs(float(reference)), 1.0))


def _resolve_corpus_path(report_path, payload):
    value = payload.get("corpus") or payload.get("input")
    if not value:
        raise ValueError("report does not identify its source corpus")
    candidate = Path(value)
    if not candidate.is_absolute():
        roots = (*report_path.parents, Path.cwd())
        for root in roots:
            resolved = root / candidate
            if resolved.exists():
                candidate = resolved
                break
    if candidate.is_dir():
        candidate = candidate / "rows.json"
    if not candidate.is_file():
        raise FileNotFoundError(f"controlled corpus was not found: {candidate}")
    return candidate


def _load_rows(path):
    report_path = Path(path).resolve()
    payload = json.loads(report_path.read_text())
    rows = payload.get("results")
    if not isinstance(rows, list) or len(rows) != 3200:
        raise ValueError(
            f"expected the 3,200-row controlled report, got {len(rows or ())} rows"
        )

    corpus_path = _resolve_corpus_path(report_path, payload)
    corpus = json.loads(corpus_path.read_text()).get("rows", ())
    lookup = {
        (
            int(item["case_id"]),
            str(item["profile"]),
            f"{float(item['x']):.17g}",
            f"{float(item['y']):.17g}",
        ): item
        for item in corpus
    }

    joined = []
    for index, original in enumerate(rows):
        row = dict(original)
        key = (
            int(row["case_id"]),
            str(row["profile"]),
            f"{float(row['x']):.17g}",
            f"{float(row['y']):.17g}",
        )
        source = lookup.get(key)
        if source is None:
            raise KeyError(f"row {index} has no matching controlled-corpus geometry")
        for name in ("s", "q", "rho", "limb_darkening_c"):
            if row.get(name) is None:
                row[name] = source[name]

        reference = row.get("reference")
        if not isinstance(reference, list) or len(reference) != 4 or not all(
            _finite(value) for value in reference
        ):
            reference = [
                source["references"][str(epoch)]["value"]
                for epoch in REFERENCE_INDICES
            ]
            row["reference_source"] = "controlled_corpus"
        else:
            row["reference_source"] = "report_vbm"
        row["reference"] = [float(value) for value in reference]
        row["row_index"] = index
        joined.append(row)
    return joined, report_path, corpus_path


def _positions(row):
    rho = float(row["rho"])
    block = np.linspace(
        float(row["x"]) - 0.5 * BLOCK_SPAN_IN_RADII * rho,
        float(row["x"]) + 0.5 * BLOCK_SPAN_IN_RADII * rho,
        BLOCK_EPOCHS,
    )
    return block[list(REFERENCE_INDICES)]


def _plan(row, fallback_nbin, cartesian, polar):
    grids = row.get("chosen_grid")
    bins = row.get("chosen_nbin")
    if (isinstance(grids, list) and isinstance(bins, list) and
            len(grids) == 4 and len(bins) == 4 and
            all(grid in ("cartesian", "polar") for grid in grids) and
            all(value is not None for value in bins)):
        methods = [cartesian if grid == "cartesian" else polar for grid in grids]
        return methods, [int(value) for value in bins], "saved_native"
    return [cartesian] * 4, [int(fallback_nbin)] * 4, "fallback_cartesian"


def _summary(epochs):
    completed = [entry for entry in epochs if entry.get("status") == "completed"]
    seconds = np.asarray([entry["seconds"] for entry in completed], dtype=float)
    errors = np.asarray([
        entry["relative_reference_error"]
        for entry in completed
        if entry.get("relative_reference_error") is not None
    ], dtype=float)
    counters = {
        key: float(sum(entry["counters"][key] for entry in completed))
        for key in COUNTER_KEYS
    }
    result = {
        "epochs": len(epochs),
        "completed": len(completed),
        "errors": len(epochs) - len(completed),
        "support_proven": sum(bool(entry["support_proven"]) for entry in completed),
        "support_unproven": sum(not bool(entry["support_proven"]) for entry in completed),
        "within_target": sum(
            entry.get("relative_reference_error") is not None and
            entry["relative_reference_error"] <= entry["target"]
            for entry in completed
        ),
        "seconds": {
            "total": float(seconds.sum()) if seconds.size else 0.0,
            "median": float(np.median(seconds)) if seconds.size else None,
            "p90": float(np.percentile(seconds, 90)) if seconds.size else None,
            "p99": float(np.percentile(seconds, 99)) if seconds.size else None,
        },
        "reference_error": {
            "median": float(np.median(errors)) if errors.size else None,
            "p99": float(np.percentile(errors, 99)) if errors.size else None,
            "max": float(errors.max()) if errors.size else None,
        },
        "counters": counters,
    }
    return result


def run(arguments):
    if os.environ.get("OMP_NUM_THREADS") not in (None, "1"):
        raise RuntimeError("probe counters are thread-local; use OMP_NUM_THREADS=1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    from . import probe_build

    lcbinint = probe_build.activate()
    from lcbinint import warmup

    native = lcbinint._lcbinint
    if not native.probe_counters()["enabled"]:
        raise RuntimeError("set LCBININT_PROBE_STATS=1 before starting Python")

    rows, report_path, corpus_path = _load_rows(arguments.input)
    if arguments.limit:
        rows = rows[:arguments.limit]

    epochs = []
    started = time.perf_counter()
    for row_number, row in enumerate(rows, 1):
        profile_c = float(row.get("limb_darkening_c") or 0.0)
        target = float(row["target"])
        limb = (lcbinint.LimbDarkening.linear(profile_c) if profile_c
                else lcbinint.LimbDarkening.none())
        curve = lcbinint.LightCurve(
            lens="binary",
            options=lcbinint.Options(
                coordinates="vbm",
                nbin="auto",
                reltol=target,
                max_source_bins=max(int(arguments.fallback_nbin), 800),
            ),
            limb_darkening=limb,
        )
        params = {
            "t0": 0.0,
            "tE": 1.0,
            "u0": float(row["y"]),
            "alpha": 0.0,
            "s": float(row["s"]),
            "q": float(row["q"]),
            "rho": float(row["rho"]),
        }
        positions = _positions(row)
        methods, resolutions, plan_source = _plan(
            row, arguments.fallback_nbin, warmup.CARTESIAN, warmup.POLAR
        )

        # Build the lens and caustic cache without memoizing any measured point.
        warm_x = float(row["x"]) + 3.0 * float(row["rho"])
        curve._native._evaluate_preplanned_xy(
            np.asarray([warm_x]),
            np.asarray([float(row["y"])]),
            params,
            [methods[0]],
            [resolutions[0]],
        )

        for epoch_index, (source_x, method, resolution, reference) in enumerate(
            zip(positions, methods, resolutions, row["reference"])
        ):
            native.reset_probe_counters()
            try:
                result = curve._native._evaluate_preplanned_xy(
                    np.asarray([float(source_x)]),
                    np.asarray([float(row["y"])]),
                    params,
                    [int(method)],
                    [int(resolution)],
                )
                magnification = float(np.asarray(result["magnification"])[0])
                error_estimate = float(np.asarray(result["error_estimate"])[0])
                counters = native.probe_counters()
                entry = {
                    "status": "completed",
                    "row_index": int(row["row_index"]),
                    "epoch_index": epoch_index,
                    "case_id": int(row["case_id"]),
                    "profile": str(row["profile"]),
                    "target": target,
                    "source_x": float(source_x),
                    "source_y": float(row["y"]),
                    "method": "cartesian" if method == warmup.CARTESIAN else "polar",
                    "resolution": int(resolution),
                    "plan_source": plan_source,
                    "reference_source": row["reference_source"],
                    "reference": float(reference),
                    "magnification": magnification,
                    "relative_reference_error": _relative_error(
                        magnification, reference
                    ),
                    "seconds": float(np.asarray(result["seconds"])[0]),
                    "error_estimate": error_estimate,
                    "converged": bool(np.asarray(result["converged"])[0]),
                    "support_proven": math.isfinite(error_estimate),
                    "counters": {key: counters[key] for key in COUNTER_KEYS},
                }
            except Exception as error:  # noqa: BLE001
                entry = {
                    "status": "error",
                    "row_index": int(row["row_index"]),
                    "epoch_index": epoch_index,
                    "case_id": int(row["case_id"]),
                    "profile": str(row["profile"]),
                    "target": target,
                    "source_x": float(source_x),
                    "source_y": float(row["y"]),
                    "method": "cartesian" if method == warmup.CARTESIAN else "polar",
                    "resolution": int(resolution),
                    "plan_source": plan_source,
                    "error": f"{type(error).__name__}: {error}",
                }
            epochs.append(entry)

        if row_number % 50 == 0 or row_number == len(rows):
            rate = (4.0 * row_number) / max(time.perf_counter() - started, 1.0e-9)
            print(f"[{row_number}/{len(rows)}] {rate:.1f} epochs/s", flush=True)

    payload = {
        "policy": native.probe_counters()["policy"],
        "build": str(Path(probe_build.BUILD).resolve()),
        "input": str(report_path),
        "corpus": str(corpus_path),
        "fallback_nbin": int(arguments.fallback_nbin),
        "wall_seconds": time.perf_counter() - started,
        "summary": _summary(epochs),
        "epochs": epochs,
    }
    Path(arguments.output).write_text(json.dumps(payload))
    print(json.dumps(payload["summary"], indent=2))
    print(f"wrote {arguments.output}")


def compare(arguments):
    before = json.loads(Path(arguments.before).read_text())
    after = json.loads(Path(arguments.after).read_text())
    before_by_key = {
        (entry["row_index"], entry["epoch_index"]): entry
        for entry in before["epochs"]
    }
    after_by_key = {
        (entry["row_index"], entry["epoch_index"]): entry
        for entry in after["epochs"]
    }

    compared = []
    newly_unproven = []
    structural = []
    target_regressions = []
    materially_worse = []
    for key in sorted(set(before_by_key) & set(after_by_key)):
        old = before_by_key[key]
        new = after_by_key[key]
        if old.get("status") != "completed" or new.get("status") != "completed":
            continue
        difference = _relative_error(new["magnification"], old["magnification"])
        record = {
            "row_index": key[0],
            "epoch_index": key[1],
            "case_id": old["case_id"],
            "profile": old["profile"],
            "target": old["target"],
            "method": old["method"],
            "resolution": old["resolution"],
            "before": old["magnification"],
            "after": new["magnification"],
            "relative_difference": difference,
            "before_reference_error": old.get("relative_reference_error"),
            "after_reference_error": new.get("relative_reference_error"),
        }
        compared.append(record)
        if old["support_proven"] and not new["support_proven"]:
            newly_unproven.append(record)
        if difference is not None and difference > STRUCTURAL_DIFFERENCE:
            structural.append(record)
        old_error = old.get("relative_reference_error")
        new_error = new.get("relative_reference_error")
        if old_error is not None and new_error is not None:
            if old_error <= old["target"] and new_error > new["target"]:
                target_regressions.append(record)
            if new_error > max(10.0 * old_error, 1.0e-6):
                materially_worse.append(record)

    old_seconds = np.asarray([
        before_by_key[(entry["row_index"], entry["epoch_index"])]["seconds"]
        for entry in after["epochs"]
        if entry.get("status") == "completed" and
        (entry["row_index"], entry["epoch_index"]) in before_by_key and
        before_by_key[(entry["row_index"], entry["epoch_index"])].get("status") == "completed"
    ], dtype=float)
    new_seconds = np.asarray([
        entry["seconds"] for entry in after["epochs"]
        if entry.get("status") == "completed" and
        (entry["row_index"], entry["epoch_index"]) in before_by_key and
        before_by_key[(entry["row_index"], entry["epoch_index"])].get("status") == "completed"
    ], dtype=float)

    def worst(rows, field, count=20):
        return sorted(
            rows,
            key=lambda entry: -(entry.get(field) or 0.0),
        )[:count]

    result = {
        "before_policy": before.get("policy"),
        "after_policy": after.get("policy"),
        "compared_epochs": len(compared),
        "newly_unproven": len(newly_unproven),
        "structural_differences": len(structural),
        "target_regressions": len(target_regressions),
        "materially_worse_against_reference": len(materially_worse),
        "timing": {
            "before_total_seconds": float(old_seconds.sum()),
            "after_total_seconds": float(new_seconds.sum()),
            "speedup_total": float(old_seconds.sum() / new_seconds.sum()),
            "before_median_seconds": float(np.median(old_seconds)),
            "after_median_seconds": float(np.median(new_seconds)),
            "median_epoch_speedup": float(np.median(old_seconds / new_seconds)),
        },
        "worst_structural_differences": worst(
            structural, "relative_difference"
        ),
        "worst_target_regressions": worst(
            target_regressions, "after_reference_error"
        ),
        "worst_materially_worse": worst(
            materially_worse, "after_reference_error"
        ),
        "newly_unproven_rows": newly_unproven[:20],
        "before_summary": before.get("summary"),
        "after_summary": after.get("summary"),
    }
    rendered = json.dumps(result, indent=2)
    if arguments.output:
        Path(arguments.output).write_text(rendered)
    print(rendered)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    runner = subparsers.add_parser("run")
    runner.add_argument("--input", required=True, type=Path)
    runner.add_argument("--output", required=True, type=Path)
    runner.add_argument("--fallback-nbin", type=int, default=50)
    runner.add_argument("--limit", type=int, default=0)
    runner.set_defaults(function=run)

    comparison = subparsers.add_parser("compare")
    comparison.add_argument("before", type=Path)
    comparison.add_argument("after", type=Path)
    comparison.add_argument("--output", type=Path)
    comparison.set_defaults(function=compare)

    arguments = parser.parse_args()
    arguments.function(arguments)


if __name__ == "__main__":
    main()
