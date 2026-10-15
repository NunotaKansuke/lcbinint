#!/usr/bin/env python3
"""Render matched six-panel maps for the measured faster Cartesian/Polar route.

The per-input winner is selected only from the median of three seed-inclusive
cold timings. Accuracy is audited after selection against the saved,
adjudicated high-accuracy reference; the reference does not choose the winner.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN = ROOT / "tests/diagnostics/results/recal2026/seed-inclusive-cart-polar-20261015/full"
INPUTS = ROOT / "figures/recal2026/adaptive-vbm-six-panel-20261014/inputs.tsv"
VBM_JOINED = ROOT / "figures/recal2026/polar-relative-matched-six-panel-20261015/joined.tsv"
ADJUDICATIONS = ROOT / "figures/recal2026/polar-relative-matched-six-panel-20261015/reference-adjudications.tsv"
STYLE = ROOT / "tests/diagnostics/recal2026/paper_six_panel_style.py"
OUTPUT = ROOT / "figures/recal2026/fastest-cartesian-polar-six-panel-20261015"
MINIMUM_PER_CELL = 8
TARGETS = (1.0e-3, 1.0e-4)
METHODS = ("cartesian", "polar")

BEST_FIELDS = [
    "geom_id", "profile", "target", "q", "rho", "a_finite", "winner",
    "cartesian_cold_ms", "polar_cold_ms", "selected_cold_ms", "vbm_ms",
    "vbm_over_selected", "magnification", "relative_reference_error",
    "estimate_accepted", "actual_pass", "false_accept", "support_proven",
    "maps", "subdivisions", "cold_warm_identical", "stop_reason",
]
EXCLUSION_FIELDS = ["geom_id", "profile", "target", "reason"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def finite(value: str | float) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def key_target(value: str | float) -> str:
    return f"{float(value):.17g}"


def parse_exception_logs() -> dict[str, str]:
    failures: dict[str, str] = {}
    for path in sorted((CAMPAIGN / "shards").glob("*.log")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            head, separator, tail = line.partition(" ")
            if separator and head.isdecimal() and tail.strip():
                failures[head] = tail.strip()
    return failures


def method_measurement(rows: list[dict[str, str]]) -> tuple[float, dict[str, str]] | None:
    rows = sorted(rows, key=lambda row: int(row["repeat"]))
    if len(rows) != 3 or [int(row["repeat"]) for row in rows] != [0, 1, 2]:
        return None
    if any(row["warm_met"] == "exception" for row in rows):
        return None
    times = [float(row["cold_ms"]) for row in rows]
    magnifications = [float(row["cold_magnification"]) for row in rows]
    if not all(math.isfinite(value) and value > 0.0 for value in times):
        return None
    if not all(math.isfinite(value) for value in magnifications):
        return None
    # The runner writes the final deterministic cold result on all three rows.
    if len(set(magnifications)) != 1:
        raise ValueError(f"cold magnification changed across repeats for {rows[0]['sample_id']}")
    for field in ("cold_met", "cold_maps", "cold_subdivisions", "stop_reason"):
        if len({row[field] for row in rows}) != 1:
            raise ValueError(f"cold result field {field} changed across repeats for {rows[0]['sample_id']}")
    return statistics.median(times), rows[-1]


def main() -> None:
    global CAMPAIGN, VBM_JOINED, OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=CAMPAIGN)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--vbm-joined", type=Path, default=VBM_JOINED)
    args = parser.parse_args()
    CAMPAIGN, VBM_JOINED, OUTPUT = args.campaign.resolve(), args.vbm_joined.resolve(), args.output.resolve()
    output = OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    os_cache = output / "matplotlib-cache"
    os_cache.mkdir(exist_ok=True)
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(os_cache))
    sys.path.insert(0, str(STYLE.parent))
    import numpy as np
    import paper_six_panel_style as style

    inputs = read_tsv(INPUTS)
    vbm_rows = read_tsv(VBM_JOINED)
    raw_rows = read_tsv(CAMPAIGN / "raw.tsv")
    if len(inputs) != 7216:
        raise RuntimeError(f"expected 7,216 input geometries, found {len(inputs)}")
    if len(vbm_rows) != 2 * len(inputs):
        raise RuntimeError("the matched VBM six-panel data do not cover both targets")
    if len(raw_rows) != len(inputs) * len(METHODS) * len(TARGETS) * 3:
        raise RuntimeError("the Cartesian/Polar raw result row count is incomplete")

    input_by_id = {row["geom_id"]: row for row in inputs}
    adjudicated_reference = {row["geom_id"]: float(row["reference"]) for row in inputs}
    for row in read_tsv(ADJUDICATIONS):
        adjudicated_reference[row["geom_id"]] = float(row["adjudicated_reference"])

    vbm_by_key = {}
    for row in vbm_rows:
        pair = (row["geom_id"], key_target(row["target"]))
        if pair in vbm_by_key:
            raise RuntimeError(f"duplicate matched VBM row: {pair}")
        vbm_by_key[pair] = row
        expected_reference = adjudicated_reference[row["geom_id"]]
        if not math.isclose(float(row["a_finite"]), expected_reference, rel_tol=2e-15, abs_tol=0.0):
            raise RuntimeError(f"reference mismatch after adjudication for {pair}")
    if len(vbm_by_key) != 2 * len(inputs):
        raise RuntimeError("duplicate or missing VBM geometry/target keys")

    raw_by_key: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in raw_rows:
        raw_by_key[(row["sample_id"], key_target(row["target"]), row["method"])].append(row)
    exception_logs = parse_exception_logs()
    exception_ids = {
        sample_id
        for (sample_id, _target, _method), rows in raw_by_key.items()
        if any(row["warm_met"] == "exception" for row in rows)
    }

    best_rows: list[dict] = []
    exclusions: list[dict] = []
    groups: dict[tuple[str, float], list[dict]] = defaultdict(list)
    excluded_by_group: dict[tuple[str, float], int] = defaultdict(int)
    method_wins: dict[tuple[str, float], dict[str, int]] = defaultdict(lambda: {m: 0 for m in METHODS})

    for geom_id, input_row in input_by_id.items():
        for target in TARGETS:
            target_key = key_target(target)
            options = []
            for method in METHODS:
                measured = method_measurement(raw_by_key[(geom_id, target_key, method)])
                if measured is not None:
                    median_ms, result = measured
                    options.append((median_ms, method, result))

            if not options:
                reason = exception_logs.get(geom_id, "no finite result from either method")
                exclusions.append({
                    "geom_id": geom_id,
                    "profile": input_row["profile"],
                    "target": target,
                    "reason": reason,
                })
                excluded_by_group[(input_row["profile"], target)] += 1
                continue

            selected_ms, winner, selected_result = min(options, key=lambda item: (item[0], item[1]))
            measurements = {method: next((item[0] for item in options if item[1] == method), math.nan)
                            for method in METHODS}
            base = vbm_by_key[(geom_id, target_key)]
            reference = adjudicated_reference[geom_id]
            magnification = float(selected_result["cold_magnification"])
            error = abs(magnification / reference - 1.0)
            target_float = float(target)
            vbm_ms = float(base["vbm_ms"])
            ratio = vbm_ms / selected_ms
            accepted = selected_result["cold_met"] == "1"
            actual_pass = error <= target_float
            group_key = (input_row["profile"], target_float)
            method_wins[group_key][winner] += 1
            record = {
                "geom_id": geom_id,
                "profile": input_row["profile"],
                "target": target_float,
                "q": float(input_row["q"]),
                "rho": float(input_row["rho"]),
                "a_finite": reference,
                "winner": winner,
                "cartesian_cold_ms": measurements["cartesian"],
                "polar_cold_ms": measurements["polar"],
                "selected_cold_ms": selected_ms,
                "vbm_ms": vbm_ms,
                "vbm_over_selected": ratio,
                "magnification": magnification,
                "relative_reference_error": error,
                "estimate_accepted": accepted,
                "actual_pass": actual_pass,
                "false_accept": accepted and not actual_pass,
                "support_proven": selected_result["support_proven"] == "1",
                "maps": int(selected_result["cold_maps"]),
                "subdivisions": int(selected_result["cold_subdivisions"]),
                "cold_warm_identical": selected_result["cold_warm_identical"] == "1",
                "stop_reason": selected_result["stop_reason"],
                "error": error,
                "ratio": ratio,
                "accepted": accepted,
            }
            best_rows.append(record)
            groups[group_key].append(record)

    if len(best_rows) + len(exclusions) != 2 * len(inputs):
        raise RuntimeError("winner plus exclusion count differs from campaign size")
    excluded_ids = {row["geom_id"] for row in exclusions}
    write_tsv(output / "best_of.tsv", best_rows, BEST_FIELDS)
    write_tsv(output / "exclusions.tsv", exclusions, EXCLUSION_FIELDS)

    summaries = []
    for profile in ("uniform", "linear"):
        for target in TARGETS:
            key = (profile, target)
            records = groups[key]
            ratios = [record["ratio"] for record in records]
            errors = [record["error"] for record in records]
            selected_ms = [record["selected_cold_ms"] for record in records]
            summary = {
                "profile": profile,
                "target": target,
                "valid_conditions": len(records),
                "excluded_conditions": excluded_by_group[key],
                "cartesian_wins": method_wins[key]["cartesian"],
                "polar_wins": method_wins[key]["polar"],
                "faster_than_vbm": sum(value > 1.0 for value in ratios),
                "median_vbm_over_selected": statistics.median(ratios),
                "median_selected_ms": statistics.median(selected_ms),
                "estimate_accepted": sum(record["accepted"] for record in records),
                "actual_error_pass": sum(record["actual_pass"] for record in records),
                "false_accepts": sum(record["false_accept"] for record in records),
                "p95_actual_relative_error": percentile(errors, 0.95),
                "max_actual_relative_error": max(errors),
                "cold_warm_identical": sum(record["cold_warm_identical"] for record in records),
            }
            summaries.append(summary)

            if profile == "uniform":
                speed_bounds = (0.01, 0.1, 1, 2, 3, 4, 6, 8, 16)
            else:
                speed_bounds = (0.03, 0.1, 1, 2, 4, 6, 8, 12, 16)
            style.SPEED_BOUNDARIES = np.asarray(speed_bounds, dtype=float)
            style.SPEED_LABELS = tuple(
                [f"$<{speed_bounds[0]:g}\\times$"]
                + [f"${value:g}\\times$" for value in speed_bounds[:-1]]
                + [f"$\\geq{speed_bounds[-1]:g}\\times$"]
            )
            error_bounds = (
                (1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3)
                if target == 1e-4
                else (1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2)
            )
            style.ERROR_BOUNDARIES = np.asarray(error_bounds, dtype=float)
            style.ERROR_LABELS = tuple(
                [f"$\\leq{error_bounds[0]:g}$"]
                + [f"${value:g}$" for value in error_bounds[1:]]
                + [f"$\\geq{error_bounds[-1]:g}$"]
            )
            tag = f"{target:.0e}".replace("-0", "-")
            output_stem = output / f"{profile}_{tag}"
            style.make_figure(
                records,
                [str(CAMPAIGN / "raw.tsv"), str(VBM_JOINED), str(ADJUDICATIONS)],
                output_stem,
                profile,
                target,
                MINIMUM_PER_CELL,
                speed_ratio_label=r"median $R=t_{\rm VBM}/t_{\rm best}$",
                error_label="cell p95 relative difference from reference",
            )
            group = summary
            output_stem.with_suffix(".md").write_text(
                f"# Faster-of-Cartesian/Polar six-panel diagnostic\n\n"
                f"Profile: `{profile}`; target: `{target:g}`.\n\n"
                f"Selected method per geometry from the lower median of three seed-inclusive cold timings. "
                f"Valid conditions: {group['valid_conditions']}; excluded: {group['excluded_conditions']}. "
                f"Cartesian wins: {group['cartesian_wins']}; Polar wins: {group['polar_wins']}.\n\n"
                f"The upper panels show cell-median `t_VBM / t_best`; the lower panels show cell p95 "
                f"relative difference to the adjudicated reference. This is an empirical diagnostic.\n"
            )

    summary_doc = {
        "selection": "per geometry and target, choose the lower median of three seed-inclusive cold timings; ties choose Cartesian",
        "speed_ratio": "retimed matched VBM ms / selected Cartesian-or-Polar median cold ms; >1 favors selected inverse-ray method",
        "accuracy": "relative difference to saved RelTol=1e-6 reference with three adjudications; empirical, not a formal bound",
        "reference_certified": False,
        "production_selector": False,
        "input_geometries": len(inputs),
        "excluded_geometry_count": len(excluded_ids),
        "any_method_exception_geometry_count": len(exception_ids),
        "valid_conditions": len(best_rows),
        "excluded_conditions": len(exclusions),
        "groups": summaries,
    }
    (output / "summary.json").write_text(json.dumps(summary_doc, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Measured faster-of-Cartesian/Polar six-panel comparison",
        "",
        "Four six-panel figures cover uniform and linear limb darkening at relative targets 1e-3 and 1e-4. "
        "For each geometry and target, the winner is the method with the lower median of three `cold_ms` "
        "measurements. `cold_ms` includes the indexed seed builder and adaptive integration; caustic geometry "
        "and center-image roots were prepared outside the timed region.",
        "",
        "The winner is selected from measured timings only, without using the reference error. This is a "
        "retrospective per-case timing oracle, not a deployable run-time selector or a population-wide speed guarantee. "
        "The speed ratio is the freshly matched direct VBM time divided by the selected method's seed-inclusive "
        "median time; values above 1 favor the selected inverse-ray method. Lower panels show empirical error "
        "against the saved RelTol=1e-6 reference with the three adjudications applied.",
        "",
        f"The source corpus has {len(inputs)} geometries and {2 * len(inputs)} target conditions. "
        f"{len(exception_ids)} geometries had an exception in at least one method. "
        f"{len(excluded_ids)} geometries ({len(exclusions)} target conditions) had no usable result from either method; "
        "these are listed in `exclusions.tsv`. "
        "All remaining finite winner rows are retained, including estimator rejections and actual-error failures.",
        "",
        "| profile | target | valid / total | excluded | Cartesian wins | Polar wins | faster than VBM | median VBM / best | median best (ms) | reference pass | accepted | false accepts | p95 error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in summaries:
        total = group["valid_conditions"] + group["excluded_conditions"]
        lines.append(
            f"| {group['profile']} | {group['target']:g} | {group['valid_conditions']}/{total} | "
            f"{group['excluded_conditions']} | {group['cartesian_wins']} | {group['polar_wins']} | "
            f"{group['faster_than_vbm']}/{group['valid_conditions']} | "
            f"{group['median_vbm_over_selected']:.3f}x | {group['median_selected_ms']:.4f} | "
            f"{group['actual_error_pass']}/{group['valid_conditions']} | {group['estimate_accepted']}/"
            f"{group['valid_conditions']} | {group['false_accepts']} | "
            f"{group['p95_actual_relative_error']:.4g} |"
        )
    lines += [
        "",
        "The q-rho coverage cohort was designed for parameter-space coverage, not as an iid speed sample. "
        "Per-cell maps mask cells with fewer than eight finite conditions. The data, exclusions, summary, "
        "and provenance are saved alongside the figures. Reproduce with:",
        "",
        f"`python {Path(__file__).relative_to(ROOT)} --campaign {CAMPAIGN} --output {OUTPUT} --vbm-joined {VBM_JOINED}`",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    manifest = json.loads((CAMPAIGN / "manifest.json").read_text())
    provenance = {
        "git_head": __import__("subprocess").check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "python": platform.python_version(),
        "host": platform.node(),
        "campaign_manifest": str((CAMPAIGN / "manifest.json").relative_to(ROOT)),
        "campaign_manifest_sha256": sha256(CAMPAIGN / "manifest.json"),
        "campaign_raw_sha256": sha256(CAMPAIGN / "raw.tsv"),
        "inputs_sha256": sha256(INPUTS),
        "matched_vbm_joined_sha256": sha256(VBM_JOINED),
        "reference_adjudications_sha256": sha256(ADJUDICATIONS),
        "plot_style_sha256": sha256(STYLE),
        "selection_script_sha256": sha256(Path(__file__)),
        "timing_definition": "median of three per-input cold calls; indexed seed cache cleared before each cold call; caustic geometry warmed",
        "reference_certified": False,
        "selection_is_production_selector": False,
    }
    artifact_paths = [
        output / "best_of.tsv", output / "exclusions.tsv", output / "summary.json",
        output / "REPORT.md",
        *sorted(output.glob("uniform_*.pdf")), *sorted(output.glob("uniform_*.png")),
        *sorted(output.glob("linear_*.pdf")), *sorted(output.glob("linear_*.png")),
        *sorted(output.glob("uniform_*.md")), *sorted(output.glob("linear_*.md")),
    ]
    provenance["artifact_sha256"] = {
        str(path.relative_to(ROOT)): sha256(path) for path in artifact_paths
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary_doc["groups"], indent=2), flush=True)
    print(f"figures: {output.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
