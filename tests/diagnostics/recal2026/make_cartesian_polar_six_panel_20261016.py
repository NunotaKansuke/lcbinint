#!/usr/bin/env python3
"""Render separate, matched six-panel maps for current adaptive Cartesian and Polar."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
INPUTS = ROOT / "figures/recal2026/adaptive-vbm-six-panel-20261014/inputs.tsv"
ADJUDICATIONS = ROOT / "figures/recal2026/polar-relative-matched-six-panel-20261015/reference-adjudications.tsv"
VBM_RESULTS = ROOT / "tests/diagnostics/results/recal2026/polar-relative-matched-six-panel-20261015/vbm.tsv"
VBM_MANIFEST = ROOT / "tests/diagnostics/results/recal2026/polar-relative-matched-six-panel-20261015/manifest.json"
STYLE = ROOT / "tests/diagnostics/recal2026/paper_six_panel_style.py"
METHODS = ("cartesian", "polar")
PROFILES = ("uniform", "linear")
TARGETS = (1.0e-3, 1.0e-4)
REPEATS = (0, 1, 2)
MINIMUM_PER_CELL = 8


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict], fields: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def target_key(value: str | float) -> str:
    return f"{float(value):.17g}"


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def percentile(values: list[float], p: float) -> float | None:
    values = sorted(value for value in values if math.isfinite(value))
    if not values:
        return None
    position = (len(values) - 1) * p
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] * (hi - position) + values[hi] * (position - lo) if lo != hi else values[lo]


def summarize(rows: list[dict]) -> list[dict]:
    result = []
    for method in METHODS:
        for profile in PROFILES:
            for target in TARGETS:
                selected = [r for r in rows if r["method"] == method and r["profile"] == profile and r["target"] == target]
                valid = [r for r in selected if finite(r["vbm_over_method"]) and float(r["vbm_over_method"]) > 0.0]
                ratios = [float(r["vbm_over_method"]) for r in valid]
                passing = [r for r in valid if bool(r["actual_pass"])]
                passing_ratios = [float(r["vbm_over_method"]) for r in passing]
                errors = [float(r["relative_reference_error"]) for r in selected if finite(r["relative_reference_error"])]
                result.append({
                    "method": method,
                    "profile": profile,
                    "target": target,
                    "conditions": len(selected),
                    "finite_timing_and_value": len(valid),
                    "exceptions_or_invalid": len(selected) - len(valid),
                    "estimate_accepted": sum(bool(r["estimate_accepted"]) for r in selected),
                    "reference_pass": sum(bool(r["actual_pass"]) for r in selected),
                    "false_accepts": sum(bool(r["false_accept"]) for r in selected),
                    "faster_than_vbm": sum(value > 1.0 for value in ratios),
                    "median_vbm_over_method": statistics.median(ratios) if ratios else None,
                    "reference_pass_conditions_with_timing": len(passing),
                    "median_vbm_over_method_reference_pass": statistics.median(passing_ratios) if passing_ratios else None,
                    "median_method_ms": statistics.median(float(r["method_ms"]) for r in valid) if valid else None,
                    "median_vbm_ms": statistics.median(float(r["vbm_ms"]) for r in valid) if valid else None,
                    "p95_reference_error": percentile(errors, 0.95),
                    "max_reference_error": max(errors) if errors else None,
                    "median_error_over_target": statistics.median(
                        float(r["relative_reference_error"]) / target for r in selected
                        if finite(r["relative_reference_error"])
                    ) if errors else None,
                })
    return result


def render(campaign: Path, output: Path) -> None:
    if output.exists() and (output / "provenance.json").exists():
        previous = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
        if previous.get("campaign_manifest_sha256") != sha256(campaign / "manifest.json"):
            raise RuntimeError(f"existing figure directory belongs to a different campaign: {output}")
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "matplotlib-cache"
    cache.mkdir(exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache)
    sys.path.insert(0, str(STYLE.parent))
    import matplotlib as mpl
    import numpy as np
    import paper_six_panel_style as style

    inputs = read_tsv(INPUTS)
    if len(inputs) != 7216 or len({r["geom_id"] for r in inputs}) != 7216:
        raise RuntimeError("expected 7,216 unique geometries")
    campaign_manifest = json.loads((campaign / "manifest.json").read_text(encoding="utf-8"))
    if campaign_manifest.get("inputs_sha256") != sha256(INPUTS):
        raise RuntimeError("current method campaign used a different input corpus")
    vbm_manifest = json.loads(VBM_MANIFEST.read_text(encoding="utf-8"))
    if vbm_manifest.get("input_sha256") != sha256(INPUTS):
        raise RuntimeError("saved VBM timings used a different input corpus")
    refs = {r["geom_id"]: float(r["reference"]) for r in inputs}
    for row in read_tsv(ADJUDICATIONS):
        refs[row["geom_id"]] = float(row["adjudicated_reference"])
    input_by_id = {r["geom_id"]: r for r in inputs}
    vbm = read_tsv(VBM_RESULTS)
    expected = {(row["geom_id"], target_key(target)) for row in inputs for target in TARGETS}
    vbm_by_key = {}
    for row in vbm:
        key = (row["sample_id"], target_key(row["target"]))
        if key in vbm_by_key:
            raise RuntimeError(f"duplicate VBM result: {key}")
        vbm_by_key[key] = row
    if set(vbm_by_key) != expected:
        raise RuntimeError("saved VBM timing cohort does not match all geometries and targets")

    raw = read_tsv(campaign / "raw.tsv")
    expected_raw_count = len(inputs) * len(METHODS) * len(TARGETS) * len(REPEATS)
    if len(raw) != expected_raw_count:
        raise RuntimeError(f"expected {expected_raw_count} raw method rows, got {len(raw)}")
    raw_by_key: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in raw:
        raw_by_key[(row["sample_id"], target_key(row["target"]), row["method"])].append(row)

    records = []
    for geom_id, inp in input_by_id.items():
        reference = refs[geom_id]
        for target in TARGETS:
            target_s = target_key(target)
            vbm_row = vbm_by_key[(geom_id, target_s)]
            vbm_ms = float(vbm_row["ms"])
            for method in METHODS:
                reps = sorted(raw_by_key[(geom_id, target_s, method)], key=lambda r: int(r["repeat"]))
                if [int(r["repeat"]) for r in reps] != list(REPEATS):
                    raise RuntimeError(f"incomplete repeats for {geom_id}, {method}, {target:g}")
                for field in ("cold_magnification", "cold_met", "cold_maps", "cold_subdivisions", "stop_reason"):
                    if len({r[field] for r in reps}) != 1:
                        raise RuntimeError(f"nondeterministic {field}: {geom_id}, {method}, {target:g}")
                times = [float(r["cold_ms"]) for r in reps if finite(r["cold_ms"]) and float(r["cold_ms"]) > 0.0]
                method_ms = statistics.median(times) if len(times) == len(REPEATS) else math.nan
                magnification = float(reps[0]["cold_magnification"])
                error = abs(magnification / reference - 1.0) if finite(magnification) and reference > 0.0 else math.nan
                ratio = vbm_ms / method_ms if math.isfinite(method_ms) and vbm_ms > 0.0 else math.nan
                accepted = reps[0]["cold_met"] == "1"
                records.append({
                    "geom_id": int(geom_id),
                    "case_id": int(inp["case_id"]),
                    "d_bin_index": int(inp["d_bin_index"]),
                    "epoch_index": int(inp["epoch_index"]),
                    "profile": inp["profile"],
                    "method": method,
                    "target": target,
                    "q": float(inp["q"]),
                    "rho": float(inp["rho"]),
                    "a_finite": reference,
                    "magnification": magnification,
                    "relative_reference_error": error,
                    "estimate_indicator": float(reps[0]["cold_indicator"]),
                    "estimate_accepted": accepted,
                    "actual_pass": math.isfinite(error) and error <= target,
                    "false_accept": accepted and math.isfinite(error) and error > target,
                    "status": reps[0]["stop_reason"],
                    "method_ms": method_ms,
                    "method_repeat_ms": ",".join(r["cold_ms"] for r in reps),
                    "vbm_magnification": float(vbm_row["magnification"]),
                    "vbm_reference_error": abs(float(vbm_row["magnification"]) / reference - 1.0),
                    "vbm_ms": vbm_ms,
                    "vbm_over_method": ratio,
                    "maps": int(reps[0]["cold_maps"]),
                    "subdivisions": int(reps[0]["cold_subdivisions"]),
                    "cold_warm_identical": all(r["cold_warm_identical"] == "1" for r in reps),
                })

    if len(records) != len(inputs) * len(METHODS) * len(TARGETS):
        raise RuntimeError("joined result count is incomplete")
    fields = list(records[0])
    write_tsv(output / "joined.tsv", records, fields)

    # Keep a shared reciprocal scale for all eight figures and white at R=1.
    exponents = tuple(range(-6, 7))
    speed_bounds = tuple(2.0**exponent for exponent in exponents)
    style.SPEED_BOUNDARIES = np.asarray(speed_bounds, dtype=float)
    cmap = mpl.colormaps["RdBu_r"]
    colours = [mpl.colors.to_hex(cmap(p)) for p in np.linspace(0.035, 0.965, len(speed_bounds) + 1)]
    labels = [r"$<1/64\times$"]
    for exponent in exponents[:-1]:
        labels.append(rf"$1/{2 ** (-exponent)}\times$" if exponent < 0 else rf"${2 ** exponent}\times$")
    labels.append(r"$\geq64\times$")
    colours[labels.index(r"$1\times$")] = "#ffffff"
    style.SPEED_COLOURS = tuple(colours)
    style.SPEED_LABELS = tuple(labels)

    for profile in PROFILES:
        for target in TARGETS:
            bounds = (
                (1.0e-5, 2.0e-5, 5.0e-5, 1.0e-4, 2.0e-4, 5.0e-4, 1.0e-3)
                if target == 1.0e-4
                else (1.0e-4, 2.0e-4, 5.0e-4, 1.0e-3, 2.0e-3, 5.0e-3, 1.0e-2)
            )
            style.ERROR_BOUNDARIES = np.asarray(bounds, dtype=float)
            style.ERROR_LABELS = tuple(
                [rf"$\leq{bounds[0]:g}$"] + [rf"${value:g}$" for value in bounds[1:]] + [rf"$\geq{bounds[-1]:g}$"]
            )
            for method in METHODS:
                selected = [
                    {**row, "ratio": float(row["vbm_over_method"]), "error": float(row["relative_reference_error"])}
                    for row in records
                    if row["method"] == method and row["profile"] == profile and row["target"] == target
                    and finite(row["vbm_over_method"]) and float(row["vbm_over_method"]) > 0.0
                ]
                stem = output / f"{method}_{profile}_{target:.0e}"
                style.make_figure(
                    selected,
                    [str(campaign / "raw.tsv"), str(VBM_RESULTS), str(ADJUDICATIONS)],
                    stem,
                    profile,
                    target,
                    MINIMUM_PER_CELL,
                    speed_ratio_label=rf"median $R=t_{{\rm VBM}}/t_{{\rm {method}}}$ ($R>1$: {method} faster)",
                    error_label="cell p95 relative difference from adjudicated reference",
                )

    summaries = summarize(records)
    for group in summaries:
        method = group["method"]
        profile = group["profile"]
        target = group["target"]
        stem = output / f"{method}_{profile}_{target:.0e}"
        ratio = group["median_vbm_over_method"]
        passing_ratio = group["median_vbm_over_method_reference_pass"]
        lines = [
            f"# Adaptive {method.capitalize()} six-panel comparison",
            "",
            f"Profile: `{profile}`; target: `{target:g}`; conditions: `{group['conditions']}`.",
            "",
            "The upper panels show cell median `t_VBM/t_method`; values above one favor this adaptive flood-fill route. The lower panels show cell p95 absolute relative difference from the adjudicated saved RelTol=1e-6 reference, not difference from VBM.",
            "",
            f"Finite timing/value rows: `{group['finite_timing_and_value']}`; estimate accepted: `{group['estimate_accepted']}`; actual reference pass: `{group['reference_pass']}/{group['conditions']}`; false accepts: `{group['false_accepts']}`.",
            f"Median VBM/method ratio over finite rows: `{ratio:.6g}`; among actual reference-pass rows: `{passing_ratio:.6g}`.",
            f"Reference-error p95: `{group['p95_reference_error']:.6g}`; maximum: `{group['max_reference_error']:.6g}`.",
            "",
            "Agreement is empirical, not a formal error certificate. Per-condition values and rejections are in `joined.tsv`.",
            "",
        ]
        stem.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps({
        "rows": len(records),
        "geometries": len(inputs),
        "methods": list(METHODS),
        "targets": list(TARGETS),
        "reference_certified": False,
        "complete_error_model": False,
        "speed_ratio": "median of three cold, seed-cache-miss adaptive calls; matched saved VBM median direct-call time / method time; R>1 favors method",
        "accuracy": "empirical relative difference from saved RelTol=1e-6 reference after three adjudications; not a formal error bound",
        "groups": summaries,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest = campaign_manifest
    lines = [
        "# Current Cartesian and Polar six-panel benchmark",
        "",
        "Separate six-panel maps compare the adaptive flood-fill Cartesian and Polar routes with the same matched VBM timings. The cohort contains 7,216 geometries (3,608 each uniform and linear LD), both requested relative tolerances, and no per-case method selection.",
        "",
        "Each method time is the median of three cold calls after clearing the source-seed cache. The indexed shared seed builder and adaptive integration are timed together; center-image roots and caustic-geometry construction are outside the timer, with caustic geometry prepared once per input. VBM uses matched saved direct-binding timings, one warm call plus three measured calls per input/target. Thus the ratio includes a small difference in timing setup, but uses the same geometries, LD profiles, and tolerances.",
        "",
        "The runner uses Cartesian initial bins 16×16 and Polar initial radial bins 2 with angular ratio 256; both use the indexed `cached_binary_image_seeds` provider and a 1,000,000-map budget. Directional adaptation, local row interpolation, local boundary strips, local vertical caps, and defect correction are explicitly enabled for both routes by the matched diagnostic runner.",
        "",
        "The top row of each six-panel figure shows the cell median `R=t_VBM/t_method`; values above one favor that method. The table gives the median both over every finite timing/value and over the subset whose actual difference is within target. The bottom row shows cell p95 absolute relative difference from the adjudicated saved RelTol=1e-6 reference. This agreement measure is empirical, not a formal error certificate. The reference corrections are applied before calculating errors. Cells with fewer than eight finite timings are masked.",
        "",
        "All eight speed maps share reciprocal factor-of-two bins, and the 1× category is white. The q-rho cohort is a parameter-space coverage grid rather than an iid speed sample.",
        "",
        "| method | profile | target | conditions | finite | estimate accepted | reference pass | false accepts | faster than VBM | median VBM/method | median ratio on passing rows | median method ms | p95 error | max error |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in summaries:
        ratio = f"{group['median_vbm_over_method']:.3f}x" if group["median_vbm_over_method"] is not None else "NA"
        pass_ratio = (
            f"{group['median_vbm_over_method_reference_pass']:.3f}x"
            if group["median_vbm_over_method_reference_pass"] is not None else "NA"
        )
        med_ms = f"{group['median_method_ms']:.5f}" if group["median_method_ms"] is not None else "NA"
        p95 = f"{group['p95_reference_error']:.4g}" if group["p95_reference_error"] is not None else "NA"
        max_err = f"{group['max_reference_error']:.4g}" if group["max_reference_error"] is not None else "NA"
        lines.append(
            f"| {group['method']} | {group['profile']} | {group['target']:g} | {group['conditions']} | "
            f"{group['finite_timing_and_value']} | {group['estimate_accepted']} | {group['reference_pass']} | "
            f"{group['false_accepts']} | {group['faster_than_vbm']}/{group['finite_timing_and_value']} | "
            f"{ratio} | {pass_ratio} | {med_ms} | {p95} | {max_err} |"
        )
    lines += [
        "",
        "Accuracy pass counts are against the saved adjudicated reference and do not certify all inputs. All measured rows, including rejections, actual-error misses, and exceptions, are in `joined.tsv`.",
        "",
        f"Benchmark manifest: `{campaign / 'manifest.json'}` (HEAD `{manifest['git_head'][:12]}` plus the exact dirty-source/library hashes in that manifest).",
        f"Reproduction: `/home/nunota/.miniconda3/envs/myenv/bin/python tests/diagnostics/recal2026/seed_inclusive_cart_polar_benchmark.py --output {campaign.relative_to(ROOT)}` then `/home/nunota/.miniconda3/envs/myenv/bin/python {Path(__file__).relative_to(ROOT)} --campaign {campaign.relative_to(ROOT)} --output {output.relative_to(ROOT)}`.",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    files = [p for p in output.iterdir() if p.is_file() and p.name != "provenance.json"]
    (output / "provenance.json").write_text(json.dumps({
        "git_head": manifest["git_head"],
        "campaign_manifest_sha256": sha256(campaign / "manifest.json"),
        "campaign_raw_sha256": sha256(campaign / "raw.tsv"),
        "vbm_manifest_sha256": sha256(VBM_MANIFEST),
        "vbm_results_sha256": sha256(VBM_RESULTS),
        "inputs_sha256": sha256(INPUTS),
        "adjudications_sha256": sha256(ADJUDICATIONS),
        "plot_style_sha256": sha256(STYLE),
        "renderer_sha256": sha256(Path(__file__)),
        "artifact_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sorted(files)},
        "certified": False,
        "complete_error_model": False,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2), flush=True)
    print(f"figures: {output.relative_to(ROOT)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.campaign.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
