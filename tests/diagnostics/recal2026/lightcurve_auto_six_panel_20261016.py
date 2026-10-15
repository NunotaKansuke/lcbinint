#!/usr/bin/env python3
"""Benchmark the default LightCurve Cartesian/Polar auto route on the six-panel corpus.

Each (case, impact-parameter bin, limb-darkening profile) is evaluated as a
small vector of epochs through the public ``LightCurve.info`` API. One warm
batch and three measured batches are used; the median batch time is divided by
the number of epochs when forming the per-epoch speed ratio.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
INPUTS = ROOT / "figures/recal2026/adaptive-vbm-six-panel-20261014/inputs.tsv"
VBM_RESULTS = ROOT / "tests/diagnostics/results/recal2026/polar-relative-matched-six-panel-20261015/vbm.tsv"
VBM_MEASUREMENT = ROOT / "tests/diagnostics/results/recal2026/polar-relative-matched-six-panel-20261015/measurement.json"
ADJUDICATIONS = ROOT / "figures/recal2026/polar-relative-matched-six-panel-20261015/reference-adjudications.tsv"
NATIVE_BUILD = ROOT / "build-adaptive-production"
SOURCE = Path(__file__).resolve()
RUN = ROOT / "tests/diagnostics/results/recal2026/lightcurve-auto-six-panel-20261016"
ARTIFACT = ROOT / "figures/recal2026/lightcurve-auto-vbm-six-panel-20261016"
TARGETS = (1.0e-3, 1.0e-4)
REPEATS = 3
MINIMUM_PER_CELL = 8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0]) if rows else []
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)
    temporary.replace(path)


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def key_target(value: str | float) -> str:
    return f"{float(value):.17g}"


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def load_cohort():
    inputs = read_tsv(INPUTS)
    if len(inputs) != 7216 or len({row["geom_id"] for row in inputs}) != 7216:
        raise RuntimeError(f"expected 7,216 unique geometries, found {len(inputs)}")
    if sum(row["profile"] == "uniform" for row in inputs) != 3608:
        raise RuntimeError("unexpected uniform cohort size")
    if sum(row["profile"] == "linear" for row in inputs) != 3608:
        raise RuntimeError("unexpected linear-LD cohort size")

    references = {row["geom_id"]: float(row["reference"]) for row in inputs}
    for row in read_tsv(ADJUDICATIONS):
        references[row["geom_id"]] = float(row["adjudicated_reference"])

    vbm = read_tsv(VBM_RESULTS)
    expected = {(row["geom_id"], key_target(target)) for row in inputs for target in TARGETS}
    vbm_by_key = {}
    for row in vbm:
        key = (row["sample_id"], key_target(row["target"]))
        if key in vbm_by_key:
            raise RuntimeError(f"duplicate VBM result {key}")
        vbm_by_key[key] = row
    if set(vbm_by_key) != expected:
        raise RuntimeError("saved VBM timings do not cover the six-panel cohort")

    # The existing matched join contains the adjudicated references used by
    # the original six-panel figures. Confirm those values before reuse.
    old_join = read_tsv(ROOT / "figures/recal2026/polar-relative-matched-six-panel-20261015/joined.tsv")
    for row in old_join:
        if not math.isclose(float(row["a_finite"]), references[row["geom_id"]], rel_tol=2e-15, abs_tol=0.0):
            raise RuntimeError(f"reference disagreement after adjudication: {row['geom_id']}")

    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in inputs:
        groups[(row["case_id"], row["d_bin_index"], row["profile"])].append(row)
    ordered = []
    for key in sorted(groups, key=lambda item: (int(item[0]), int(item[1]), item[2])):
        rows = sorted(groups[key], key=lambda row: (int(row["epoch_index"]), int(row["geom_id"])))
        # Parameters other than time must remain fixed within one vector call.
        for field in ("s", "q", "rho", "y", "c"):
            if len({row[field] for row in rows}) != 1:
                raise RuntimeError(f"{field} changes within LightCurve batch {key}")
        ordered.append((key, rows))
    if sum(len(rows) for _, rows in ordered) != len(inputs):
        raise RuntimeError("grouping changed the input coverage")
    return inputs, references, vbm_by_key, ordered


def setup_runtime():
    # The installed editable finder can otherwise shadow the active build.
    for name in list(os.environ):
        if name.startswith("LCBININT_"):
            os.environ.pop(name)
    allowed = os.sched_getaffinity(0)
    cpu = 0 if 0 in allowed else min(allowed)
    os.sched_setaffinity(0, {cpu})
    os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    sys.meta_path = [
        finder for finder in sys.meta_path
        if "editable" not in getattr(type(finder), "__module__", "")
    ]
    sys.path.insert(0, str(NATIVE_BUILD))
    import numpy as np
    import lcbinint as lc

    native_path = Path(lc._native.__file__).resolve()
    if NATIVE_BUILD.resolve() not in native_path.parents:
        raise RuntimeError(f"loaded wrong native module: {native_path}")
    defaults = lc.Options()
    expected_defaults = {
        "mode": 4,
        "inverse_ray_grid": "auto",
        "nbin": "auto",
        "max_source_bins": 512,
        "point_source_threshold": 20.0,
        "hexadecapole_threshold": 3.0,
        "adaptive_hex_threshold": 0.001,
    }
    for key, expected in expected_defaults.items():
        if getattr(defaults, key) != expected:
            raise RuntimeError(f"unexpected LightCurve default {key}={getattr(defaults, key)!r}")
    return lc, np, native_path, cpu, expected_defaults


def evaluate_group(rows, target, lc, np):
    first = rows[0]
    # Keep the production automatic grid selector and ordinary point/hex
    # short-circuits. Only coordinate frame and requested relative tolerance
    # are fixed to match this saved cohort and its six-panel targets.
    options = lc.Options(
        coordinates="center_of_mass",
        finite_source_tol=0.0,
        finite_source_reltol=target,
    )
    if options.inverse_ray_grid != "auto" or options.nbin != "auto":
        raise RuntimeError("LightCurve auto routing was not selected")
    curve = lc.LightCurve(lens="binary", options=options)
    epochs = np.asarray([float(row["time"]) for row in rows], dtype=float)
    params = dict(
        t0=0.0,
        tE=1.0,
        u0=float(first["y"]),
        alpha=0.0,
        s=float(first["s"]),
        q=1.0 / float(first["q"]),
        rho=float(first["rho"]),
        limb_darkening_c=float(first["c"]),
    )

    def call():
        return curve.info(epochs, **params)

    warm = call()
    if len(warm.finite_source_magnifications) != len(rows):
        raise RuntimeError("warm LightCurve.info result length differs from input batch")
    if any(abs(float(x) - float(row["time"])) > 1.0e-12
           for x, row in zip(warm.source_x, rows)):
        raise RuntimeError("LightCurve trajectory did not reproduce the saved source x")

    timings = []
    measured = []
    for _ in range(REPEATS):
        start = time.perf_counter_ns()
        info = call()
        timings.append((time.perf_counter_ns() - start) / 1.0e6)
        measured.append(info)

    median_batch_ms = statistics.median(timings)
    first_result = measured[0]
    records = []
    for index, row in enumerate(rows):
        values = [float(info.finite_source_magnifications[index]) for info in measured]
        methods = [str(info.finite_source_method_names[index]) for info in measured]
        statuses = [str(info.statuses[index]) for info in measured]
        estimates = [float(info.finite_source_error_estimates[index]) for info in measured]
        converged = [bool(info.finite_source_converged[index]) for info in measured]
        if len(set(values)) != 1 or len(set(methods)) != 1 or len(set(statuses)) != 1:
            raise RuntimeError(f"nondeterministic LightCurve result: {row['geom_id']} target={target}")
        if len(set(converged)) != 1:
            raise RuntimeError(f"nondeterministic convergence state: {row['geom_id']} target={target}")
        reference = float(row["_reference"])
        vbm = row["_vbm"][key_target(target)]
        magnification = values[0]
        error = abs(magnification / reference - 1.0) if math.isfinite(magnification) and reference > 0.0 else math.nan
        vbm_magnification = float(vbm["magnification"])
        vbm_error = abs(vbm_magnification / reference - 1.0)
        auto_ms_per_epoch = median_batch_ms / len(rows)
        vbm_ms = float(vbm["ms"])
        records.append({
            "geom_id": int(row["geom_id"]),
            "case_id": int(row["case_id"]),
            "d_bin_index": int(row["d_bin_index"]),
            "epoch_index": int(row["epoch_index"]),
            "profile": row["profile"],
            "target": target,
            "q": float(row["q"]),
            "rho": float(row["rho"]),
            "a_finite": reference,
            "magnification": magnification,
            "relative_reference_error": error,
            "estimated_error": estimates[0],
            "estimate_converged": converged[0],
            "status": statuses[0],
            "method": methods[0],
            "auto_batch_n": len(rows),
            "auto_batch_ms": median_batch_ms,
            "auto_repeat_ms": ",".join(f"{value:.17g}" for value in timings),
            "auto_ms_per_epoch": auto_ms_per_epoch,
            "vbm_magnification": vbm_magnification,
            "vbm_reference_error": vbm_error,
            "vbm_ms": vbm_ms,
            "vbm_over_auto": vbm_ms / auto_ms_per_epoch if auto_ms_per_epoch > 0.0 else math.nan,
            "actual_pass": error <= target,
            "false_accept": converged[0] and not (error <= target),
        })
    return records


def run_campaign(limit_groups: int | None):
    RUN.mkdir(parents=True, exist_ok=True)
    batches = RUN / "batches"
    batches.mkdir(exist_ok=True)
    inputs, references, vbm_by_key, ordered = load_cohort()
    if limit_groups is not None:
        ordered = ordered[:limit_groups]
    lc, np, native_path, cpu, defaults = setup_runtime()
    manifest = {
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "input_sha256": sha256(INPUTS),
        "vbm_results_sha256": sha256(VBM_RESULTS),
        "vbm_measurement_sha256": sha256(VBM_MEASUREMENT),
        "adjudications_sha256": sha256(ADJUDICATIONS),
        "native_module": str(native_path),
        "native_sha256": sha256(native_path),
        "script_sha256": sha256(SOURCE),
        "python": platform.python_version(),
        "host": platform.node(),
        "cpu_affinity": [cpu],
        "threads": 1,
        "default_options": defaults,
        "options_overrides": {
            "coordinates": "center_of_mass",
            "finite_source_tol": 0.0,
            "finite_source_reltol": list(TARGETS),
        },
        "targets": list(TARGETS),
        "timing": "public LightCurve.info vector call; one warm batch and three timed batches; median batch wall time divided by epoch count; LightCurve object and NumPy epoch array prepared outside timer",
        "vbm_timing": "reused matched direct Python binding timings; same cohort and targets; one warm plus three timed calls per epoch; median; VBM solver setup excluded",
        "reference_certified": False,
    }
    manifest_path = RUN / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError("existing LightCurve campaign identity differs; use a new result directory")
    else:
        write_json(manifest_path, manifest)

    input_by_id = {row["geom_id"]: dict(row) for row in inputs}
    for geom_id, row in input_by_id.items():
        row["_reference"] = references[geom_id]
        row["_vbm"] = {
            key_target(target): vbm_by_key[(geom_id, key_target(target))]
            for target in TARGETS
        }

    all_records = []
    started = time.monotonic()
    completed = 0
    total = len(ordered)
    for key, raw_rows in ordered:
        batch_path = batches / f"case-{int(key[0]):04d}_d-{int(key[1]):02d}_{key[2]}.json"
        if batch_path.exists():
            group_records = json.loads(batch_path.read_text())
            expected_keys = {
                (int(row["geom_id"]), float(target))
                for row in raw_rows for target in TARGETS
            }
            seen = {(int(row["geom_id"]), float(row["target"])) for row in group_records}
            if seen != expected_keys:
                raise RuntimeError(f"incomplete cached batch {batch_path}")
        else:
            group_records = []
            rows_with_refs = [input_by_id[row["geom_id"]] for row in raw_rows]
            for target in TARGETS:
                group_records.extend(evaluate_group(rows_with_refs, target, lc, np))
            write_json(batch_path, group_records)
        all_records.extend(group_records)
        completed += 1
        if completed % 25 == 0 or completed == total:
            write_json(RUN / "status.json", {
                "state": "running", "completed_groups": completed,
                "total_groups": total, "completed_conditions": len(all_records),
                "elapsed_seconds": time.monotonic() - started,
                "updated_unix": time.time(), "cpu_affinity": [cpu], "threads": 1,
            })
            print(f"LightCurve auto: {completed}/{total} batches, {len(all_records)} conditions, "
                  f"{time.monotonic() - started:.1f}s", flush=True)

    expected_count = sum(len(rows) for _, rows in ordered) * len(TARGETS)
    if len(all_records) != expected_count:
        raise RuntimeError(f"campaign result count {len(all_records)} != {expected_count}")
    if limit_groups is None and len(all_records) != len(inputs) * len(TARGETS):
        raise RuntimeError("full campaign does not cover all 14,432 target conditions")
    write_tsv(RUN / "raw.tsv", all_records)
    write_json(RUN / "status.json", {
        "state": "measurement_complete" if limit_groups is None else "partial_smoke_complete",
        "completed_groups": completed, "total_groups": total,
        "completed_conditions": len(all_records), "elapsed_seconds": time.monotonic() - started,
        "updated_unix": time.time(), "cpu_affinity": [cpu], "threads": 1,
    })
    if limit_groups is None:
        render(inputs, references, all_records, manifest)
    return all_records


def summarize(records):
    groups = []
    for profile in ("uniform", "linear"):
        for target in TARGETS:
            selected = [row for row in records if row["profile"] == profile and row["target"] == target]
            finite_rows = [row for row in selected if finite(row["vbm_over_auto"]) and finite(row["relative_reference_error"])]
            ratios = [row["vbm_over_auto"] for row in finite_rows]
            errors = [row["relative_reference_error"] for row in finite_rows]
            methods = Counter(row["method"] for row in selected)
            accepted = sum(bool(row["estimate_converged"]) for row in selected)
            actual_pass = sum(bool(row["actual_pass"]) for row in selected)
            false_accepts = sum(bool(row["false_accept"]) for row in selected)
            failure_routes = Counter(
                row["method"] for row in selected
                if not bool(row["actual_pass"])
            )
            false_accept_routes = Counter(
                row["method"] for row in selected
                if bool(row["false_accept"])
            )
            groups.append({
                "profile": profile,
                "target": target,
                "conditions": len(selected),
                "finite_conditions": len(finite_rows),
                "accepted": accepted,
                "reference_pass": actual_pass,
                "false_accepts": false_accepts,
                "reference_failures_by_method": dict(sorted(failure_routes.items())),
                "false_accepts_by_method": dict(sorted(false_accept_routes.items())),
                "median_vbm_over_auto": statistics.median(ratios) if ratios else None,
                "faster_than_vbm": sum(ratio > 1.0 for ratio in ratios),
                "median_vbm_ms": statistics.median(row["vbm_ms"] for row in finite_rows) if finite_rows else None,
                "median_auto_ms_per_epoch": statistics.median(row["auto_ms_per_epoch"] for row in finite_rows) if finite_rows else None,
                "p95_relative_reference_error": sorted(errors)[math.ceil(0.95 * len(errors)) - 1] if errors else None,
                "max_relative_reference_error": max(errors) if errors else None,
                "methods": dict(sorted(methods.items())),
            })
    return groups


def render(inputs, references, records, manifest):
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(RUN / "matplotlib-cache"))
    sys.path.insert(0, str(ROOT / "tests/diagnostics/recal2026"))
    import matplotlib as mpl
    import numpy as np
    import paper_six_panel_style as style

    joined_path = ARTIFACT / "joined.tsv"
    joined_fields = [
        "geom_id", "case_id", "d_bin_index", "epoch_index", "profile", "target",
        "q", "rho", "a_finite", "magnification", "relative_reference_error",
        "estimated_error", "estimate_converged", "status", "method", "auto_batch_n",
        "auto_batch_ms", "auto_repeat_ms", "auto_ms_per_epoch", "vbm_magnification",
        "vbm_reference_error", "vbm_ms", "vbm_over_auto", "actual_pass", "false_accept",
    ]
    write_tsv(joined_path, records, joined_fields)

    # A shared reciprocal scale makes equal visual distances correspond to
    # equal multiplicative changes on either side of R=1. R<1 favors VBM;
    # R>1 favors the LightCurve auto path.
    speed_exponents = tuple(range(-6, 7))
    speed_bounds = tuple(2.0**exponent for exponent in speed_exponents)
    style.SPEED_BOUNDARIES = np.asarray(speed_bounds, dtype=float)
    speed_colour_count = len(speed_bounds) + 1
    speed_cmap = mpl.colormaps["RdBu_r"]
    speed_colours = [
        mpl.colors.to_hex(speed_cmap(position))
        for position in np.linspace(0.035, 0.965, speed_colour_count)
    ]
    def format_speed_threshold(exponent: int) -> str:
        if exponent < 0:
            return f"$1/{2 ** (-exponent)}\\times$"
        return f"${2 ** exponent}\\times$"

    style.SPEED_LABELS = tuple(
        [r"$<1/64\times$"]
        + [format_speed_threshold(exponent) for exponent in speed_exponents[:-1]]
        + [r"$\geq64\times$"]
    )
    speed_colours[style.SPEED_LABELS.index(r"$1\times$")] = "#ffffff"
    style.SPEED_COLOURS = tuple(speed_colours)

    for profile in ("uniform", "linear"):
        for target in TARGETS:
            error_bounds = (
                (1.0e-5, 2.0e-5, 5.0e-5, 1.0e-4, 2.0e-4, 5.0e-4, 1.0e-3)
                if target == 1.0e-4
                else (1.0e-4, 2.0e-4, 5.0e-4, 1.0e-3, 2.0e-3, 5.0e-3, 1.0e-2)
            )
            style.ERROR_BOUNDARIES = np.asarray(error_bounds, dtype=float)
            style.ERROR_LABELS = tuple(
                [f"$\\leq{error_bounds[0]:g}$"]
                + [f"${value:g}$" for value in error_bounds[1:]]
                + [f"$\\geq{error_bounds[-1]:g}$"]
            )
            selected = [
                {**row, "ratio": float(row["vbm_over_auto"]), "error": float(row["relative_reference_error"])}
                for row in records
                if row["profile"] == profile and row["target"] == target
                and finite(row["vbm_over_auto"]) and row["vbm_over_auto"] > 0.0
            ]
            tag = f"{target:.0e}".replace("-0", "-")
            style.make_figure(
                selected,
                [str(INPUTS), str(VBM_RESULTS), str(ADJUDICATIONS)],
                ARTIFACT / f"{profile}_{tag}",
                profile,
                target,
                MINIMUM_PER_CELL,
                speed_ratio_label=(
                    r"median $R=t_{\rm VBM}/t_{\rm LightCurve,auto}$ "
                    r"($R<1$: VBM faster; $R>1$: auto faster)"
                ),
                error_label="cell p95 relative difference from adjudicated reference",
            )

    summary = {
        "rows": len(records),
        "input_geometries": len(inputs),
        "certified": False,
        "complete_error_model": False,
        "reference_certified": False,
        "speed_ratio": "retimed matched direct VBM binding time / amortized public LightCurve.info batch time per epoch; values above 1 favor LightCurve auto",
        "accuracy": "absolute relative difference from saved RelTol=1e-6 reference with three adjudications; empirical, not a formal error bound",
        "groups": summarize(records),
    }
    write_json(ARTIFACT / "summary.json", summary)
    lines = [
        "# LightCurve auto vs VBM: six-panel comparison",
        "",
        "Four six-panel figures compare the actual public binary `LightCurve.info` auto path against the saved matched VBM direct-binding measurements, for uniform and linear limb darkening at relative targets 1e-3 and 1e-4.",
        "",
        "The 7,216 geometries are grouped by case, impact-parameter bin, and brightness profile. Each full group of sampled epochs is passed together to `LightCurve.info`, with one untimed warm batch followed by three measured batches. The median batch wall time is divided by the number of epochs in that batch for the per-epoch speed ratio. `LightCurve` construction and the epoch array are outside the timer. The auto route keeps its default `inverse_ray_grid=auto`, `nbin=auto`, 512-bin cap, and default point-source/hexadecapole short-circuits; only the coordinate frame and requested tolerance are fixed to match the saved cohort. The VBM measurements are the fresh matched direct `BinaryMag2`/`BinaryMagDark` timings in the Polar six-panel campaign (one warm and three timed calls per epoch). Since VBM uses a direct kernel binding while the numerator uses the public LightCurve API, this speed ratio includes different wrapper and batching boundaries and is an empirical comparison, not a perfectly symmetric kernel-only benchmark.",
        "",
        "The upper panels show the cell median `t_VBM / t_LightCurve,auto` (values above one favor the LightCurve auto path). The lower panels show the cell p95 relative difference from the saved RelTol=1e-6 reference after the documented three reference adjudications. This is an empirical agreement diagnostic, not a formal error certificate. All conditions, including non-convergence and actual-error failures, remain in `joined.tsv`.",
        "",
        "All four speed maps share the same reciprocal factor-of-two color bins, with boundaries `R = 2^k` for `k = -6,...,6`. The lower tail is `<1/64` and the upper tail is `>=64`; blue denotes `R<1` (VBM faster) and red denotes `R>1` (LightCurve auto faster).",
        "",
        "| profile | target | conditions | converged | reference pass | false accepts | faster than VBM | median VBM / auto | median auto ms/epoch | p95 error | max error | methods |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for group in summary["groups"]:
        lines.append(
            f"| {group['profile']} | {group['target']:g} | {group['conditions']} | {group['accepted']} | "
            f"{group['reference_pass']} | {group['false_accepts']} | "
            f"{group['faster_than_vbm']}/{group['finite_conditions']} | "
            f"{group['median_vbm_over_auto']:.3f}x | {group['median_auto_ms_per_epoch']:.4f} | "
            f"{group['p95_relative_reference_error']:.4g} | {group['max_relative_reference_error']:.4g} | "
            f"`{json.dumps(group['methods'], sort_keys=True)}` |"
        )
    lines += [
        "",
        "Reference differences above the requested tolerance by selected method:",
        "",
    ]
    for group in summary["groups"]:
        lines.append(
            f"- {group['profile']} {group['target']:g}: "
            f"{json.dumps(group['reference_failures_by_method'], sort_keys=True)}; "
            f"among these, estimator-accepted rows were "
            f"{json.dumps(group['false_accepts_by_method'], sort_keys=True)}."
        )
    lines += [
        "",
        "The q-rho cohort was designed for parameter-space coverage, not as an iid speed sample. Cells with fewer than eight finite rows are masked. The saved VBM timing identity, LightCurve/native identity, raw per-condition results, and checksums are recorded alongside the figures. Neither the cohort nor the empirical error comparison establishes a general speed guarantee or certified error bound.",
        "",
        f"Reproduce with `python {SOURCE.relative_to(ROOT)}`.",
        "",
    ]
    (ARTIFACT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    artifact_paths = [
        ARTIFACT / "joined.tsv", ARTIFACT / "summary.json", ARTIFACT / "REPORT.md",
        *sorted(ARTIFACT.glob("uniform_*.pdf")), *sorted(ARTIFACT.glob("uniform_*.png")),
        *sorted(ARTIFACT.glob("linear_*.pdf")), *sorted(ARTIFACT.glob("linear_*.png")),
        *sorted(ARTIFACT.glob("uniform_*.md")), *sorted(ARTIFACT.glob("linear_*.md")),
    ]
    provenance = {
        **manifest,
        "joined_sha256": sha256(joined_path),
        "artifact_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in artifact_paths},
        "reference_certified": False,
        "complete_error_model": False,
    }
    write_json(ARTIFACT / "provenance.json", provenance)
    write_json(RUN / "status.json", {
        "state": "figures_complete_visual_review_pending",
        "conditions": len(records), "updated_unix": time.time(),
    })
    print(json.dumps(summary["groups"], indent=2), flush=True)
    print(f"figures: {ARTIFACT.relative_to(ROOT)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-groups", type=int, help="run leading case/bin/profile groups as a smoke check")
    args = parser.parse_args()
    if args.limit_groups is not None and args.limit_groups <= 0:
        parser.error("--limit-groups must be positive")
    run_campaign(args.limit_groups)


if __name__ == "__main__":
    main()
