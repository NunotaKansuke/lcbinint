"""Matched direct-polar pilot for the 2026-10-16 cost-cut experiment."""
import csv
import hashlib
import json
import math
import os
import argparse
from pathlib import Path
import random
import statistics
import subprocess
import time

import make_polar_adaptive_vbm_six_panel_20261014 as bench

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "tests/diagnostics/results/recal2026/polar-cost-cut-20261016"
RUN = OUT / "paired-pilot-128-plus-edge-v1"
PARENT = OUT / "parent-e27f2663-runner"
CANDIDATE = OUT / "candidate-polynomial-cut-v1-runner"
MANDATORY = {"1360", "1361", "1364", "1366", "3486", "3679", "1717", "5099"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_rows(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default=RUN.name)
    parser.add_argument("--candidate", type=Path, default=CANDIDATE)
    args = parser.parse_args()
    run_dir = OUT / args.run_name
    candidate_executable = args.candidate
    if not candidate_executable.is_absolute():
        candidate_executable = ROOT / candidate_executable
    if run_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing run: {run_dir}")
    cpu = 0 if 0 in os.sched_getaffinity(0) else min(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpu})
    os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    inputs = bench.read_inputs()
    selected = random.Random(20261014097).sample(inputs, 128)
    selected_ids = {row["geom_id"] for row in selected}
    selected.extend(row for row in inputs
                    if row["geom_id"] in MANDATORY - selected_ids)
    run_dir.mkdir(parents=True)
    input_path = run_dir / "inputs.tsv"
    with input_path.open("w") as stream:
        for row in selected:
            stream.write("\t".join(row[k] for k in
                ("geom_id", "s", "q", "rho", "time", "y", "c")) + "\n")
    manifest = {
        "baseline_commit": "e27f2663b3f2cebc3f3a6bf1196f7b2268f7e197",
        "parent_runner_sha256": sha256(PARENT),
        "candidate_runner_sha256": sha256(candidate_executable),
        "source_files": {
            path: sha256(ROOT / path) for path in (
                "src/lcbinint/magnification/finite_source_magnifier.cpp",
                "src/lcbinint/magnification/adaptive_polar_impl.inc",
                "src/lcbinint/magnification/adaptive_polar_roots.hpp",
                "tests/diagnostics/recal2026/polar_adaptive_six_panel_runner.cpp",
            )
        },
        "input_source_sha256": sha256(bench.INPUTS),
        "selected_input_sha256": sha256(input_path),
        "input_count": len(selected),
        "mandatory_ids_added": sorted(MANDATORY - selected_ids),
        "targets": [1e-3, 1e-4],
        "polynomial_shadow": os.environ.get("LCBININT_DIAGNOSTIC_POLAR_POLYNOMIAL_SHADOW") == "1",
        "cpu_affinity": [cpu],
        "threads": 1,
        "timing": "per-geometry paired executable order alternates; each call has one warmup and median of three",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    baseline_rows, candidate_rows = [], []
    started = time.monotonic()
    for index, row in enumerate(selected):
        one = run_dir / f"{index:04d}-input.tsv"
        one.write_text("\t".join(row[k] for k in
            ("geom_id", "s", "q", "rho", "time", "y", "c")) + "\n")
        order = [("parent", PARENT), ("candidate", candidate_executable)]
        if index % 2:
            order.reverse()
        for label, executable in order:
            output = run_dir / f"{index:04d}-{label}.tsv"
            with (run_dir / f"{index:04d}-{label}.log").open("w") as log:
                subprocess.run([str(executable), str(one), str(output)], cwd=ROOT,
                               stdout=log, stderr=log, check=True, timeout=600)
            values = load_rows(output)
            if len(values) != 2:
                raise RuntimeError(f"expected two targets for geometry {row['geom_id']}")
            (baseline_rows if label == "parent" else candidate_rows).extend(values)
        if (index + 1) % 32 == 0:
            print(f"completed {index + 1}/{len(selected)}", flush=True)
    write_rows(run_dir / "parent.tsv", baseline_rows)
    write_rows(run_dir / "candidate.tsv", candidate_rows)
    inputs_by_id = {row["geom_id"]: row for row in selected}
    parents = {(row["sample_id"], float(row["target"])): row for row in baseline_rows}
    joined = []
    for candidate in candidate_rows:
        key = (candidate["sample_id"], float(candidate["target"]))
        parent = parents[key]
        inp = inputs_by_id[key[0]]
        reference = float(inp["reference"])
        tolerance = key[1]
        parent_error = abs(float(parent["magnification"]) / reference - 1.0)
        candidate_error = abs(float(candidate["magnification"]) / reference - 1.0)
        joined.append({
            "id": key[0], "profile": inp["profile"], "target": tolerance,
            "parent": parent, "candidate": candidate,
            "parent_error": parent_error, "candidate_error": candidate_error,
            "speed_ratio": float(parent["ms"]) / float(candidate["ms"]),
        })
    groups = []
    for profile in ("uniform", "linear"):
        for target in (1e-3, 1e-4):
            rows = [row for row in joined
                    if row["profile"] == profile and row["target"] == target]
            ratios = [row["speed_ratio"] for row in rows]
            groups.append({
                "profile": profile, "target": target, "conditions": len(rows),
                "median_parent_over_candidate": statistics.median(ratios),
                "median_parent_ms": statistics.median(float(r["parent"]["ms"]) for r in rows),
                "median_candidate_ms": statistics.median(float(r["candidate"]["ms"]) for r in rows),
                "median_map_ratio": statistics.median(
                    int(r["parent"]["maps"]) / max(1, int(r["candidate"]["maps"])) for r in rows),
                "parent_false_accepts": sum(r["parent"]["converged"] == "1" and
                                             r["parent_error"] > target for r in rows),
                "candidate_false_accepts": sum(r["candidate"]["converged"] == "1" and
                                                r["candidate_error"] > target for r in rows),
                "parent_exceptions": sum(r["parent"]["stop_reason"] == "exception" for r in rows),
                "candidate_exceptions": sum(r["candidate"]["stop_reason"] == "exception" for r in rows),
                "new_false_accepts": [r["id"] for r in rows
                    if r["candidate"]["converged"] == "1" and r["candidate_error"] > target and
                    not (r["parent"]["converged"] == "1" and r["parent_error"] > target)],
                "rejected": sum(r["candidate"]["converged"] != "1" for r in rows),
            })
    summary = {
        "inputs": len(selected), "conditions": len(joined),
        "elapsed_seconds": time.monotonic() - started,
        "groups": groups,
        "exact_magnification_matches": sum(
            r["parent"]["magnification"] == r["candidate"]["magnification"] for r in joined),
        "exact_indicator_matches": sum(
            r["parent"]["indicator"] == r["candidate"]["indicator"] for r in joined),
        "max_relative_magnification_delta": max(
            abs(float(r["parent"]["magnification"]) /
                float(r["candidate"]["magnification"]) - 1.0) for r in joined
            if math.isfinite(float(r["candidate"]["magnification"]))),
        "certified": False, "complete_error_model": False,
    }
    (run_dir / "joined.json").write_text(json.dumps(joined, indent=2) + "\n")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
