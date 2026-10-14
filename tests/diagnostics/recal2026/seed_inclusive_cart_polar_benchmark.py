#!/usr/bin/env python3
"""Measure Cartesian and Polar with either a fresh or cached source seed set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
INPUTS = ROOT / "figures/recal2026/adaptive-vbm-six-panel-20261014/inputs.tsv"
RUN = ROOT / "tests/diagnostics/results/recal2026/seed-inclusive-cart-polar-20261015"
BUILD = ROOT / "build-adaptive-production"
RUNNER_SOURCE = ROOT / "tests/diagnostics/recal2026/seed_inclusive_cart_polar_runner.cpp"
CHUNK_SIZE = 32
TARGETS = (1.0e-3, 1.0e-4)
METHODS = ("cartesian", "polar")
FIELDS = [
    "sample_id", "method", "target", "repeat", "center_root_us", "caustic_build_ms",
    "seed_only_us", "seed_count", "support_proven", "cold_ms", "warm_ms",
    "cold_magnification", "warm_magnification", "cold_indicator", "warm_indicator",
    "cold_met", "warm_met", "cold_maps", "warm_maps", "cold_subdivisions",
    "warm_subdivisions", "cold_warm_identical", "stop_reason",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def select_inputs(rows: list[dict[str, str]], count: int | None) -> list[dict[str, str]]:
    if count is None:
        return rows
    if count < 1:
        raise ValueError("pilot size must be positive")
    rng = random.Random(20261015)
    selected: list[dict[str, str]] = []
    profiles = sorted({row["profile"] for row in rows})
    counts = {profile: count // len(profiles) for profile in profiles}
    for profile in profiles[:count % len(profiles)]:
        counts[profile] += 1
    for profile in profiles:
        group = [row for row in rows if row["profile"] == profile]
        selected.extend(rng.sample(group, min(counts[profile], len(group))))
    return sorted(selected, key=lambda row: int(row["geom_id"]))


def compile_runner(output: Path) -> list[str]:
    executable = output / "runner"
    command = [
        "/usr/bin/c++", "-O3", "-DNDEBUG", "-std=gnu++17", "-fopenmp",
        "-I" + str(ROOT / "include"), "-I" + str(ROOT / "src"),
        str(RUNNER_SOURCE), "-o", str(executable), "-Wl,--start-group",
        str(BUILD / "liblcbinint_lightcurve.a"),
        str(BUILD / "liblcbinint_magnification.a"), "-Wl,--end-group",
        "-Wl,-rpath,/home/nunota/.miniconda3/envs/myenv/lib",
        "/home/nunota/.miniconda3/envs/myenv/lib/libgsl.so",
        "/home/nunota/.miniconda3/envs/myenv/lib/libgslcblas.so",
        "-lquadmath", "-lm",
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    return command


def native_line(row: dict[str, str]) -> str:
    # Match the established Polar runner: native source x is trajectory time.
    return "\t".join(row[key] for key in ("geom_id", "s", "q", "rho", "time", "y", "c"))


def valid_shard(path: Path, ids: list[str]) -> bool:
    if not path.exists():
        return False
    rows = read_tsv(path)
    expected = {(sample_id, method, f"{target:.17g}", repeat)
                for sample_id in ids for method in METHODS
                for target in TARGETS for repeat in range(3)}
    actual = {(row["sample_id"], row["method"], f"{float(row['target']):.17g}",
               int(row["repeat"])) for row in rows}
    return len(rows) == len(expected) and actual == expected


def summarize(
    rows: list[dict[str, str]],
    selected: list[dict[str, str]],
    reference_by_id: dict[str, float],
) -> dict:
    profile_by_id = {row["geom_id"]: row["profile"] for row in selected}
    by_case: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["sample_id"] not in profile_by_id:
            continue
        by_case[(profile_by_id[row["sample_id"]], row["method"],
                 f"{float(row['target']):.17g}")].append(row)

    groups = []
    for profile in sorted(set(profile_by_id.values())):
        for method in METHODS:
            for target in TARGETS:
                key = (profile, method, f"{target:.17g}")
                cases = by_case.get(key, [])
                by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
                for row in cases:
                    by_id[row["sample_id"]].append(row)
                per_case = []
                for sample_id, reps in by_id.items():
                    reps.sort(key=lambda row: int(row["repeat"]))
                    if len(reps) != 3:
                        continue
                    cold = statistics.median(float(row["cold_ms"]) for row in reps)
                    warm = statistics.median(float(row["warm_ms"]) for row in reps)
                    actual_error = abs(
                        float(reps[0]["cold_magnification"]) / reference_by_id[sample_id] - 1.0
                    )
                    target = float(reps[0]["target"])
                    met = all(row["cold_met"] == "1" for row in reps)
                    per_case.append({
                        "seed_us": float(reps[0]["seed_only_us"]),
                        "cold_ms": cold,
                        "warm_ms": warm,
                        "incremental_us": (cold - warm) * 1000.0,
                        "identical": all(row["cold_warm_identical"] == "1" for row in reps),
                        "support": all(row["support_proven"] == "1" for row in reps),
                        "met": all(row["cold_met"] == row["warm_met"] for row in reps),
                        "actual_error_over_target": actual_error / target,
                        "actual_pass": actual_error <= target,
                        "false_accept": met and actual_error > target,
                        "maps": all(row["cold_maps"] == row["warm_maps"] for row in reps),
                    })
                groups.append({
                    "profile": profile,
                    "method": method,
                    "target": target,
                    "n": len(per_case),
                    "median_seed_us": statistics.median(v["seed_us"] for v in per_case) if per_case else None,
                    "median_cold_ms": statistics.median(v["cold_ms"] for v in per_case) if per_case else None,
                    "median_warm_ms": statistics.median(v["warm_ms"] for v in per_case) if per_case else None,
                    "median_seed_increment_us": statistics.median(v["incremental_us"] for v in per_case) if per_case else None,
                    "cold_warm_identical": sum(v["identical"] for v in per_case),
                    "support_proven": sum(v["support"] for v in per_case),
                    "met_status_pairs_equal": sum(v["met"] for v in per_case),
                    "map_count_pairs_equal": sum(v["maps"] for v in per_case),
                    "actual_pass": sum(v["actual_pass"] for v in per_case),
                    "false_accepts": sum(v["false_accept"] for v in per_case),
                    "median_actual_error_over_target": statistics.median(
                        v["actual_error_over_target"] for v in per_case
                    ) if per_case else None,
                })
    return {"groups": groups, "row_count": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=int, help="run a deterministic, profile-stratified subset")
    args = parser.parse_args()
    rows = read_tsv(INPUTS)
    if len(rows) != 7216:
        raise RuntimeError(f"expected 7216 campaign inputs, found {len(rows)}")
    selected = select_inputs(rows, args.pilot)
    reference_by_id = {row["geom_id"]: float(row["reference"]) for row in selected}
    adjudications = ROOT / "figures/recal2026/polar-relative-matched-six-panel-20261015/reference-adjudications.tsv"
    if adjudications.exists():
        for row in read_tsv(adjudications):
            if row["geom_id"] in reference_by_id:
                reference_by_id[row["geom_id"]] = float(row["adjudicated_reference"])
    output = RUN / (f"pilot-{len(selected)}" if args.pilot is not None else "full")
    output.mkdir(parents=True, exist_ok=True)
    (output / "shards").mkdir(exist_ok=True)

    cpu_set = os.sched_getaffinity(0)
    cpu = 0 if 0 in cpu_set else min(cpu_set)
    os.sched_setaffinity(0, {cpu})
    for key in list(os.environ):
        if key.startswith("LCBININT_DIAGNOSTIC_") or key in {
            "LCBININT_REPRESENTATIVE_PROBES",
            "LCBININT_REPRESENTATIVE_COUNT_ABLATION",
            "LCBININT_CRITICAL_SUPPORT_CERTIFICATE",
            "LCBININT_SEED_FULL_COVERAGE_AUDIT",
            "LCBININT_SEED_LATTICE_CERTIFICATE",
        }:
            os.environ.pop(key)
    os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")

    command = compile_runner(output)
    manifest = {
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "inputs_sha256": sha256(INPUTS),
        "selected_ids_sha256": hashlib.sha256(
            "\n".join(row["geom_id"] for row in selected).encode()).hexdigest(),
        "selected_rows": len(selected),
        "runner_sha256": sha256(output / "runner"),
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in (
                RUNNER_SOURCE,
                ROOT / "src/lcbinint/magnification/finite_source_magnifier.cpp",
                ROOT / "src/lcbinint/magnification/finite_source_magnifier.hpp",
                ROOT / "src/lcbinint/magnification/adaptive_polar_impl.inc",
                ROOT / "src/lcbinint/magnification/adaptive_cartesian_impl.inc",
                BUILD / "liblcbinint_magnification.a",
                BUILD / "liblcbinint_lightcurve.a",
            )
        },
        "command": command,
        "host": platform.node(),
        "cpu": cpu,
        "threads": 1,
        "targets": list(TARGETS),
        "timing": "center roots and caustic geometry separately timed; seed cache miss or hit plus adaptive kernel in each call; caustic geometry remains warm",
        "seed_provider": "cached_binary_image_seeds via fixed_inverse_ray_binary for both methods",
    }
    write_json(output / "manifest.json", manifest)

    started = time.time()
    shards = output / "shards"
    all_rows: list[dict[str, str]] = []
    shard_total = (len(selected) + CHUNK_SIZE - 1) // CHUNK_SIZE
    for shard_index, offset in enumerate(range(0, len(selected), CHUNK_SIZE)):
        part = selected[offset:offset + CHUNK_SIZE]
        ids = [row["geom_id"] for row in part]
        input_path = shards / f"{shard_index:04d}-input.tsv"
        raw_path = shards / f"{shard_index:04d}-raw.tsv"
        log_path = shards / f"{shard_index:04d}.log"
        input_path.write_text("\n".join(native_line(row) for row in part) + "\n")
        if not valid_shard(raw_path, ids):
            if raw_path.exists():
                raw_path.rename(raw_path.with_suffix(".incomplete.tsv"))
            with log_path.open("a") as log:
                subprocess.run([str(output / "runner"), str(input_path), str(raw_path)],
                               cwd=ROOT, stdout=log, stderr=log, check=True)
        if not valid_shard(raw_path, ids):
            raise RuntimeError(f"shard {shard_index} incomplete or mismatched")
        all_rows.extend(read_tsv(raw_path))
        write_json(output / "status.json", {
            "state": "running", "completed_inputs": min(offset + len(part), len(selected)),
            "total_inputs": len(selected), "completed_shards": shard_index + 1,
            "total_shards": shard_total, "elapsed_seconds": time.time() - started,
            "updated_unix": time.time(),
        })
        print(f"progress {min(offset + len(part), len(selected))}/{len(selected)} "
              f"elapsed={time.time() - started:.1f}s", flush=True)

    write_tsv(output / "raw.tsv", all_rows)
    summary = summarize(all_rows, selected, reference_by_id)
    write_json(output / "summary.json", summary)
    with (output / "REPORT.md").open("w") as report:
        report.write("# Seed-inclusive Cartesian/Polar timing\n\n")
        report.write("The seed provider is the indexed `cached_binary_image_seeds` path, shared by the Cartesian and Polar `fixed_inverse_ray_binary` routes. Each source geometry warms caustic geometry once. Timed cold calls clear only the source-seed cache; warm calls reuse the exact seed vector. Center point-image roots and caustic construction are timed separately and excluded from `cold_ms`/`warm_ms`.\n\n")
        report.write("`cold_ms` therefore includes fast seed construction plus adaptive integration; `warm_ms` includes a seed-cache hit plus the same integration. `median_seed_increment_us` is the paired end-to-end difference and may include timing noise. Seed-only cost is shown independently. The 10/14 port checkpoint previously measured a 7.0–7.2 µs cohort median under its 192-input provider protocol; this run measures its own input cohort.\n\n")
        report.write("| profile | method | target | n | seed only median (µs) | seed-inclusive median (ms) | warm median (ms) | paired delta (µs) | identical cold/warm | support proven | actual pass | false accepts | median actual error / target |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for group in summary["groups"]:
            report.write(
                f"| {group['profile']} | {group['method']} | {group['target']:g} | {group['n']} | "
                f"{group['median_seed_us']:.2f} | {group['median_cold_ms']:.5f} | "
                f"{group['median_warm_ms']:.5f} | {group['median_seed_increment_us']:.2f} | "
                f"{group['cold_warm_identical']}/{group['n']} | "
                f"{group['support_proven']}/{group['n']} | {group['actual_pass']}/{group['n']} | "
                f"{group['false_accepts']} | {group['median_actual_error_over_target']:.4g} |\n"
            )
        report.write("\nAccuracy is compared to the saved RelTol=1e-6 reference corpus, with the three separately adjudicated reference corrections applied. It is empirical agreement, not a formal error bound. Exact cold/warm equality requires the magnification, indicator, status, map count and refinement counts to agree for all three repeats of an input.\n")
    write_json(output / "status.json", {
        "state": "complete", "completed_inputs": len(selected), "total_inputs": len(selected),
        "elapsed_seconds": time.time() - started, "updated_unix": time.time(),
    })
    print(f"complete: {output.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
