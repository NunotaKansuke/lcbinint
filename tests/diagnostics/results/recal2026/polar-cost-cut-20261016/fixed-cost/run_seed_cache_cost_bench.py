import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
INPUT = HERE / "source-positions.tsv"
OLD = HERE / "seed-cache-probe-fair-vector"
CANDIDATE = HERE / "seed-cache-probe-lazy-index"
OPERATIONS = 300_000
REPEATS = 5
CAPACITIES = (1, 4, 8, 16, 20, 24, 32, 48, 64, 256, 1024)
FIRST_REPLAY_CAPACITIES = (64, 128, 256, 1024)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(binary, capacity, operations):
    output = subprocess.check_output(
        [str(binary), str(INPUT), str(capacity), str(operations)], text=True
    )
    return {line.split("\t")[0]: float(line.split("\t")[1])
            for line in output.splitlines()}


def main():
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    binaries = {"vector": OLD, "lazy_index": CANDIDATE}
    warm = {}
    for capacity in CAPACITIES:
        runs = {name: [] for name in binaries}
        for repeat in range(REPEATS):
            order = tuple(binaries) if repeat % 2 == 0 else tuple(reversed(binaries))
            for name in order:
                runs[name].append(run(binaries[name], capacity, OPERATIONS))
        phases = {}
        for phase in ("oldest_hit", "newest_hit", "cyclic_hits"):
            before = [row[phase] for row in runs["vector"]]
            after = [row[phase] for row in runs["lazy_index"]]
            phases[phase] = {
                "vector_median_ns": statistics.median(before),
                "lazy_index_median_ns": statistics.median(after),
                "paired_median_ratio": statistics.median(
                    b / a for a, b in zip(before, after)
                ),
                "vector_samples_ns": before,
                "lazy_index_samples_ns": after,
            }
        warm[str(capacity)] = phases

    fill = {name: [] for name in binaries}
    for repeat in range(REPEATS):
        order = tuple(binaries) if repeat % 2 == 0 else tuple(reversed(binaries))
        for name in order:
            start = time.perf_counter()
            run(binaries[name], 1024, 1)
            fill[name].append((time.perf_counter() - start) * 1e3)
    first_replay = {}
    for capacity in FIRST_REPLAY_CAPACITIES:
        runs = {name: [] for name in binaries}
        for repeat in range(REPEATS):
            order = tuple(binaries) if repeat % 2 == 0 else tuple(reversed(binaries))
            for name in order:
                start = time.perf_counter()
                run(binaries[name], capacity, 1)
                runs[name].append((time.perf_counter() - start) * 1e3)
        before = runs["vector"]
        after = runs["lazy_index"]
        first_replay[str(capacity)] = {
            "vector_median_ms": statistics.median(before),
            "lazy_index_median_ms": statistics.median(after),
            "paired_median_ratio": statistics.median(
                a / b for a, b in zip(before, after)
            ),
            "vector_samples_ms": before,
            "lazy_index_samples_ms": after,
        }
    payload = {
        "warm_scope": (
            "Warm cache lookup plus returned seed-vector copy/checksum; population excluded; "
            "CPU pinned; 300000 operations per phase; alternating paired order."
        ),
        "cold_population_scope": (
            "Fresh process: construct point images, populate 1024 distinct seed-cache "
            "entries, request all retained entries for exact identity verification (the "
            "first repeated pass), and run three single-hit checks. Process startup included; "
            "integration map excluded."
        ),
        "input_unique_positions": 1024,
        "repeats": REPEATS,
        "native_source_sha256": sha256(ROOT / "src/lcbinint/magnification/finite_source_magnifier.cpp"),
        "native_header_sha256": sha256(ROOT / "src/lcbinint/magnification/finite_source_magnifier.hpp"),
        "probe_source_sha256": sha256(HERE / "seed_cache_probe_timing.cpp"),
        "benchmark_script_sha256": sha256(Path(__file__).resolve()),
        "input_sha256": sha256(INPUT),
        "binaries_sha256": {name: sha256(path) for name, path in binaries.items()},
        "warm_lookup": warm,
        "cold_population_wall_ms": {
            "vector_median_ms": statistics.median(fill["vector"]),
            "lazy_index_median_ms": statistics.median(fill["lazy_index"]),
            "paired_median_ratio": statistics.median(
                a / b for a, b in zip(fill["vector"], fill["lazy_index"])
            ),
            "vector_samples_ms": fill["vector"],
            "lazy_index_samples_ms": fill["lazy_index"],
        },
        "first_replay_wall_ms": first_replay,
    }
    output = HERE / "timings-final-lazy-index.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(output)
    for capacity in CAPACITIES:
        phase = warm[str(capacity)]["cyclic_hits"]
        delta = phase["lazy_index_median_ns"] - phase["vector_median_ns"]
        print(
            f"{capacity}\t{phase['vector_median_ns']:.2f}\t"
            f"{phase['lazy_index_median_ns']:.2f}\t{delta:+.2f}\t"
            f"{phase['paired_median_ratio']:.3f}x"
        )
    cold = payload["cold_population_wall_ms"]
    print(
        "fill-plus-identity-replay\t"
        f"{cold['vector_median_ms']:.3f}\t"
        f"{cold['lazy_index_median_ms']:.3f}\t"
        f"{cold['paired_median_ratio']:.3f}x"
    )
    for capacity, row in first_replay.items():
        print(
            f"first-replay-{capacity}\t{row['vector_median_ms']:.3f}\t"
            f"{row['lazy_index_median_ms']:.3f}\t"
            f"{row['paired_median_ratio']:.3f}x"
        )


if __name__ == "__main__":
    main()
