#!/usr/bin/env bash
# Run from the repository root. Timed programs run sequentially, one CPU each.
set -euo pipefail
adaptive_evidence=${1:-evidence/holonomic/adaptive_radial_20260911}
adaptive_repeats=${2:-3}
mkdir -p "$adaptive_evidence"
cmake -S tests/holonomic_cpp -B build-holonomic-m7 -DCMAKE_BUILD_TYPE=Release
cmake --build build-holonomic-m7 --target test_adaptive_radial bench_adaptive_radial bench_adaptive_contracts bench_adaptive_controls check_adaptive_reference check_adaptive_jacobian check_adaptive_limits -j2
./build-holonomic-m7/test_adaptive_radial > "$adaptive_evidence/unit.txt"
./build-holonomic-m7/check_adaptive_reference "$adaptive_evidence/reference_cases.tsv" > "$adaptive_evidence/reference.csv"
./build-holonomic-m7/check_adaptive_reference "$adaptive_evidence/reference_warm_cases.tsv" > "$adaptive_evidence/reference_warm.csv"
./build-holonomic-m7/check_adaptive_limits "$adaptive_evidence/reference_cases.tsv" > "$adaptive_evidence/limits.csv"
./build-holonomic-m7/check_adaptive_jacobian > "$adaptive_evidence/jacobian_reference.csv"
./build-holonomic-m7/bench_adaptive_radial evidence/holonomic/gm_coverage_cases.tsv "$adaptive_repeats" > "$adaptive_evidence/paired.csv"
./build-holonomic-m7/bench_adaptive_contracts "$adaptive_evidence/reference_cases.tsv" "$adaptive_repeats" > "$adaptive_evidence/contracts.csv"
./build-holonomic-m7/bench_adaptive_controls "$adaptive_evidence/reference_cases.tsv" > "$adaptive_evidence/controls.csv"
python checks/holonomic/summarize_adaptive_radial.py "$adaptive_evidence" > "$adaptive_evidence/summary.txt"
