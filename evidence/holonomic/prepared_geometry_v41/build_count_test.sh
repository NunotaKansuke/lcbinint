#!/usr/bin/env bash
set -euo pipefail
export TMPDIR="$PWD/build-seeded-events"
g++ -std=c++17 -O3 -march=native -fext-numeric-literals -Isrc tests/holonomic_cpp/test_quartic_fast_count.cpp -lquadmath -o build-seeded-events/test_quartic_fast_count_v41
OMP_NUM_THREADS=1 build-seeded-events/test_quartic_fast_count_v41 > evidence/holonomic/prepared_geometry_v41/test_quartic_fast_count.log 2>&1
