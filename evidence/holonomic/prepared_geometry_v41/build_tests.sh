#!/usr/bin/env bash
set -euo pipefail
export TMPDIR="$PWD/build-seeded-events"
D=evidence/holonomic/prepared_geometry_v41
for test in weighted_radial adaptive_radial seeded_radial_frontier; do
 g++ -std=c++17 -O3 -march=native -fext-numeric-literals -DHOLO_TRANSPORT_DIRECT_AUDIT -DHOLO_ADAPTIVE_FOLD_MAP_AUDIT -DHOLO_ARC_SIGN_AUDIT -DHOLO_ADAPTIVE_ANCHOR_AUDIT -DHOLO_ADAPTIVE_COUNT_WITNESS_AUDIT -Isrc "tests/holonomic_cpp/test_${test}.cpp" -lquadmath -o "build-seeded-events/test_${test}_v41"
 env OMP_NUM_THREADS=1 "build-seeded-events/test_${test}_v41" > "$D/test_${test}.log" 2>&1
done
