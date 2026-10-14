#!/usr/bin/env bash
set -eu
# Run from this checkout. Complete all compilation before accepted timing runs.
g++ -std=c++17 -O3 -march=native -fext-numeric-literals -Isrc tests/holonomic_cpp/test_chebyshev_prediction.cpp -lquadmath -o build-seeded-events/test_prediction_v20
build-seeded-events/test_prediction_v20 > evidence/holonomic/interpolation_v20/prediction_test.log 2>&1
g++ -std=c++17 -O3 -march=native -fext-numeric-literals -DHOLO_ARC_SIGN_AUDIT -DHOLO_ADAPTIVE_ANCHOR_AUDIT -DHOLO_ADAPTIVE_COUNT_WITNESS_AUDIT -Isrc tests/holonomic_cpp/test_adaptive_radial.cpp -lquadmath -o build-seeded-events/test_adaptive_v20
OMP_NUM_THREADS=1 build-seeded-events/test_adaptive_v20 > evidence/holonomic/interpolation_v20/adaptive.log 2>&1
g++ -std=c++17 -O3 -march=native -fext-numeric-literals -DHOLO_ARC_SIGN_AUDIT -DHOLO_ADAPTIVE_ANCHOR_AUDIT -DHOLO_ADAPTIVE_COUNT_WITNESS_AUDIT -Isrc tests/holonomic_cpp/test_seeded_radial_frontier.cpp -lquadmath -o build-seeded-events/test_frontier_v20
OMP_NUM_THREADS=1 build-seeded-events/test_frontier_v20 > evidence/holonomic/interpolation_v20/frontier.log 2>&1
