#!/usr/bin/env bash
set -eu
# Run from the existing experimental checkout. Finish all compilation before timing.
g++ -std=c++17 -O3 -march=native -fext-numeric-literals -DHOLO_ARC_SIGN_AUDIT -DHOLO_ADAPTIVE_ANCHOR_AUDIT -Isrc tests/holonomic_cpp/test_adaptive_radial.cpp -lquadmath -o build-seeded-events/test_adaptive_v18
OMP_NUM_THREADS=1 build-seeded-events/test_adaptive_v18 > evidence/holonomic/node_dispatch_v18/adaptive_final.log 2>&1
g++ -std=c++17 -O3 -march=native -fext-numeric-literals -DHOLO_ARC_SIGN_AUDIT -DHOLO_ADAPTIVE_ANCHOR_AUDIT -Isrc tests/holonomic_cpp/test_seeded_radial_frontier.cpp -lquadmath -o build-seeded-events/test_frontier_v18
OMP_NUM_THREADS=1 build-seeded-events/test_frontier_v18 > evidence/holonomic/node_dispatch_v18/frontier_final.log 2>&1
