#!/usr/bin/env bash
set -eu
cd "$(git rev-parse --show-toplevel)"
out=evidence/holonomic/adaptive_extreme_phase50
flags=(-O3 -DNDEBUG -std=gnu++17 -fext-numeric-literals -Isrc -march=native -funroll-loops -ffp-contract=fast -fno-math-errno -DHOLO_D14_NATIVE_RESIDUAL_SCREEN -DHOLO_D14_DIRECT_CONVOLUTION -DHOLO_ADAPTIVE_STATIONARY_ENDPOINT -DHOLO_D14_WARM_NOISE_HANDOFF -DHOLO_D14_SCALAR_CONSTANTS -DHOLO_ARC_COMPACT_STORAGE -DHOLO_ADAPTIVE_SPARSE_ANCHOR -DHOLO_ADAPTIVE_SAMPLE_GROWTH -DHOLO_ENDPOINT_WORK_PROBE)
c++ "${flags[@]}" benchmarks/holonomic/bench_adaptive_stage_profile.cpp -o /tmp/p50_profile -lquadmath
taskset -c 0 /tmp/p50_profile evidence/holonomic/v2_adaptive_best_trajectory_20260913/input_snapshot.tsv warm-timing > "$out/profile.tsv"
python benchmarks/holonomic/summarize_endpoint_work.py --baseline evidence/holonomic/adaptive_extreme_phase48/profile_candidate.tsv --probe "$out/profile.tsv" --output "$out/summary.json"
c++ -O3 -DNDEBUG -std=gnu++17 -fext-numeric-literals -Isrc -DHOLO_ENDPOINT_WORK_PROBE -DHOLO_ADAPTIVE_STATIONARY_ENDPOINT tests/holonomic_cpp/test_adaptive_radial.cpp -o /tmp/p50_unit -lquadmath
/tmp/p50_unit > "$out/unit.txt"
