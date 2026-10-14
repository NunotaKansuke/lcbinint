#!/usr/bin/env bash
set -euo pipefail
export TMPDIR="$PWD/build-seeded-events"
bash benchmarks/holonomic/build_seeded_image_seeds.sh
D=evidence/holonomic/planning_count_v31
stages=(before after profile_before profile_after audit)
if [ "$#" -gt 0 ]; then stages=("$@"); fi
for stage in "${stages[@]}"; do
  extra=()
  case "$stage" in
    before) extra=(-DHOLO_PLANNING_BASELINE -I"$D/before_include") ;;
    after) ;;
    profile_before) extra=(-DHOLO_PLANNING_BASELINE -DHOLO_PLANNING_PROFILE -I"$D/before_include") ;;
    profile_after) extra=(-DHOLO_PLANNING_PROFILE) ;;
    audit) extra=(-DHOLO_QUARTIC_COUNT_AUDIT) ;;
    *) exit 2 ;;
  esac
  g++ "${extra[@]}" -std=c++17 -O3 -DNDEBUG -march=native -fext-numeric-literals \
    -I/rogue1_8/nunota/lcbinint/src -Isrc -I/rogue1_8/nunota/lcbinint/include \
    benchmarks/holonomic/bench_seeded_planning.cpp build-seeded-events/seed_indexed_v30.o \
    -Wl,--start-group build-standard-current/liblcbinint_magnification.a \
    build-standard-current/liblcbinint_lightcurve.a -Wl,--end-group \
    -L/home/nunota/.miniconda3/envs/myenv/lib \
    -Wl,-rpath,/home/nunota/.miniconda3/envs/myenv/lib \
    -lgsl -lgslcblas -lquadmath -fopenmp \
    -o "build-seeded-events/bench_planning_${stage}_v31"
done
