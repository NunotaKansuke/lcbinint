#!/usr/bin/env bash
set -euo pipefail
export TMPDIR="$PWD/build-seeded-events"
D=evidence/holonomic/low_magnification_fixed_cost_v30
bash benchmarks/holonomic/build_seeded_image_seeds.sh
stages=(before indexed combined two zero)
if [ "$#" -gt 0 ]; then stages=("$@"); fi
for stage in "${stages[@]}"; do
  extra=()
  object=build-seeded-events/seed_indexed_v30.o
  case "$stage" in
    before) extra=(-I"$D/before_include"); object=build-seeded-events/seed_after_v26.o ;;
    indexed) extra=(-DHOLO_FRONTIER_EDGE_BISECTIONS=12) ;;
    combined) extra=(-I"$D/candidate_include" -DHOLO_FRONTIER_EDGE_BISECTIONS=4) ;;
    two) extra=(-I"$D/candidate_include" -DHOLO_FRONTIER_EDGE_BISECTIONS=2) ;;
    zero) extra=(-I"$D/candidate_include" -DHOLO_FRONTIER_EDGE_BISECTIONS=0) ;;
  esac
  g++ "${extra[@]}" -std=c++17 -O3 -DNDEBUG -march=native -fext-numeric-literals \
    -I/rogue1_8/nunota/lcbinint/src -Isrc -I/rogue1_8/nunota/lcbinint/include \
    benchmarks/holonomic/bench_seeded_fixed_cost.cpp "$object" \
    -Wl,--start-group build-standard-current/liblcbinint_magnification.a \
    build-standard-current/liblcbinint_lightcurve.a -Wl,--end-group \
    -L/home/nunota/.miniconda3/envs/myenv/lib \
    -Wl,-rpath,/home/nunota/.miniconda3/envs/myenv/lib \
    -lgsl -lgslcblas -lquadmath -fopenmp \
    -o "build-seeded-events/bench_fixed_cost_${stage}_v30"
done
for stage in before indexed; do
  object=build-seeded-events/seed_after_v26.o
  if [ "$stage" = indexed ]; then object=build-seeded-events/seed_indexed_v30.o; fi
  g++ -std=c++17 -O3 -DNDEBUG -march=native \
    -I/rogue1_8/nunota/lcbinint/src -I/rogue1_8/nunota/lcbinint/include \
    evidence/holonomic/seed_count_v26/audit_seed.cpp "$object" \
    -Wl,--start-group build-standard-current/liblcbinint_magnification.a \
    build-standard-current/liblcbinint_lightcurve.a -Wl,--end-group \
    -L/home/nunota/.miniconda3/envs/myenv/lib \
    -Wl,-rpath,/home/nunota/.miniconda3/envs/myenv/lib \
    -lgsl -lgslcblas -lquadmath -fopenmp \
    -o "build-seeded-events/audit_seed_${stage}_v30"
done
