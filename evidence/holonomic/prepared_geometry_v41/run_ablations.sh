#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
D=evidence/holonomic/prepared_geometry_v41
for direction in forward reverse; do
 stages=(baseline pair count both)
 if [ "$direction" = reverse ]; then stages=(both count pair baseline); fi
 for stage in "${stages[@]}"; do
  case "$stage" in
   baseline) binary=build-seeded-events/bench_planning_after_v35 ;;
   pair) binary=build-seeded-events/bench_prepared_pair_only_v41 ;;
   count) binary=build-seeded-events/bench_prepared_count_only_v41 ;;
   both) binary=build-seeded-events/bench_prepared_geometry_v41 ;;
  esac
  "$binary" evidence/holonomic/low_magnification_fixed_cost_v30/input.tsv "$D/ablation_${direction}_${stage}.tsv" 5 > "$D/ablation_${direction}_${stage}.log" 2>&1
 done
 done
