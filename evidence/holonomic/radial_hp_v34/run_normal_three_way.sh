#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
D=evidence/holonomic/radial_hp_v34
for direction in forward reverse; do
 stages=(v32 v33 v34)
 if [ "$direction" = reverse ]; then stages=(v34 v33 v32); fi
 for stage in "${stages[@]}"; do
  case "$stage" in
   v32) binary=build-seeded-events/bench_normal_before_v33 ;;
   v33) binary=build-seeded-events/bench_seeded_radial_master ;;
   v34) binary=build-seeded-events/bench_normal_hp_v34 ;;
  esac
  "$binary" evidence/holonomic/caustic_seeded_connection_v4/initial_minimal_seeds.tsv "$D/three_way_${direction}_${stage}.tsv" 5 0 > "$D/three_way_${direction}_${stage}.log" 2>&1
 done
 done
