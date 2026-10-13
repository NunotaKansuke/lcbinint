#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
D=evidence/holonomic/ld_split_radial_v43
for direction in forward reverse; do
 stages=(before after)
 if [ "$direction" = reverse ]; then stages=(after before); fi
 for stage in "${stages[@]}"; do
  binary=build-seeded-events/bench_ld_handoff_v42
  if [ "$stage" = after ]; then binary=build-seeded-events/bench_ld_split_v43; fi
  "$binary" evidence/holonomic/low_magnification_fixed_cost_v30/input.tsv "$D/${direction}_${stage}.tsv" 5 > "$D/${direction}_${stage}.log" 2>&1
 done
 for stage in "${stages[@]}"; do
  binary=build-seeded-events/bench_normal_ld_handoff_v42
  if [ "$stage" = after ]; then binary=build-seeded-events/bench_normal_ld_split_v43; fi
  "$binary" evidence/holonomic/caustic_seeded_connection_v4/initial_minimal_seeds.tsv "$D/normal_${direction}_${stage}.tsv" 5 0 > "$D/normal_${direction}_${stage}.log" 2>&1
 done
 done
