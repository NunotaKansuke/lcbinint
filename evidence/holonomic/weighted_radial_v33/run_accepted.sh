#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
D=evidence/holonomic/weighted_radial_v33
INPUT=evidence/holonomic/low_magnification_fixed_cost_v30/input.tsv
for stage in before after; do
 "build-seeded-events/bench_weighted_radial_${stage}_v33" "$INPUT" "$D/forward_${stage}.tsv" 5 > "$D/forward_${stage}.log" 2>&1
done
for stage in after before; do
 "build-seeded-events/bench_weighted_radial_${stage}_v33" "$INPUT" "$D/reverse_${stage}.tsv" 5 > "$D/reverse_${stage}.log" 2>&1
done
for direction in forward reverse; do
 stages=(before after)
 if [ "$direction" = reverse ]; then stages=(after before); fi
 for stage in "${stages[@]}"; do
  binary=build-seeded-events/bench_normal_before_v33
  if [ "$stage" = after ]; then binary=build-seeded-events/bench_seeded_radial_master; fi
  "$binary" evidence/holonomic/caustic_seeded_connection_v4/initial_minimal_seeds.tsv "$D/normal_${direction}_${stage}.tsv" 5 0 > "$D/normal_${direction}_${stage}.log" 2>&1
 done
done
