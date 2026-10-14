#!/usr/bin/env bash
set -eu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
D=evidence/holonomic/intra_epoch_integral_v27
for input in initial fold deeper; do
 for stage in after audit; do
  build-seeded-events/bench_integral_${stage}_v27 evidence/holonomic/caustic_seeded_connection_v4/${input}_minimal_seeds.tsv "$D/${stage}_${input}.tsv" 1 0 > "$D/${stage}_${input}.stdout" 2> "$D/${stage}_${input}.log"
 done
done
for stage in before after after before; do
 tag=matched
 if [ -f "$D/${stage}_matched.tsv" ]; then tag=reverse; fi
 build-seeded-events/bench_integral_${stage}_v27 evidence/holonomic/caustic_seeded_connection_v4/initial_minimal_seeds.tsv "$D/${stage}_${tag}.tsv" 5 0 > "$D/${stage}_${tag}.stdout" 2> "$D/${stage}_${tag}.log"
done
build-seeded-events/bench_integral_profile_v27 evidence/holonomic/caustic_seeded_connection_v4/initial_minimal_seeds.tsv "$D/profile_after.tsv" 3 0 > "$D/profile_after.stdout" 2> "$D/profile_after.log"
