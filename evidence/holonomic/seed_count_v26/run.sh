#!/usr/bin/env bash
set -eu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
D=evidence/holonomic/seed_count_v26
build-seeded-events/audit_count_v26 > "$D/count_audit.log"
for input in initial fold deeper; do
 for stage in before after; do
  build-seeded-events/audit_seed_${stage}_v26 evidence/holonomic/caustic_seeded_connection_v4/${input}_minimal_seeds.tsv > "$D/seed_${stage}_${input}.txt"
  build-seeded-events/bench_seed_${stage}_v26 evidence/holonomic/caustic_seeded_connection_v4/${input}_minimal_seeds.tsv "$D/${stage}_${input}.tsv" 1 0 > "$D/${stage}_${input}.stdout" 2> "$D/${stage}_${input}.log"
 done
 cmp "$D/seed_before_${input}.txt" "$D/seed_after_${input}.txt"
done
for stage in before after after before; do
 tag=matched
 if [ -f "$D/${stage}_matched.tsv" ]; then tag=reverse; fi
 build-seeded-events/bench_seed_${stage}_v26 evidence/holonomic/caustic_seeded_connection_v4/initial_minimal_seeds.tsv "$D/${stage}_${tag}.tsv" 5 0 > "$D/${stage}_${tag}.stdout" 2> "$D/${stage}_${tag}.log"
done
