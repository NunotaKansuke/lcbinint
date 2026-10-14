#!/usr/bin/env bash
set -eu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
D=evidence/holonomic/integration_push_v29
for input in initial fold deeper; do
 for stage in after; do
  build-seeded-events/bench_push_${stage}_v29 evidence/holonomic/caustic_seeded_connection_v4/${input}_minimal_seeds.tsv "$D/${stage}_${input}.tsv" 1 0 > "$D/${stage}_${input}.stdout" 2> "$D/${stage}_${input}.log"
 done
done
before_runs=0; after_runs=0
for stage in before after after before; do
 tag=matched
 if [ "$stage" = before ]; then before_runs=$((before_runs+1)); count=$before_runs; else after_runs=$((after_runs+1)); count=$after_runs; fi
 if [ "$count" -gt 1 ]; then tag=reverse; fi
 build-seeded-events/bench_push_${stage}_v29 evidence/holonomic/caustic_seeded_connection_v4/initial_minimal_seeds.tsv "$D/${stage}_${tag}.tsv" 5 0 > "$D/${stage}_${tag}.stdout" 2> "$D/${stage}_${tag}.log"
done
build-seeded-events/bench_push_profile_v29 evidence/holonomic/caustic_seeded_connection_v4/initial_minimal_seeds.tsv "$D/profile_after.tsv" 3 0 > "$D/profile_after.stdout" 2> "$D/profile_after.log"

build-seeded-events/bench_push_profile_before_v29 evidence/holonomic/caustic_seeded_connection_v4/initial_minimal_seeds.tsv "$D/profile_before.tsv" 3 0 > "$D/profile_before.stdout" 2> "$D/profile_before.log"
