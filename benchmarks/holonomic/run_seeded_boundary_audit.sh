#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$project_root"
result_dir=${1:-evidence/holonomic/seeded_boundary_events_v1}
mkdir -p build-seeded-events "$result_dir"
seed_file=evidence/holonomic/seeded_boundary_events_v1/minimal_seeds.tsv
compiler=${CXX:-g++}
for target in bench_seeded_boundary_events audit_seeded_adaptive; do
  "$compiler" -std=c++17 -O3 -march=native -fext-numeric-literals -Isrc \
    "benchmarks/holonomic/$target.cpp" -lquadmath -o "build-seeded-events/$target"
done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
./build-seeded-events/bench_seeded_boundary_events "$seed_file" "$result_dir/rootpair_final.tsv" 7
# This strict audit returns 1 also for reference-unresolved cases. Keep that
# status visible; the summary separately reports candidate regressions.
set +e
./build-seeded-events/audit_seeded_adaptive "$seed_file" "$result_dir/adaptive_final.tsv"
audit_status=$?
set -e
printf '%s\n' "$audit_status" > "$result_dir/adaptive_exit_status.txt"
python benchmarks/holonomic/summarize_seeded_boundary_audit.py "$result_dir"
exit "$audit_status"
