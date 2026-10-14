#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
output=${1:?provide a new evidence directory}
mkdir -p "$output" build-seeded-events
cmake -S . -B build-seeded-native -DCMAKE_BUILD_TYPE=Release -DLCBININT_BUILD_PYTHON=OFF -DLCBININT_BUILD_TESTS=OFF -DBUILD_TESTING=OFF -DLCBININT_ENABLE_JAX_FFI=OFF -DGSL_ROOT_DIR=/home/nunota/.miniconda3/envs/myenv
cmake --build build-seeded-native --target lcbinint_lightcurve --parallel 2
common=(-std=c++17 -O3 -march=native -fext-numeric-literals -Isrc -Iinclude)
native=(-Wl,--start-group build-seeded-native/liblcbinint_magnification.a build-seeded-native/liblcbinint_lightcurve.a -Wl,--end-group -L/home/nunota/.miniconda3/envs/myenv/lib -Wl,-rpath,/home/nunota/.miniconda3/envs/myenv/lib -lgsl -lgslcblas -lquadmath -fopenmp)
for name in export_point_seed_proposals bench_point_seeded_cold; do
 g++ "${common[@]}" "benchmarks/holonomic/$name.cpp" "${native[@]}" -o "build-seeded-events/$name"
done
for name in generate_seeded_fold_holdout bench_seeded_boundary_events audit_seeded_adaptive; do
 g++ "${common[@]}" "benchmarks/holonomic/$name.cpp" -lquadmath -o "build-seeded-events/$name"
done
cp evidence/holonomic/seeded_cold_provider_v2/inputs.tsv "$output/initial_inputs.tsv"
./build-seeded-events/generate_seeded_fold_holdout "$output/fold_inputs.tsv"
./build-seeded-events/generate_seeded_fold_holdout "$output/deeper_inputs.tsv" 1.13 .9999
for name in initial fold deeper; do
 ./build-seeded-events/export_point_seed_proposals "$output/${name}_inputs.tsv" "$output/${name}_seeds.tsv"
 rc=0
 ./build-seeded-events/bench_seeded_boundary_events "$output/${name}_seeds.tsv" "$output/${name}_audit.tsv" 1 || rc=$?
 echo "$name event audit exit=$rc (deeper currently has two failures)"
 if (( rc>1 )); then exit "$rc"; fi
 rc=0
 ./build-seeded-events/audit_seeded_adaptive "$output/${name}_seeds.tsv" "$output/${name}_adaptive.tsv" || rc=$?
 echo "$name adaptive audit exit=$rc (reference unresolved conditions are failures)"
 if (( rc>1 )); then exit "$rc"; fi
done
# All builds/audits finish before serial timed runs.
for name in initial fold; do
 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./build-seeded-events/bench_point_seeded_cold "$output/${name}_inputs.tsv" "$output/${name}_cold.tsv"
done
