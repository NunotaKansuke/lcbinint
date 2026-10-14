#!/usr/bin/env bash
set -eu
export TMPDIR="$PWD/build-seeded-events"
D=evidence/holonomic/intra_epoch_integral_v27
for stage in before after; do
 EXTRA=
 if [ "$stage" = before ]; then EXTRA="-I$D/before_include -Isrc/lcbinint/magnification/holonomic"; fi
 g++ $EXTRA -std=c++17 -O3 -march=native -fext-numeric-literals -I/rogue1_8/nunota/lcbinint/src -Isrc -I/rogue1_8/nunota/lcbinint/include benchmarks/holonomic/bench_seeded_radial_full.cpp build-seeded-events/seed_after_v26.o -Wl,--start-group build-standard-current/liblcbinint_magnification.a build-standard-current/liblcbinint_lightcurve.a -Wl,--end-group -L/home/nunota/.miniconda3/envs/myenv/lib -Wl,-rpath,/home/nunota/.miniconda3/envs/myenv/lib -lgsl -lgslcblas -lquadmath -fopenmp -o build-seeded-events/bench_integral_${stage}_v27
done
