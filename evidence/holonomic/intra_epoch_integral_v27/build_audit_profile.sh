#!/usr/bin/env bash
set -eu
export TMPDIR="$PWD/build-seeded-events"
for stage in audit profile; do
 EXTRA=-DHOLO_ADAPTIVE_VALUE_BODY_AUDIT
 if [ "$stage" = profile ]; then EXTRA="-DHOLO_FRONTIER_COST_PROFILE -DHOLO_QUARTIC_WORK_PROBE"; fi
 SOURCE=benchmarks/holonomic/bench_seeded_radial_full.cpp
 if [ "$stage" = profile ]; then SOURCE=benchmarks/holonomic/bench_seeded_integration_profile.cpp; fi
 g++ $EXTRA -std=c++17 -O3 -march=native -fext-numeric-literals -I/rogue1_8/nunota/lcbinint/src -Isrc -I/rogue1_8/nunota/lcbinint/include "$SOURCE" build-seeded-events/seed_after_v26.o -Wl,--start-group build-standard-current/liblcbinint_magnification.a build-standard-current/liblcbinint_lightcurve.a -Wl,--end-group -L/home/nunota/.miniconda3/envs/myenv/lib -Wl,-rpath,/home/nunota/.miniconda3/envs/myenv/lib -lgsl -lgslcblas -lquadmath -fopenmp -o build-seeded-events/bench_integral_${stage}_v27
done
