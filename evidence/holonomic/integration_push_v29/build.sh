#!/usr/bin/env bash
set -eu
export TMPDIR="$PWD/build-seeded-events"
D=evidence/holonomic/integration_push_v29
for stage in before profile_before after profile; do
 EXTRA=
 if [ "$stage" = profile ] || [ "$stage" = profile_before ]; then EXTRA="-DHOLO_FRONTIER_COST_PROFILE -DHOLO_QUARTIC_WORK_PROBE"; fi
 if [ "$stage" = before ] || [ "$stage" = profile_before ]; then EXTRA="$EXTRA -I$D/before_include"; fi
 SOURCE=benchmarks/holonomic/bench_seeded_radial_full.cpp
 if [ "$stage" = profile ]; then SOURCE=benchmarks/holonomic/bench_seeded_integration_profile.cpp; fi
 if [ "$stage" = profile_before ]; then SOURCE=benchmarks/holonomic/bench_seeded_integration_profile.cpp; fi
 g++ $EXTRA -std=c++17 -O3 -march=native -fext-numeric-literals -I/rogue1_8/nunota/lcbinint/src -Isrc -I/rogue1_8/nunota/lcbinint/include "$SOURCE" build-seeded-events/seed_after_v26.o -Wl,--start-group build-standard-current/liblcbinint_magnification.a build-standard-current/liblcbinint_lightcurve.a -Wl,--end-group -L/home/nunota/.miniconda3/envs/myenv/lib -Wl,-rpath,/home/nunota/.miniconda3/envs/myenv/lib -lgsl -lgslcblas -lquadmath -fopenmp -o build-seeded-events/bench_push_${stage}_v29
done
