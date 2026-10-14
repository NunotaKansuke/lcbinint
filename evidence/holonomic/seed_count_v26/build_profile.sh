#!/usr/bin/env bash
set -eu
D=evidence/holonomic/seed_count_v26
for stage in before after; do
 EXTRA=
 if [ "$stage" = before ]; then EXTRA=-I$D/before_include; fi
 g++ $EXTRA -DHOLO_FRONTIER_COST_PROFILE -DHOLO_QUARTIC_WORK_PROBE -std=c++17 -O3 -march=native -fext-numeric-literals -I/rogue1_8/nunota/lcbinint/src -Isrc -I/rogue1_8/nunota/lcbinint/include benchmarks/holonomic/bench_seeded_integration_profile.cpp "build-seeded-events/seed_${stage}_v26.o" -Wl,--start-group build-standard-current/liblcbinint_magnification.a build-standard-current/liblcbinint_lightcurve.a -Wl,--end-group -L/home/nunota/.miniconda3/envs/myenv/lib -Wl,-rpath,/home/nunota/.miniconda3/envs/myenv/lib -lgsl -lgslcblas -lquadmath -fopenmp -o "build-seeded-events/profile_seed_${stage}_v26"
done
