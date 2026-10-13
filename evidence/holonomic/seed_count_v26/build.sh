#!/usr/bin/env bash
set -eu
D=evidence/holonomic/seed_count_v26
for stage in before after; do
 g++ -O3 -DNDEBUG -march=native -std=gnu++17 -fPIC -I/rogue1_8/nunota/lcbinint/include -I/rogue1_8/nunota/lcbinint/src -I/rogue1_8/nunota/lcbinint/third_party/skowron_gould -isystem /home/nunota/.miniconda3/envs/myenv/include -c "$D/seed_${stage}.cpp" -o "build-seeded-events/seed_${stage}_v26.o"
 EXTRA=
 if [ "$stage" = before ]; then EXTRA=-I$D/before_include; fi
 g++ $EXTRA -std=c++17 -O3 -march=native -fext-numeric-literals -I/rogue1_8/nunota/lcbinint/src -Isrc -I/rogue1_8/nunota/lcbinint/include benchmarks/holonomic/bench_seeded_radial_full.cpp "build-seeded-events/seed_${stage}_v26.o" -Wl,--start-group build-standard-current/liblcbinint_magnification.a build-standard-current/liblcbinint_lightcurve.a -Wl,--end-group -L/home/nunota/.miniconda3/envs/myenv/lib -Wl,-rpath,/home/nunota/.miniconda3/envs/myenv/lib -lgsl -lgslcblas -lquadmath -fopenmp -o "build-seeded-events/bench_seed_${stage}_v26"
done
