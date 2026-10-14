#!/usr/bin/env bash
set -eu
export TMPDIR="$PWD/build-seeded-events"
# Run from this experimental checkout. Same master seed probes/positions;
# pinned support-only scan optimization, local frontier and Sturm headers.
D=evidence/holonomic/seed_count_v26
OBJ=build-seeded-events/seed_after_v26.o
if [ ! -f "$OBJ" ] || [ "$D/seed_after.cpp" -nt "$OBJ" ]; then
 g++ -O3 -DNDEBUG -march=native -std=gnu++17 -fPIC -I/rogue1_8/nunota/lcbinint/include -I/rogue1_8/nunota/lcbinint/src -I/rogue1_8/nunota/lcbinint/third_party/skowron_gould -isystem /home/nunota/.miniconda3/envs/myenv/include -c "$D/seed_after.cpp" -o "$OBJ"
fi
g++ -std=c++17 -O3 -march=native -fext-numeric-literals -I/rogue1_8/nunota/lcbinint/src -Isrc -I/rogue1_8/nunota/lcbinint/include benchmarks/holonomic/bench_seeded_radial_full.cpp "$OBJ" -Wl,--start-group build-standard-current/liblcbinint_magnification.a build-standard-current/liblcbinint_lightcurve.a -Wl,--end-group -L/home/nunota/.miniconda3/envs/myenv/lib -Wl,-rpath,/home/nunota/.miniconda3/envs/myenv/lib -lgsl -lgslcblas -lquadmath -fopenmp -o build-seeded-events/bench_seeded_radial_master
