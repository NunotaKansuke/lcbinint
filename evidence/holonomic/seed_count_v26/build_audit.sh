#!/usr/bin/env bash
set -eu
for stage in before after; do
 g++ -std=c++17 -O3 -march=native -I/rogue1_8/nunota/lcbinint/src -I/rogue1_8/nunota/lcbinint/include evidence/holonomic/seed_count_v26/audit_seed.cpp build-seeded-events/seed_${stage}_v26.o -Wl,--start-group build-standard-current/liblcbinint_magnification.a build-standard-current/liblcbinint_lightcurve.a -Wl,--end-group -L/home/nunota/.miniconda3/envs/myenv/lib -Wl,-rpath,/home/nunota/.miniconda3/envs/myenv/lib -lgsl -lgslcblas -lquadmath -fopenmp -o build-seeded-events/audit_seed_${stage}_v26
done
