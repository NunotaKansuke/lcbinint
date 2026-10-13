#!/usr/bin/env bash
set -euo pipefail
export TMPDIR="$PWD/build-seeded-events"
 g++ -std=c++17 -O3 -DNDEBUG -march=native -fext-numeric-literals \
  -Isrc/lcbinint/magnification/holonomic -I/rogue1_8/nunota/lcbinint/src -Isrc -I/rogue1_8/nunota/lcbinint/include \
  evidence/holonomic/accuracy_three_v44/tolerance_probe.cpp build-seeded-events/seed_indexed_v30.o \
  -Wl,--start-group build-standard-current/liblcbinint_magnification.a \
  build-standard-current/liblcbinint_lightcurve.a -Wl,--end-group \
  -L/home/nunota/.miniconda3/envs/myenv/lib -Wl,-rpath,/home/nunota/.miniconda3/envs/myenv/lib \
  -lgsl -lgslcblas -lquadmath -fopenmp -o build-seeded-events/probe_accuracy_v44
