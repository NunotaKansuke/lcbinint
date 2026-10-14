#!/usr/bin/env bash
set -euo pipefail
export TMPDIR="$PWD/build-seeded-events"
mkdir -p "$TMPDIR"
bash benchmarks/holonomic/build_seeded_image_seeds.sh

g++ -std=c++17 -O3 -DNDEBUG -march=native -fext-numeric-literals \
  -I/rogue1_8/nunota/lcbinint/src -Isrc \
  -I/rogue1_8/nunota/lcbinint/include \
  benchmarks/holonomic/bench_seeded_paper_qrho.cpp \
  build-seeded-events/seed_indexed_v30.o \
  -Wl,--start-group \
  build-standard-current/liblcbinint_magnification.a \
  build-standard-current/liblcbinint_lightcurve.a \
  -Wl,--end-group \
  -L/home/nunota/.miniconda3/envs/myenv/lib \
  -Wl,-rpath,/home/nunota/.miniconda3/envs/myenv/lib \
  -lgsl -lgslcblas -lquadmath -fopenmp \
  -o build-seeded-events/bench_seeded_paper_qrho

sha256sum build-seeded-events/bench_seeded_paper_qrho
