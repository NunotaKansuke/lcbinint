#!/usr/bin/env bash
set -euo pipefail
export TMPDIR="$PWD/build-seeded-events"
mkdir -p "$TMPDIR"
bash benchmarks/holonomic/build_seeded_image_seeds.sh
g++ -std=c++17 -O3 \
  -Ibuild-seeded-events/support-include \
  -I/rogue1_8/nunota/lcbinint/src -I/rogue1_8/nunota/lcbinint/include \
  benchmarks/holonomic/check_seed_support_local_scale.cpp \
  -Wl,--start-group build-standard-current/liblcbinint_magnification.a \
  build-standard-current/liblcbinint_lightcurve.a -Wl,--end-group \
  -L/home/nunota/.miniconda3/envs/myenv/lib \
  -Wl,-rpath,/home/nunota/.miniconda3/envs/myenv/lib \
  -fopenmp -lgsl -lgslcblas -lquadmath \
  -o build-seeded-events/check_seed_support_local_scale
exec build-seeded-events/check_seed_support_local_scale
