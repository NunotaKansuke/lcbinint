#!/usr/bin/env bash
set -euo pipefail
export TMPDIR="$PWD/build-seeded-events"
mkdir -p "$TMPDIR/fused-gradient"
bash benchmarks/holonomic/build_seeded_image_seeds.sh
COMMON=(-std=c++17 -O3 -DNDEBUG -march=native -fext-numeric-literals)
INCLUDES=(-I/rogue1_8/nunota/lcbinint/src -Isrc -I/rogue1_8/nunota/lcbinint/include)
BACKEND=(build-seeded-events/seed_indexed_v30.o -Wl,--start-group
  build-standard-current/liblcbinint_magnification.a build-standard-current/liblcbinint_lightcurve.a
  -Wl,--end-group -L/home/nunota/.miniconda3/envs/myenv/lib
  -Wl,-rpath,/home/nunota/.miniconda3/envs/myenv/lib -lgsl -lgslcblas -lquadmath -fopenmp)
g++ "${COMMON[@]}" "${INCLUDES[@]}" benchmarks/holonomic/fused_gradient/bench.cpp \
  "${BACKEND[@]}" -o build-seeded-events/fused-gradient/baseline
python benchmarks/holonomic/fused_gradient/build_material_overlay.py
g++ "${COMMON[@]}" -DHOLO_FUSED_GRADIENT_EXPERIMENT \
  -Ibuild-seeded-events/fused-gradient/include -Ibenchmarks/holonomic/fused_gradient \
  "${INCLUDES[@]}" benchmarks/holonomic/fused_gradient/bench.cpp \
  "${BACKEND[@]}" -o build-seeded-events/fused-gradient/fused
