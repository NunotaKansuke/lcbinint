#!/usr/bin/env bash
set -euo pipefail
export TMPDIR="$PWD/build-seeded-events"
bash benchmarks/holonomic/build_seeded_image_seeds.sh
D=evidence/holonomic/weighted_radial_v33
for stage in after; do
 extra=()
 if [ "$stage" = before ]; then extra=(-I"$D/before_include"); fi
 g++ "${extra[@]}" -std=c++17 -O3 -DNDEBUG -march=native -fext-numeric-literals \
  -Isrc/lcbinint/magnification/holonomic -I/rogue1_8/nunota/lcbinint/src -Isrc -I/rogue1_8/nunota/lcbinint/include \
  evidence/holonomic/weighted_radial_v33/diagnose.cpp build-seeded-events/seed_indexed_v30.o \
  -Wl,--start-group build-standard-current/liblcbinint_magnification.a \
  build-standard-current/liblcbinint_lightcurve.a -Wl,--end-group \
  -L/home/nunota/.miniconda3/envs/myenv/lib -Wl,-rpath,/home/nunota/.miniconda3/envs/myenv/lib \
  -lgsl -lgslcblas -lquadmath -fopenmp -o "build-seeded-events/bench_diagnose_${stage}_v33"
done
