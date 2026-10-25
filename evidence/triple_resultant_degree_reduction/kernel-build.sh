#!/bin/bash
set -euo pipefail
# Run from experiments/algebraic_boundary.
mkdir -p evidence/triple_resultant_degree_reduction/compiler-tmp
export TMPDIR="$PWD/evidence/triple_resultant_degree_reduction/compiler-tmp"
/usr/bin/c++ -O3 -DNDEBUG -DSTAGE_SEED_COUNTERS -std=c++17 -fPIC -shared -fno-fast-math -ffp-contract=off \
 -Isrc -I/home/nunota/.miniconda3/envs/myenv/lib/python3.10/site-packages/VBMicrolensing/lib \
 evidence/triple_resultant_degree_reduction/kernel.cpp \
 build-triple-native/CMakeFiles/lcbinint_lightcurve.dir/src/lcbinint/model/triple_lens_geometry.cpp.o \
 build-triple-native/liblcbinint_magnification.a \
 /home/nunota/.miniconda3/envs/myenv/lib/python3.10/site-packages/VBMicrolensing/VBMicrolensing.so \
 -o evidence/triple_resultant_degree_reduction/kernel.so -lquadmath -pthread
