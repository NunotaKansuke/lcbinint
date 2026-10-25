#!/bin/bash
set -euo pipefail
export TMPDIR="$PWD/evidence/triple_resultant_sparse/compiler-tmp"
/usr/bin/c++ -O3 -DNDEBUG -std=c++17 -fno-fast-math -ffp-contract=off -Isrc \
 evidence/triple_resultant_sparse/point.cpp \
 build-triple-native/CMakeFiles/lcbinint_lightcurve.dir/src/lcbinint/model/triple_lens_geometry.cpp.o \
 build-triple-native/liblcbinint_magnification.a \
 -o evidence/triple_resultant_sparse/point-final -lquadmath -pthread
