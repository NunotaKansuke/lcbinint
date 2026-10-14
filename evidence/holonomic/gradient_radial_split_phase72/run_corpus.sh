#!/bin/bash
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "usage: $0 reference_cases.tsv output.tsv" >&2
  exit 2
fi
flags=(-O3 -DHOLO_ADAPTIVE_FOLD_QUARTIC_SEED -DNDEBUG -std=gnu++17
  -fext-numeric-literals -Isrc -DHOLO_D14_NATIVE_RESIDUAL_SCREEN
  -DHOLO_D14_DIRECT_CONVOLUTION -DHOLO_ADAPTIVE_STATIONARY_ENDPOINT
  -DHOLO_D14_WARM_NOISE_HANDOFF -DHOLO_D14_SCALAR_CONSTANTS
  -DHOLO_ARC_COMPACT_STORAGE -DHOLO_ADAPTIVE_SPARSE_ANCHOR
  -DHOLO_ADAPTIVE_SAMPLE_GROWTH -DHOLO_MV_RATIONAL_MIDPOINT
  -DHOLO_QUARTIC_WARM_SOLVE_TOL=1e-11)
c++ "${flags[@]}" -march=native -funroll-loops -ffp-contract=fast -fno-math-errno evidence/holonomic/gradient_radial_split_phase72/corpus.cpp -o /tmp/gradient_phase72_corpus -lquadmath
taskset -c 0 /tmp/gradient_phase72_corpus "$1" > "$2"
