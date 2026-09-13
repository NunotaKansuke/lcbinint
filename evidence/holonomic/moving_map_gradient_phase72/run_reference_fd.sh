#!/usr/bin/env bash
set -euo pipefail
root="$(git rev-parse --show-toplevel)"
here="$root/evidence/holonomic/moving_map_gradient_phase72"
input="$root/evidence/holonomic/adaptive_radial_20260911/reference_cases.tsv"
flags=(-O3 -DNDEBUG -std=gnu++17 -fext-numeric-literals -I"$root/src"
  -DHOLO_D14_NATIVE_RESIDUAL_SCREEN -DHOLO_D14_DIRECT_CONVOLUTION
  -DHOLO_ADAPTIVE_STATIONARY_ENDPOINT -DHOLO_D14_WARM_NOISE_HANDOFF
  -DHOLO_D14_SCALAR_CONSTANTS -DHOLO_ARC_COMPACT_STORAGE
  -DHOLO_ADAPTIVE_SPARSE_ANCHOR -DHOLO_ADAPTIVE_SAMPLE_GROWTH
  -DHOLO_MV_RATIONAL_MIDPOINT -DHOLO_QUARTIC_WARM_SOLVE_TOL=1e-11)
c++ "${flags[@]}" -march=native -funroll-loops -ffp-contract=fast \
  -fno-math-errno "$here/reference_fd.cpp" -o /tmp/phase72_moving_map_reference_fd -lquadmath
env -u HOLO_ADAPTIVE_EVENT_REUSE -u HOLO_MV_TRANSPORT_LEGACY \
  -u HOLO_HOLONOMIC_TRANSPORT -u HOLO_HOLONOMIC_TRANSPORT_LEGACY \
  taskset -c 0 /tmp/phase72_moving_map_reference_fd "$input" "$here/reference_fd.tsv"
