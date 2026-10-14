#!/bin/bash
set -eu
python - <<'PREP'
import shutil
from pathlib import Path
dest=Path('/tmp/p65_initial3_headers/lcbinint/magnification/holonomic')
shutil.copytree('src/lcbinint/magnification/holonomic',dest,dirs_exist_ok=True)
shutil.copyfile('evidence/holonomic/adaptive_accuracy_phase65/overlay_initial3/lcbinint/magnification/holonomic/adaptive_radial.hpp',dest/'adaptive_radial.hpp')
PREP
flags=(-O3 -DHOLO_ADAPTIVE_FOLD_QUARTIC_SEED -DNDEBUG -std=gnu++17 -fext-numeric-literals -I/tmp/p65_initial3_headers -Isrc -DHOLO_D14_NATIVE_RESIDUAL_SCREEN -DHOLO_D14_DIRECT_CONVOLUTION -DHOLO_ADAPTIVE_STATIONARY_ENDPOINT -DHOLO_D14_WARM_NOISE_HANDOFF -DHOLO_D14_SCALAR_CONSTANTS -DHOLO_ARC_COMPACT_STORAGE -DHOLO_ADAPTIVE_SPARSE_ANCHOR -DHOLO_ADAPTIVE_SAMPLE_GROWTH -DHOLO_MV_RATIONAL_MIDPOINT -DHOLO_QUARTIC_WARM_SOLVE_TOL=1e-11)
c++ "${flags[@]}" -march=native -funroll-loops -ffp-contract=fast -fno-math-errno evidence/holonomic/adaptive_accuracy_phase65/ab.cpp -o /tmp/p65_initial3 -lquadmath
