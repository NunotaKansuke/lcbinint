#!/bin/bash
set -eu
python - <<'PREP'
from pathlib import Path
import shutil,subprocess
dest=Path('/tmp/p67_old_headers/lcbinint/magnification/holonomic')
shutil.copytree('src/lcbinint/magnification/holonomic',dest,dirs_exist_ok=True)
(dest/'adaptive_radial.hpp').write_text(subprocess.check_output(['git','show','6d533a6:src/lcbinint/magnification/holonomic/adaptive_radial.hpp'],text=True))
PREP
flags=(-O3 -DHOLO_ADAPTIVE_FOLD_QUARTIC_SEED -DNDEBUG -std=gnu++17 -fext-numeric-literals -I/tmp/p67_old_headers -Isrc -DHOLO_D14_NATIVE_RESIDUAL_SCREEN -DHOLO_D14_DIRECT_CONVOLUTION -DHOLO_ADAPTIVE_STATIONARY_ENDPOINT -DHOLO_D14_WARM_NOISE_HANDOFF -DHOLO_D14_SCALAR_CONSTANTS -DHOLO_ARC_COMPACT_STORAGE -DHOLO_ADAPTIVE_SPARSE_ANCHOR -DHOLO_ADAPTIVE_SAMPLE_GROWTH -DHOLO_MV_RATIONAL_MIDPOINT -DHOLO_QUARTIC_WARM_SOLVE_TOL=1e-11)
c++ "${flags[@]}" -march=native -funroll-loops -ffp-contract=fast -fno-math-errno tests/holonomic_cpp/test_adaptive_radial.cpp -o /tmp/p67_old_unit -lquadmath
taskset -c 0 /tmp/p67_old_unit > evidence/holonomic/gradient_nodes_phase67/old_unit.log
