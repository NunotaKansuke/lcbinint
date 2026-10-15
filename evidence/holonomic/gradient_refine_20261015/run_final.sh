#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export TMPDIR="$PWD/build-algebraic-boundary/gradient-probe/scratch" PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1
bash benchmarks/holonomic/fused_gradient/build_geometry.sh
includes=(-Ibuild-algebraic-boundary/gradient-probe/geometry-include -Isrc -Iinclude -Ithird_party/skowron_gould -I/rogue1_8/nunota/local/gsl/include)
backend=(-Wl,--start-group build-algebraic-boundary/python-api/liblcbinint_magnification.a build-algebraic-boundary/python-api/liblcbinint_lightcurve.a -Wl,--end-group -L/rogue1_8/nunota/local/gsl/lib -Wl,-rpath,/rogue1_8/nunota/local/gsl/lib -lgsl -lgslcblas -lquadmath)
g++ -std=gnu++17 -O2 -DNDEBUG -fext-numeric-literals -DONLY_VALUE_MESH "${includes[@]}" evidence/holonomic/gradient_screen_20261015/probe.cpp "${backend[@]}" -o build-algebraic-boundary/gradient-probe/probe_refine_final
for name in check_bounds radius_check; do
 g++ -std=gnu++17 -O2 -fext-numeric-literals "${includes[@]}" "evidence/holonomic/gradient_refine_20261015/$name.cpp" -lquadmath -o "build-algebraic-boundary/gradient-probe/refine_$name"
 target="$name.txt"; if [[ "$name" == check_bounds ]]; then target=bounds_check.txt; fi
 "build-algebraic-boundary/gradient-probe/refine_$name" > "evidence/holonomic/gradient_refine_20261015/$target"
done
python evidence/holonomic/gradient_refine_20261015/evaluate.py screen
python evidence/holonomic/gradient_refine_20261015/evaluate.py screen_off
python evidence/holonomic/gradient_refine_20261015/matrix.py
python evidence/holonomic/gradient_refine_20261015/profile.py
python evidence/holonomic/gradient_refine_20261015/analyze.py
