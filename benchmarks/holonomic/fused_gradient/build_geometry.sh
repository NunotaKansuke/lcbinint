#!/usr/bin/env bash
# Experimental build only; leaves the default optional-gradient module intact.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/build-algebraic-boundary/gradient-probe/scratch"
python3 benchmarks/holonomic/fused_gradient/build_geometry_overlay.py
includes=(-Ibuild-algebraic-boundary/gradient-probe/geometry-include -Isrc -Iinclude -Ithird_party/skowron_gould -I/rogue1_8/nunota/local/gsl/include)
backend=(-Wl,--start-group build-algebraic-boundary/python-api/liblcbinint_magnification.a build-algebraic-boundary/python-api/liblcbinint_lightcurve.a -Wl,--end-group -L/rogue1_8/nunota/local/gsl/lib -Wl,-rpath,/rogue1_8/nunota/local/gsl/lib -lgsl -lgslcblas -lquadmath)
common=(-std=gnu++17 -O2 -DNDEBUG -fext-numeric-literals)
g++ "${common[@]}" -DONLY_VALUE_MESH "${includes[@]}" evidence/holonomic/gradient_accuracy_20261015/probe.cpp "${backend[@]}" -o build-algebraic-boundary/gradient-probe/probe_geometry
mkdir -p build-algebraic-boundary/gradient-probe/geometry-python
g++ "${common[@]}" -fPIC "${includes[@]}" -c src/lcbinint/magnification/holonomic/fused_gradient.cpp -o build-algebraic-boundary/gradient-probe/geometry_api.o
read -ra pybind_flags <<< "$(python3 -m pybind11 --includes)"
extension="$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
g++ "${common[@]}" -fPIC -shared "${pybind_flags[@]}" -Isrc -Iinclude python/bind_holonomic_fused.cpp build-algebraic-boundary/gradient-probe/geometry_api.o "${backend[@]}" -o "build-algebraic-boundary/gradient-probe/geometry-python/_lcbinint_holonomic_fused$extension"
