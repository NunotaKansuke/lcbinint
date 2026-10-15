#!/bin/bash
set -euo pipefail
export TMPDIR="$PWD/build-algebraic-boundary/fused-gradient/scratch" PYTHONDONTWRITEBYTECODE=1
read -ra flags <<< "$(python -m pybind11 --includes)"
extension="$(python -c 'import sysconfig;print(sysconfig.get_config_var("EXT_SUFFIX"))')"
g++ -std=gnu++17 -O2 -DNDEBUG -fext-numeric-literals -fPIC -shared "${flags[@]}" -Isrc -Iinclude -Ithird_party/skowron_gould -I/rogue1_8/nunota/local/gsl/include python/bind_holonomic_fused.cpp build-algebraic-boundary/fused-gradient/liblcbinint_fused_gradient.a -Wl,--start-group build-algebraic-boundary/python-api/liblcbinint_magnification.a build-algebraic-boundary/python-api/liblcbinint_lightcurve.a -Wl,--end-group -L/rogue1_8/nunota/local/gsl/lib -Wl,-rpath,/rogue1_8/nunota/local/gsl/lib -lgsl -lgslcblas -lquadmath -o "build-algebraic-boundary/fused-gradient/python/_lcbinint_holonomic_fused$extension"
