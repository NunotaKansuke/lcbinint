#!/usr/bin/env bash
# Build the optional gradient API against this checkout's native value libraries.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
root="$PWD"
export LCBININT_FUSED_BUILD="$root/build-algebraic-boundary/fused-gradient"
export TMPDIR="$LCBININT_FUSED_BUILD/scratch" PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR" "$LCBININT_FUSED_BUILD/python"
python3 benchmarks/holonomic/fused_gradient/build_material_overlay.py
python3 benchmarks/holonomic/fused_gradient/build_api_overlay.py
common=(-std=gnu++17 -O2 -DNDEBUG -fext-numeric-literals -fPIC)
includes=(-Isrc -Iinclude -Ithird_party/skowron_gould -I/rogue1_8/nunota/local/gsl/include)
backend=(-Wl,--start-group build-algebraic-boundary/python-api/liblcbinint_magnification.a
 build-algebraic-boundary/python-api/liblcbinint_lightcurve.a -Wl,--end-group
 -L/rogue1_8/nunota/local/gsl/lib -Wl,-rpath,/rogue1_8/nunota/local/gsl/lib -lgsl -lgslcblas -lquadmath)
g++ "${common[@]}" -I"$LCBININT_FUSED_BUILD/api-include" "${includes[@]}" \
 -c src/lcbinint/magnification/holonomic/fused_gradient.cpp -o "$LCBININT_FUSED_BUILD/fused_gradient_api.o"
ar rcs "$LCBININT_FUSED_BUILD/liblcbinint_fused_gradient.a" "$LCBININT_FUSED_BUILD/fused_gradient_api.o"
read -ra pybind_flags <<< "$(python3 -m pybind11 --includes)"
extension="$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
g++ "${common[@]}" -shared "${pybind_flags[@]}" "${includes[@]}" python/bind_holonomic_fused.cpp \
 "$LCBININT_FUSED_BUILD/liblcbinint_fused_gradient.a" "${backend[@]}" \
 -o "$LCBININT_FUSED_BUILD/python/_lcbinint_holonomic_fused$extension"
g++ "${common[@]}" "${includes[@]}" benchmarks/holonomic/fused_gradient/api_check.cpp \
 "$LCBININT_FUSED_BUILD/liblcbinint_fused_gradient.a" "${backend[@]}" -o "$LCBININT_FUSED_BUILD/api_check"
