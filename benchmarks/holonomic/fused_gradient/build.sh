#!/usr/bin/env bash
set -euo pipefail
export TMPDIR="$PWD/build-seeded-events"
mkdir -p "$TMPDIR/fused-gradient"
bash benchmarks/holonomic/build_seeded_image_seeds.sh
COMMON=(-std=c++17 -O3 -DNDEBUG -march=native -fext-numeric-literals)
INCLUDES=(-I/rogue1_8/nunota/lcbinint/src -Isrc -I/rogue1_8/nunota/lcbinint/include)
BACKEND=(build-seeded-events/seed_indexed_v30.o -Wl,--start-group
  build-standard-current/liblcbinint_magnification.a build-standard-current/liblcbinint_lightcurve.a
  -Wl,--end-group -L/home/nunota/.miniconda3/envs/myenv/lib
  -Wl,-rpath,/home/nunota/.miniconda3/envs/myenv/lib -lgsl -lgslcblas -lquadmath -fopenmp)
g++ "${COMMON[@]}" "${INCLUDES[@]}" benchmarks/holonomic/fused_gradient/bench.cpp \
  "${BACKEND[@]}" -o build-seeded-events/fused-gradient/baseline
python benchmarks/holonomic/fused_gradient/build_material_overlay.py
g++ "${COMMON[@]}" -DHOLO_FUSED_GRADIENT_EXPERIMENT \
  -Ibuild-seeded-events/fused-gradient/include -Ibenchmarks/holonomic/fused_gradient \
  "${INCLUDES[@]}" benchmarks/holonomic/fused_gradient/bench.cpp \
  "${BACKEND[@]}" -o build-seeded-events/fused-gradient/fused

# Optional compiled API: its private namespace can coexist with the value kernel.
python benchmarks/holonomic/fused_gradient/build_api_overlay.py
g++ "${COMMON[@]}" -fPIC -Ibuild-seeded-events/fused-gradient/api-include "${INCLUDES[@]}" \
  -c src/lcbinint/magnification/holonomic/fused_gradient.cpp \
  -o build-seeded-events/fused-gradient/fused_gradient_api.o
ar rcs build-seeded-events/fused-gradient/liblcbinint_fused_gradient.a \
  build-seeded-events/fused-gradient/fused_gradient_api.o
g++ "${COMMON[@]}" -DHOLO_COMPILED_FUSED_API "${INCLUDES[@]}" \
  benchmarks/holonomic/fused_gradient/bench.cpp \
  build-seeded-events/fused-gradient/liblcbinint_fused_gradient.a "${BACKEND[@]}" \
  -o build-seeded-events/fused-gradient/api_bench
g++ "${COMMON[@]}" "${INCLUDES[@]}" benchmarks/holonomic/fused_gradient/api_check.cpp \
  build-seeded-events/fused-gradient/liblcbinint_fused_gradient.a "${BACKEND[@]}" \
  -o build-seeded-events/fused-gradient/api_check
g++ "${COMMON[@]}" -Ibuild-seeded-events/fused-gradient/include \
  -Ibenchmarks/holonomic/fused_gradient "${INCLUDES[@]}" \
  benchmarks/holonomic/fused_gradient/batch_check.cpp -lquadmath \
  -o build-seeded-events/fused-gradient/batch_check
mkdir -p build-seeded-events/fused-gradient/python
PYBIND_FLAGS=($(python -m pybind11 --includes))
PYTHON_EXTENSION=$(python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
g++ "${COMMON[@]}" -shared -fPIC "${PYBIND_FLAGS[@]}" "${INCLUDES[@]}" \
  python/bind_holonomic_fused.cpp \
  build-seeded-events/fused-gradient/liblcbinint_fused_gradient.a "${BACKEND[@]}" \
  -o "build-seeded-events/fused-gradient/python/_lcbinint_holonomic_fused${PYTHON_EXTENSION}"
