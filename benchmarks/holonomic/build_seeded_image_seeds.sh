#!/usr/bin/env bash
set -euo pipefail
export TMPDIR="$PWD/build-seeded-events"
mkdir -p "$TMPDIR"
SOURCE=evidence/holonomic/low_magnification_fixed_cost_v30/seed_indexed.cpp
OBJECT=build-seeded-events/seed_indexed_v30.o
SUPPORT_HEADER=src/lcbinint/magnification/component_certificate.hpp
SUPPORT_INCLUDE=build-seeded-events/support-include
mkdir -p "$SUPPORT_INCLUDE/lcbinint/magnification"
SUPPORT_COPY="$SUPPORT_INCLUDE/lcbinint/magnification/component_certificate.hpp"
# Override only this header: the seed provider's remaining ABI headers must
# continue to match the existing backend archives.
if ! cmp -s "$SUPPORT_HEADER" "$SUPPORT_COPY"; then
  cp "$SUPPORT_HEADER" "$SUPPORT_COPY"
fi
if [ ! -f "$OBJECT" ] || [ "$SOURCE" -nt "$OBJECT" ] || [ "$SUPPORT_COPY" -nt "$OBJECT" ]; then
  g++ -O3 -DNDEBUG -march=native -std=gnu++17 -fPIC \
    -I"$SUPPORT_INCLUDE" -I/rogue1_8/nunota/lcbinint/include -I/rogue1_8/nunota/lcbinint/src \
    -I/rogue1_8/nunota/lcbinint/third_party/skowron_gould \
    -isystem /home/nunota/.miniconda3/envs/myenv/include \
    -c "$SOURCE" -o "$OBJECT"
fi
