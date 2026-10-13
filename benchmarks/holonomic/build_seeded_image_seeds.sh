#!/usr/bin/env bash
set -euo pipefail
export TMPDIR="$PWD/build-seeded-events"
mkdir -p "$TMPDIR"
SOURCE=evidence/holonomic/low_magnification_fixed_cost_v30/seed_indexed.cpp
OBJECT=build-seeded-events/seed_indexed_v30.o
if [ ! -f "$OBJECT" ] || [ "$SOURCE" -nt "$OBJECT" ]; then
  g++ -O3 -DNDEBUG -march=native -std=gnu++17 -fPIC \
    -I/rogue1_8/nunota/lcbinint/include -I/rogue1_8/nunota/lcbinint/src \
    -I/rogue1_8/nunota/lcbinint/third_party/skowron_gould \
    -isystem /home/nunota/.miniconda3/envs/myenv/include \
    -c "$SOURCE" -o "$OBJECT"
fi
