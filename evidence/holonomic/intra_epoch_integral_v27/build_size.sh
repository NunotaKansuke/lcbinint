#!/usr/bin/env bash
set -eu
export TMPDIR="$PWD/build-seeded-events"
D=evidence/holonomic/intra_epoch_integral_v27
for stage in before after; do
 EXTRA=
 if [ "$stage" = before ]; then EXTRA=-I$D/before_include; fi
 g++ $EXTRA -O3 -std=c++17 -fext-numeric-literals -Isrc benchmarks/holonomic/inspect_adaptive_storage.cpp -lquadmath -o build-seeded-events/inspect_storage_${stage}_v27
 build-seeded-events/inspect_storage_${stage}_v27 > "$D/storage_${stage}.txt"
done
