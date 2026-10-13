#!/usr/bin/env bash
set -euo pipefail
cd /rogue1_8/nunota/lcbinint/experiments/seeded-boundary-events
export OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$PWD/build-seeded-events"
exec build-seeded-events/bench_seeded_paper_qrho_v45 \
  evidence/holonomic/seeded_paper_qrho_20261014/input_snapshot.tsv \
  evidence/holonomic/seeded_paper_qrho_20261014/raw_results.tsv 3 0 0
