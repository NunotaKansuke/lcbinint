#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$PWD/build-seeded-events"
mkdir -p "$TMPDIR"
OUT_DIR="${1:-evidence/holonomic/support_accuracy_20261014/reproduce}"
mkdir -p "$OUT_DIR"
if [ -e "$OUT_DIR/baseline_standard.tsv" ] || [ -e "$OUT_DIR/final_repeat.tsv" ]; then
  echo "Select a fresh output directory; existing benchmark rows are preserved." >&2
  exit 1
fi
sha256sum -c evidence/holonomic/support_accuracy_20261014/source_and_backend_sha256.txt
bash benchmarks/holonomic/build_seeded_paper_qrho.sh
bash benchmarks/holonomic/check_seed_support_local_scale.sh >"$OUT_DIR/analytic_check.txt"
build-seeded-events/bench_seeded_paper_qrho_v45 \
  evidence/holonomic/seeded_paper_qrho_20261014/input_snapshot.tsv \
  "$OUT_DIR/baseline_standard.tsv" 3 0 0 2>"$OUT_DIR/baseline_standard.log"
build-seeded-events/bench_seeded_paper_qrho \
  evidence/holonomic/seeded_paper_qrho_20261014/input_snapshot.tsv \
  "$OUT_DIR/final_repeat.tsv" 3 0 0 2>"$OUT_DIR/final_repeat.log"
