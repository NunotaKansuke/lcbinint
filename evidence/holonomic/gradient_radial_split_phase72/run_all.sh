#!/usr/bin/env bash
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
EVIDENCE="$ROOT/evidence/holonomic/gradient_radial_split_phase72"
INPUT="$ROOT/evidence/holonomic/adaptive_radial_20260911/reference_cases.tsv"

cmake --build build-holonomic-m7 --target \
  test_adaptive_radial check_adaptive_reference check_adaptive_jacobian -j2
build-holonomic-m7/test_adaptive_radial > "$EVIDENCE/adaptive_unit.log"
ctest --test-dir build-holonomic-m7 \
  -R 'holonomic_(adaptive_radial|quartic_sturm|quartic_local_bracket)' \
  --output-on-failure > "$EVIDENCE/ctest.log" 2>&1

bash "$EVIDENCE/run_panel_audit.sh" "$INPUT" "$EVIDENCE/panel_audit.tsv"
bash "$EVIDENCE/run_corpus.sh" "$INPUT" "$EVIDENCE/corpus.tsv"
taskset -c 0 build-holonomic-m7/check_adaptive_reference "$INPUT" \
  > "$EVIDENCE/value_reference_check.tsv"
taskset -c 0 build-holonomic-m7/check_adaptive_jacobian \
  > "$EVIDENCE/gradient_reference_corpus.tsv"
python3 "$EVIDENCE/summarize.py" > "$EVIDENCE/summary_stdout.json"
