#!/usr/bin/env bash
set -euo pipefail
root="$(git rev-parse --show-toplevel)"
here="$root/evidence/holonomic/gradient_radial_split_phase72_rand035_reference"
input="$root/evidence/holonomic/adaptive_radial_20260911/reference_cases.tsv"
bash "$here/run_reference.sh" "$input" "$here/reference.tsv" all
bash "$here/run_reference.sh" "$input" "$here/reference_x_high.tsv" x-high
python3 "$here/summarize_reference.py"
ctest --test-dir "$root/build-holonomic-m7" \
  -R 'holonomic_(quartic_sturm|quartic_local_bracket|adaptive_radial)' \
  --output-on-failure > "$here/ctest.log"
