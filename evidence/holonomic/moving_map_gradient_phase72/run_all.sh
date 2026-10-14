#!/usr/bin/env bash
set -euo pipefail
root="$(git rev-parse --show-toplevel)"
cd "$root"
here="$root/evidence/holonomic/moving_map_gradient_phase72"
cmake --build build-holonomic-m7 --target \
  test_adaptive_radial test_quartic_local_bracket test_quartic_sturm test_root_pair -j2 \
  > "$here/build_tests.log" 2>&1
if [[ "${REBUILD_RAND_REFERENCE:-0}" == "1" ]]; then
  bash "$here/rand035_reference/run_reference.sh" \
    > "$here/rand035_reference/run.log" 2>&1
fi
python3 "$here/rand035_reference/summarize.py" \
  > "$here/rand035_reference/summarize.log" 2>&1
bash "$here/run_reference_fd.sh"
bash "$here/run_shadow.sh" > "$here/shadow_run.log" 2>&1
python3 "$here/validate_summary.py"
ctest --test-dir build-holonomic-m7 --output-on-failure \
  -R '^(holonomic_adaptive_radial|holonomic_quartic_local_bracket|holonomic_quartic_sturm|holonomic_root_pair)$' \
  > "$here/relevant_ctest.log" 2>&1
find "$here" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum \
  > "$here/SHA256SUMS"
cat "$here/shadow_run.log"
cat "$here/relevant_ctest.log"
