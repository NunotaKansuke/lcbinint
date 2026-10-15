#!/usr/bin/env bash
set -euo pipefail
grad_evidence="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
grad_root="$(cd -- "$grad_evidence/../../.." && pwd)"
cd "$grad_root"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export TMPDIR="$grad_root/build-algebraic-boundary/gradient-probe/scratch"
mkdir -p "$TMPDIR"
if [[ "${1:-}" == --references ]]; then python3 "$grad_evidence/reference.py"; fi
for prep in overlay moving direct allmotion post posth; do python3 "$grad_evidence/prepare_${prep}.py"; done
for grad_variant in api allmotion allmotion_post2 allmotion_posth; do
 g++ -std=gnu++17 -O2 -DNDEBUG -DONLY_VALUE_MESH -fext-numeric-literals \
  -I"build-algebraic-boundary/gradient-probe/${grad_variant}-include" -Isrc -Iinclude \
  -Ithird_party/skowron_gould -I/rogue1_8/nunota/local/gsl/include \
  "$grad_evidence/probe.cpp" -Wl,--start-group \
  build-algebraic-boundary/python-api/liblcbinint_magnification.a \
  build-algebraic-boundary/python-api/liblcbinint_lightcurve.a -Wl,--end-group \
  -L/rogue1_8/nunota/local/gsl/lib -Wl,-rpath,/rogue1_8/nunota/local/gsl/lib \
  -lgsl -lgslcblas -lquadmath \
  -o "build-algebraic-boundary/gradient-probe/probe_${grad_variant}_reproduce"
 for grad_tol in 1e-4 1e-5 1e-6; do
  grad_label="$grad_variant"; [[ "$grad_variant" != api ]] || grad_label=baseline
  LCBININT_PROBE_MU_TOL="$grad_tol" "build-algebraic-boundary/gradient-probe/probe_${grad_variant}_reproduce" \
   "$grad_evidence/inputs.tsv" "$grad_evidence/${grad_label}_${grad_tol}.tsv"
 done
 done
python3 "$grad_evidence/analyze.py"
