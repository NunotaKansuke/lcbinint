#!/usr/bin/env bash
set -euo pipefail
root="$(git rev-parse --show-toplevel)"
here="$root/evidence/holonomic/projective_fold_phase74"
input="$root/evidence/holonomic/adaptive_radial_20260911/reference_cases.tsv"
common=(-O3 -DNDEBUG -std=gnu++17 -fext-numeric-literals -I"$root/src"
  -DHOLO_D14_NATIVE_RESIDUAL_SCREEN -DHOLO_D14_DIRECT_CONVOLUTION
  -DHOLO_ADAPTIVE_STATIONARY_ENDPOINT -DHOLO_D14_WARM_NOISE_HANDOFF
  -DHOLO_D14_SCALAR_CONSTANTS -DHOLO_ARC_COMPACT_STORAGE
  -DHOLO_ADAPTIVE_SPARSE_ANCHOR -DHOLO_ADAPTIVE_SAMPLE_GROWTH
  -DHOLO_MV_RATIONAL_MIDPOINT -DHOLO_QUARTIC_WARM_SOLVE_TOL=1e-11
  -march=native -funroll-loops -ffp-contract=fast -fno-math-errno)
mkdir -p "$here"
{
  c++ "${common[@]}" "$here/projective_fold_probe.cpp" \
    -o /tmp/phase74_projective_fold_probe -lquadmath
  c++ "${common[@]}" -DHOLO_ADAPTIVE_FOLD_QUARTIC_SEED \
    "$here/adaptive_ab.cpp" -o /tmp/phase74_adaptive_baseline -lquadmath
  c++ "${common[@]}" -DHOLO_ADAPTIVE_FOLD_QUARTIC_SEED \
    -DHOLO_ADAPTIVE_PROJECTIVE_P4_FOLD_RESEARCH "$here/adaptive_ab.cpp" \
    -o /tmp/phase74_adaptive_projective -lquadmath
  c++ "${common[@]}" -DHOLO_ADAPTIVE_FOLD_QUARTIC_SEED \
    "$here/adaptive_corpus_ab.cpp" -o /tmp/phase74_corpus_baseline -lquadmath
  c++ "${common[@]}" -DHOLO_ADAPTIVE_FOLD_QUARTIC_SEED \
    -DHOLO_ADAPTIVE_PROJECTIVE_P4_FOLD_RESEARCH \
    "$here/adaptive_corpus_ab.cpp" -o /tmp/phase74_corpus_projective -lquadmath

  env -u HOLO_D14_EVENT_CONTRACT_ACCEPT -u HOLO_MV_TRANSPORT_LEGACY \
    -u HOLO_HOLONOMIC_TRANSPORT -u HOLO_HOLONOMIC_TRANSPORT_LEGACY \
    taskset -c 0 /tmp/phase74_projective_fold_probe "$input" \
      "$here/projective_fold_probe.tsv"
  env -u HOLO_D14_EVENT_CONTRACT_ACCEPT -u HOLO_MV_TRANSPORT_LEGACY \
    -u HOLO_HOLONOMIC_TRANSPORT -u HOLO_HOLONOMIC_TRANSPORT_LEGACY \
    taskset -c 0 /tmp/phase74_adaptive_baseline "$input" \
      "$here/adaptive_baseline.tsv" "$here/adaptive_baseline_panels.tsv" \
      "$here/adaptive_baseline_epoch.tsv"
  env -u HOLO_D14_EVENT_CONTRACT_ACCEPT -u HOLO_MV_TRANSPORT_LEGACY \
    -u HOLO_HOLONOMIC_TRANSPORT -u HOLO_HOLONOMIC_TRANSPORT_LEGACY \
    taskset -c 0 /tmp/phase74_adaptive_projective "$input" \
      "$here/adaptive_projective.tsv" "$here/adaptive_projective_panels.tsv" \
      "$here/adaptive_projective_epoch.tsv"
  env -u HOLO_D14_EVENT_CONTRACT_ACCEPT -u HOLO_MV_TRANSPORT_LEGACY \
    -u HOLO_HOLONOMIC_TRANSPORT -u HOLO_HOLONOMIC_TRANSPORT_LEGACY \
    taskset -c 0 /tmp/phase74_corpus_baseline "$input" \
      "$here/corpus_baseline.tsv"
  env -u HOLO_D14_EVENT_CONTRACT_ACCEPT -u HOLO_MV_TRANSPORT_LEGACY \
    -u HOLO_HOLONOMIC_TRANSPORT -u HOLO_HOLONOMIC_TRANSPORT_LEGACY \
    taskset -c 0 /tmp/phase74_corpus_projective "$input" \
      "$here/corpus_projective.tsv"
  python3 "$here/summarize_phase74.py" | tee "$here/summary_stdout.json"
} 2>&1 | tee "$here/run.log"
