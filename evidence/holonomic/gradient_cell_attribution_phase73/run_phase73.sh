#!/usr/bin/env bash
set -euo pipefail
root="$(git rev-parse --show-toplevel)"
here="$root/evidence/holonomic/gradient_cell_attribution_phase73"
phase72="$root/evidence/holonomic/moving_map_gradient_phase72"
input="$root/evidence/holonomic/adaptive_radial_20260911/reference_cases.tsv"
cells="$phase72/cells.tsv"
nodes="$phase72/nodes.tsv"
common=(-O3 -DNDEBUG -std=gnu++17 -fext-numeric-literals -I"$root/src"
  -DHOLO_D14_NATIVE_RESIDUAL_SCREEN -DHOLO_D14_DIRECT_CONVOLUTION
  -DHOLO_ADAPTIVE_STATIONARY_ENDPOINT -DHOLO_D14_WARM_NOISE_HANDOFF
  -DHOLO_D14_SCALAR_CONSTANTS -DHOLO_ARC_COMPACT_STORAGE
  -DHOLO_ADAPTIVE_SPARSE_ANCHOR -DHOLO_ADAPTIVE_SAMPLE_GROWTH
  -DHOLO_MV_RATIONAL_MIDPOINT -DHOLO_QUARTIC_WARM_SOLVE_TOL=1e-11
  -march=native -funroll-loops -ffp-contract=fast -fno-math-errno)
mkdir -p "$here"
{
  c++ "${common[@]}" -DHOLO_ADAPTIVE_FOLD_QUARTIC_SEED \
    "$here/direct_derivative.cpp" -o /tmp/phase73_direct_derivative -lquadmath
  c++ "${common[@]}" -DHOLO_ADAPTIVE_FOLD_QUARTIC_SEED \
    "$here/endpoint_audit.cpp" -o /tmp/phase73_endpoint_audit -lquadmath
  c++ "${common[@]}" -DHOLO_ADAPTIVE_FOLD_QUARTIC_SEED \
    "$here/same_node_audit.cpp" -o /tmp/phase73_same_node_audit -lquadmath
  c++ "${common[@]}" "$here/ld_decomposition.cpp" \
    -o /tmp/phase73_ld_decomposition -lquadmath

  env -u HOLO_ADAPTIVE_EVENT_REUSE -u HOLO_MV_TRANSPORT_LEGACY \
    -u HOLO_HOLONOMIC_TRANSPORT -u HOLO_HOLONOMIC_TRANSPORT_LEGACY \
    PHASE73_ONLY_CASE=caustic-cross taskset -c 0 /tmp/phase73_direct_derivative \
    "$input" "$cells" "$here/caustic_unregularized_direct.tsv"
  env -u HOLO_ADAPTIVE_EVENT_REUSE -u HOLO_MV_TRANSPORT_LEGACY \
    -u HOLO_HOLONOMIC_TRANSPORT -u HOLO_HOLONOMIC_TRANSPORT_LEGACY \
    PHASE73_ONLY_CASE=caustic-cross PHASE73_P4_FOLD_PROBE=1 taskset -c 0 \
    /tmp/phase73_direct_derivative "$input" "$cells" \
    "$here/caustic_fold_map_diagnostic.tsv"
  env -u HOLO_ADAPTIVE_EVENT_REUSE -u HOLO_MV_TRANSPORT_LEGACY \
    -u HOLO_HOLONOMIC_TRANSPORT -u HOLO_HOLONOMIC_TRANSPORT_LEGACY \
    PHASE73_ONLY_CASE=rand035 taskset -c 0 /tmp/phase73_direct_derivative \
    "$input" "$cells" "$here/rand_direct.tsv"
  env -u HOLO_ADAPTIVE_EVENT_REUSE -u HOLO_MV_TRANSPORT_LEGACY \
    -u HOLO_HOLONOMIC_TRANSPORT -u HOLO_HOLONOMIC_TRANSPORT_LEGACY \
    PHASE73_ONLY_CASE=rand035 PHASE73_RAND_EXTENDED=1 PHASE73_ONLY_CELL=2 \
    taskset -c 0 /tmp/phase73_direct_derivative "$input" "$cells" \
    "$here/rand_cell2_direct_high.tsv"
  env -u HOLO_ADAPTIVE_EVENT_REUSE -u HOLO_MV_TRANSPORT_LEGACY \
    -u HOLO_HOLONOMIC_TRANSPORT -u HOLO_HOLONOMIC_TRANSPORT_LEGACY \
    PHASE73_ONLY_CASE=rand035 PHASE73_RAND_EXTENDED=1 PHASE73_ONLY_CELL=3 \
    taskset -c 0 /tmp/phase73_direct_derivative "$input" "$cells" \
    "$here/rand_cell3_direct_high.tsv"
  env -u HOLO_ADAPTIVE_EVENT_REUSE -u HOLO_MV_TRANSPORT_LEGACY \
    -u HOLO_HOLONOMIC_TRANSPORT -u HOLO_HOLONOMIC_TRANSPORT_LEGACY \
    PHASE73_ONLY_CASE=rand035 PHASE73_RAND_EXTENDED=1 PHASE73_ONLY_CELL=8 \
    taskset -c 0 /tmp/phase73_direct_derivative "$input" "$cells" \
    "$here/rand_cell8_direct_high.tsv"
  taskset -c 0 /tmp/phase73_endpoint_audit "$input" "$cells" \
    "$here/endpoint_audit.tsv"
  taskset -c 0 /tmp/phase73_same_node_audit "$input" "$nodes" \
    "$here/same_node_nodes.tsv" "$here/same_node_cells.tsv" \
    "$here/same_node_arcs.tsv"
  taskset -c 0 /tmp/phase73_ld_decomposition "$input" "$nodes" "$cells" \
    "$here/ld"
  python3 "$here/summarize.py"
} 2>&1 | tee "$here/run.log"
(
  cd "$root"
  find evidence/holonomic/gradient_cell_attribution_phase73 -type f \
    ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum
) > "$here/SHA256SUMS"
