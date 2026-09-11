#!/usr/bin/env bash
set -euo pipefail
# Run from repository root. Timing jobs must finish before this validation.
output=${1:-/tmp/radial-atlas-validation}
mkdir -p "$output"
flags=(-O3 -march=native -funroll-loops -ffp-contract=fast -fno-math-errno -std=gnu++17 -Isrc)
g++ "${flags[@]}" tests/holonomic_cpp/emit_atlas_enclosures.cpp -lquadmath -o "$output/emit"
"$output/emit" > "$output/enclosures.tsv"
python checks/holonomic/check_atlas_enclosures.py "$output/enclosures.tsv" "$output/enclosure_audit.json"
python checks/holonomic/validate_radial_stationary_atlas_plan.py "$output/exact_plan.json"
g++ "${flags[@]}" checks/holonomic/diagnose_radial_atlas.cpp -lquadmath -o "$output/diagnose"
"$output/diagnose" evidence/holonomic/d14_precision_warm_20260912/trajectory/input_snapshot.tsv > "$output/diagnostics.txt"
cmake -S tests/holonomic_cpp -B build-holonomic-m7
cmake --build build-holonomic-m7 -j4 --target test_radial_atlas test_adaptive_radial \
  test_point_images test_quartic_sturm test_holonomic_m7 test_holonomic_transport \
  test_finite_source_binary test_root_pair test_holonomic_ode test_gm_taylor \
  test_gm_physical test_chart_p4 test_reciprocal_chart test_gm_coverage \
  test_gm_lauricella6 test_gm_phase7_shared test_d14_positive
ctest --test-dir build-holonomic-m7 --output-on-failure -j4 > "$output/ctest.txt"
