#!/usr/bin/env bash
# Run after timing measurements, to avoid compiler/reference CPU interference.
set -euo pipefail
root=evidence/holonomic/d14_positive_real_20260912
input=evidence/holonomic/d14_precision_warm_20260912/trajectory/input_snapshot.tsv
cmake -S tests/holonomic_cpp -B build-holonomic-m7 > "$root/configure.log"
cmake --build build-holonomic-m7 -j4 > "$root/build.log" 2>&1
ctest --test-dir build-holonomic-m7 --output-on-failure -j4 > "$root/ctest.log" 2>&1
c++ -O3 -std=gnu++17 -Isrc -march=native -ffp-contract=fast checks/holonomic/d14_positive_audit.cpp -o /tmp/d14-positive-audit -lquadmath
/tmp/d14-positive-audit "$input" > "$root/exact-audit.txt"
python checks/holonomic/check_d14_positive_audit.py "$root/exact-audit.txt" > "$root/exact-audit.json"
c++ -O3 -std=gnu++17 -Isrc -march=native -ffp-contract=fast checks/holonomic/diagnose_d14_soft_cuts.cpp -o /tmp/d14-soft-diagnose -lquadmath
/tmp/d14-soft-diagnose "$input" > "$root/soft-cut-diagnostic.txt"
python checks/holonomic/validate_d14_positive_real_plan.py > "$root/exact-plan.json"
