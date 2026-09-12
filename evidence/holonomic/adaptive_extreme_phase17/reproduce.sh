#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
phase_out=evidence/holonomic/adaptive_extreme_phase17
phase_input=evidence/holonomic/v2_adaptive_best_trajectory_20260913/input_snapshot.tsv
phase_flags=(-O3 -DNDEBUG -std=gnu++17 -fext-numeric-literals -Isrc -DHOLO_D14_NATIVE_RESIDUAL_SCREEN)
c++ "${phase_flags[@]}" -march=native -funroll-loops -ffp-contract=fast -fno-math-errno evidence/holonomic/adaptive_estimator_phase10_20260913/runner_hybrid.cpp -o /tmp/p17_whole -lquadmath
taskset -c 0 /tmp/p17_whole "$phase_input" "$phase_out/whole_screen.tsv" 3
python benchmarks/holonomic/summarize_adaptive_reciprocal.py --baseline ../adaptive_extreme_phase16/whole_bracket_bits.tsv --candidate ../adaptive_extreme_phase17/whole_screen.tsv --output ../adaptive_extreme_phase17/whole_summary.json
for phase_mode in base candidate; do
 phase_opt=()
 if [[ "$phase_mode" == candidate ]]; then phase_opt=(-DHOLO_D14_NATIVE_RESIDUAL_SCREEN); fi
 c++ -O3 -DNDEBUG -std=gnu++17 -fext-numeric-literals -Isrc "${phase_opt[@]}" tests/holonomic_cpp/check_adaptive_reference.cpp -o /tmp/p17_reference_check -lquadmath
 taskset -c 0 /tmp/p17_reference_check evidence/holonomic/adaptive_radial_20260911/reference_cases.tsv warm > "$phase_out/reference_${phase_mode}_rerun.csv"
 c++ -O3 -DNDEBUG -std=gnu++17 -fext-numeric-literals -Isrc "${phase_opt[@]}" benchmarks/holonomic/bench_warm_gradient_parity.cpp -o /tmp/p17_gradient_check -lquadmath
 taskset -c 0 /tmp/p17_gradient_check evidence/holonomic/adaptive_radial_20260911/reference_cases.tsv > "$phase_out/gradient_${phase_mode}_rerun.tsv"
done
cmp "$phase_out/reference_base_rerun.csv" "$phase_out/reference_candidate_rerun.csv"
cmp "$phase_out/gradient_base_rerun.tsv" "$phase_out/gradient_candidate_rerun.tsv"
