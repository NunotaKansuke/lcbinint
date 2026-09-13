set -eu
flags=(-O3 -DNDEBUG -std=gnu++17 -fext-numeric-literals -Isrc -DHOLO_D14_NATIVE_RESIDUAL_SCREEN -DHOLO_D14_DIRECT_CONVOLUTION -DHOLO_ADAPTIVE_STATIONARY_ENDPOINT -DHOLO_D14_WARM_NOISE_HANDOFF -DHOLO_D14_SCALAR_CONSTANTS -DHOLO_ARC_COMPACT_STORAGE -DHOLO_ADAPTIVE_SPARSE_ANCHOR -DHOLO_ADAPTIVE_SAMPLE_GROWTH -DHOLO_MV_RATIONAL_MIDPOINT -DHOLO_QUARTIC_WARM_SOLVE_TOL=1e-11)
c++ "${flags[@]}" tests/holonomic_cpp/test_adaptive_radial.cpp -o /tmp/p59_unit -lquadmath
c++ "${flags[@]}" tests/holonomic_cpp/check_adaptive_reference.cpp -o /tmp/p59_ref -lquadmath
c++ "${flags[@]}" benchmarks/holonomic/bench_warm_gradient_parity.cpp -o /tmp/p59_jac -lquadmath
c++ "${flags[@]}" -march=native -funroll-loops -ffp-contract=fast -fno-math-errno evidence/holonomic/adaptive_estimator_phase10_20260913/runner_hybrid.cpp -o /tmp/p59_whole -lquadmath
/tmp/p59_unit > evidence/holonomic/adaptive_extreme_phase59/unit.txt
/tmp/p59_ref evidence/holonomic/adaptive_radial_20260911/reference_cases.tsv warm > evidence/holonomic/adaptive_extreme_phase59/reference.csv
/tmp/p59_jac evidence/holonomic/adaptive_radial_20260911/reference_cases.tsv > evidence/holonomic/adaptive_extreme_phase59/gradient.tsv
