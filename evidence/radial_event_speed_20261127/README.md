# Radial event exclusions and binary quartic timing

Compared the existing working copy of experiment/algebraic_boundary at 7038d6c2e7305ef893031094d0c59bde4755922a with the same copy after the optimization. Both builds use GCC 11 Release on rogue1. The native extension SHA-256 hashes are 6d524379366569bc3dbb5add4636b59bcd647557378f36030e9ec85b5d1fb3b7 (before) and 1fc10968154065f03d13177e9bacde4dcc18af49f9657c6649eeb186b557c596 (after).

The parent checkout's build-accuracy-20261127/radial-speed directory contains both packages (before and adopted), the raw paired runs, verification logs, comparison scripts and the rejected physical-only subdivision trial. A fresh FFI binding linked against the adopted native libraries was used for JAX and finite-difference regression tests.

## Timings

* eight-modes.tsv uses ../eight_mode_speed_20261127/bench.py with case offsets 0 and 1, absolute tolerance 0 and relative tolerance 1e-4. Each two-epoch public call includes result construction. Times are the median total calling-thread CPU time across three runs divided by the epoch count. CPU 2 was pinned, OMP and OpenBLAS each used one thread, and build order alternated. Each binary mode has 128 converged value candidates with status COVERAGE_UNCERTIFIED. Each triple mode has 128 OK statuses. Across both brightness profiles, case 810 accelerated by 3.20x for values and 3.80x for gradients (median times).
* reference-speed.tsv uses the previous quick comparison driver with the build directory passed as its argument. It measures warm wall times at absolute and relative tolerance 1e-4. The binary trajectories have 13 epochs at each of q=0.01 and 10, and the triple comparison uses three epochs each from cases 568 and 889. The reference is VBMicrolensing for values and ../autodiff-vbm with synchronized JAX x64 for binary derivatives. Each curve time is the median of five warm calls and the table reports the median summed time across three processes. Compiler, import and JIT startup are outside both steady-state comparisons.

## Numerical comparisons

Across 1024 epoch-mode combinations, the maximum change normalized to max(1, baseline magnitude) was 1.30e-12 for magnification and 2.32e-13 for the gradient (per-epoch infinity norm). Convergence, validity, integration, seed and support flags agreed. An additional 48 stratified triple cases (seed 20486, 288 epoch-mode combinations) retained all OK statuses and matched all recorded flags. Maximum normalized changes were 3.88e-10 and 1.13e-8. A separate 920-epoch binary orbital sweep covering q=0.001 to 1000, unsorted and duplicate epochs, circular and Kepler motion, both brightness profiles and both derivative modes reproduced the stored numbers and flags exactly.

strict-reference.tsv compares 64 distinct epoch/profile inputs on both the value and gradient-enabled routes against original VBMicrolensing (absolute tolerance 1e-8, relative tolerance 1e-6), in an isolated process per geometry. The maximum fractional value differences are 9.50e-6 for binary and 3.97e-5 for triple. Gradients have independent finite-difference checks in the regression suite. All 299 selected Python regression tests and six C++ targets passed.

The static uniform binary gradient was also retimed in isolation on the same core. Each of four before/after pairs used the median of 51 warm calls per q and calling-thread CPU time. The median two-curve sum was 2.357 ms before and 2.276 ms after (1.035x). The shorter five-call wall-time measurement showed the opposite direction for this row.
