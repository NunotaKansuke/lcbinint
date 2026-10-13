# v20: coarse-to-fine prediction by contiguous sums and bounded sine transforms

Work remains in the existing experimental checkout. Seed, quadrature weights, numerical root gates, topology discovery and tolerance unchanged. No magnification map is constructed.

## Algorithm

- Gather one component's coarse values into a compact array once, and check coarse component validity once. For direct prediction, preserve the exact coefficient/product order and compensated sum. The norm accumulation order for each component remains unchanged.
- At the largest existing Fejer rule (255 interior points, level8), value-only prediction uses a DST-I reformulation of the same coarse interpolating polynomial, implemented with two bounded radix-2 forward FFTs. Multicomponent/gradient predictions and all smaller rules retain compensated direct prediction.
- FFT plans depend only on dyadic grid size and are immutable; no lens-geometry or source-epoch cache is introduced. Work arrays are fixed bounded stack storage; maximum FFT length512.
- `HOLO_ADAPTIVE_DISABLE_DST_INTERPOLATION` selects gathered direct prediction everywhere. `HOLO_ADAPTIVE_LEGACY_INTERPOLATION` restores the original prediction loop. Saved pre-pass binary: `build-seeded-events/bench_seeded_before_v20`; pre-pass header: `adaptive_radial_before.hpp`.

For coarse M and theta_j=pi*j/M, the interpolant can be written

p(cos(theta)) = sum_(k=1..M-1) a_k sin(k*theta)/sin(theta),
a_k = (2/M) sum_(j=1..M-1) f_j sin(theta_j) sin(k*theta_j).

A sine transform gives the coefficients; a second zero-padded transform evaluates the fine angle grid. This is polynomial equivalence in exact arithmetic, not bitwise equivalence of predictions. Separate polynomial/dense checks and full-integrator checks are reported below.

## Why only255 points?

The bounded standalone microbenchmark showed transforms slower at7,15,31,63 and127 interior points; at255, direct prediction0.02123 ms vs sine transforms0.01346 ms. This workload uses repeated smooth/oscillatory test vectors and cached plans, excludes per-panel source preparation, and does not establish universal crossover on other hardware. `prediction_micro.tsv` records all sizes and prediction differences. The threshold is based on this comparison rather than an accuracy heuristic.

Actual normal run histogram over261 instrumented repetitions: level3..8 counts1284,927,513,84,69,45. Only45/2922 (1.54%) panel estimates use level8. This limits the effect on overall runtime.

## Matched normal measurements

Same87 selected `(id,u)` conditions, tolerance1e-4; one thread, fresh workspaces and master seeds; all compilation finished before serial timing;5 repetitions per condition, mean of medians. Combined time excludes caustic construction and includes seeds/preparation/integration.

| Stage | Before ms | After ms | Reduction |
|---|---:|---:|---:|
| Integration | 0.26517 | 0.26344 | 0.65% |
| Combined excluding caustic construction | 0.48028 | 0.47591 | 0.91% |

Reverse order: integration0.26535→0.26309 ms (0.85%); combined0.47930→0.47603 ms (0.68%). This is a small observed improvement, not a large speedup. Unchanged preparation stages fluctuate; do not attribute the entire combined difference to interpolation.

Intermediate normal trials: gather-only integration0.26371 ms; initial DST prototype0.26253 ms. Those do not establish the standalone DST increment with statistical confidence. The prototype applied its rule to all component counts; final accepted code restricts transforms to value-only. These normal trials have one component throughout, so their numerical algorithms match.

Instrumented prediction stage0.01204→0.01031 ms (14.35% reduction), full estimator0.01718→0.01549 ms. Three repetitions, arithmetic means, timer overhead included. Level histogram unchanged. This agrees in magnitude with the small uninstrumented integration reduction.

## Validation

-43680 standalone checks compare predicted values against compensated dense interpolation and analytic Chebyshev polynomials across all supported levels. Zero failures. Maximum dense discrepancy relative to input scale6.47093e-13; fixed test acceptance2e-11. This validates this representative matrix, not an interval roundoff bound for arbitrary input values.
-1608 adaptive checks and23 frontier checks: zero failures. Tests retain interval-sign, nearest-anchor and count-witness audits.
-896 normal/fold/deeper conditions: maximum magnification change0 and unchanged node counts relative to pre-v20 accepted code.
-754 resolved D14 reference conditions: zero candidate failures; discrepancy/configured tolerance at most0.249.
-142 unresolved D14 reference conditions excluded from independent-reference acceptance. Before/after equality does not establish their physical correctness or universal component coverage.

## Reproduction/artifacts

Checkout `/rogue1_8/nunota/lcbinint/experiments/seeded-boundary-events`.

- Candidate: `benchmarks/holonomic/build_seeded_radial_master.sh`.
- Current transform variant: `build_dst.sh`; gathered direct: `build_gather.sh`.
- Tests: `build_tests.sh`.
- Profile scripts: `build_profile_before.sh`, `build_profile_after.sh`. The saved before-profile binary was built before the algorithm edits. The before script now includes `-DHOLO_ADAPTIVE_LEGACY_INTERPOLATION` to reconstruct that numerical path.
- Microbenchmark compilation: `g++ -std=c++17 -O3 -march=native -fext-numeric-literals -Isrc benchmarks/holonomic/bench_chebyshev_prediction.cpp -lquadmath -o build-seeded-events/bench_prediction_v20`.
- Finish all compilation before timing. Benchmark: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BINARY INPUT OUTPUT REPS 0`, with `evidence/holonomic/caustic_seeded_connection_v4/{initial,fold,deeper}_minimal_seeds.tsv`. Live master seeds regenerated. Normal5 repetitions, holdout3, profile3.
- Raw `before/gather/dst/after/before_reverse/after_reverse.tsv`, `fold/deeper.tsv`, profile logs/TSV, tests, microbenchmark retained.
- `summarize.py` creates summary, source snapshots and hash/provenance manifest. Holdout equality uses accepted v19 outputs, not historical timing for a holdout speedup claim.
- PROFILE adds column37 prediction_ms and columns38..43 level3..8 counts to v17's layout.

This bounded pass is complete. Further tiny transform tuning is lower priority than the remaining physical boundary/node evaluation costs.
