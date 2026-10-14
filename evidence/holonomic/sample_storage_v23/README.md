# v23: reduce sample reallocation and movement

## Change

AdaptiveSample, all derivative/gradient arrays, continuation state and optional angular-upgrade state are preserved. Change only the spare-capacity growth when the existing proof shows the maximum permitted samples plus panels fit the memory budget: reserve up to2*target rather than1.5*target, capped at max_node_evals. When that proof does not apply, reserve exactly the required target as before. Hermite sidecar conditions, max_bytes, configured sample/panel limits and stop policies unchanged. More spare capacity reduces allocation and movement of all existing1040-byte samples. It can use more spare RAM; it is not a sample-layout reduction.

HOLO_ADAPTIVE_LEGACY_SAMPLE_RESERVE restores1.5 growth. The change applies to value and gradient paths and removes no derivative machinery. No new mathematical approximation or numerical threshold.

## Matched timing

87 selected normal `(id,u)` cases, relative tolerance1e-4, fresh geometry/workspaces, live master seeds, one thread,5 repetitions, arithmetic mean of condition medians. All builds finish before serial timings. Combined time includes seed construction and excludes caustic construction.

| Stage | Before ms | After ms | Reduction |
|---|---:|---:|---:|
| Integration |0.22975|0.22227|3.26%|
| Combined excluding caustic construction |0.44437|0.43652|1.77%|

Reverse order: integration0.22794→0.22396 (1.75%); combined0.44088→0.43749 (0.77%). State the observed range rather than a large single speedup.

Instrumented3-repeat profile: allocation/reallocation0.021868→0.016234 ms (25.8%); allocations5.184→3.575 (31.0%). Physical evaluation0.18395→0.18325; root/arc cost0.06047→0.06035, consistent with changing storage movement rather than solver work. Allocation timers include vector growth and relocation, not all per-node saving/anchor copying. Those latter operations have not been independently timed in this pass.

## Validation

896 normal/fold/deeper conditions, values and node counts exactly unchanged. Direct-arc, count-witness, interval-sign, nearest-anchor and fold-map audits enabled for all896.754 resolved D14 references: zero failures/tolerance violations.142 unresolved reference cases excluded from independent accuracy acceptance; maximum resolved discrepancy/tolerance0.249.

1608 adaptive checks including Strict/ValueFirst gradient behavior and resource budgets;23 frontier checks;43680 interpolation checks;3000 known-root polynomial checks: all pass. Gradient runtime gains were not separately benchmarked.

## Reproduction

build_before.sh, build_after.sh, build_audit.sh, build_tests.sh and build_profile_before.sh/build_profile_after.sh use this checkout with the live master seed provider. Complete all compilation first, then run_accepted.sh for serial matched/reverse runs and full-suite audits. summarize.py generates summary.json, manifest.json and source snapshots/hashes. Raw TSV/log outputs retained. The shared root checkout is untouched.

Remaining storage candidates are per-node sample construction/saving and parent-anchor access. Compact value-only storage is not implemented; derivative compatibility must be preserved in any future layout change.
