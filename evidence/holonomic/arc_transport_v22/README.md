# v22: boundary-root and arc construction

## Accepted result

Same87 normal `(id,u)` cases, relative tolerance1e-4, five repetitions per condition, fresh geometry/workspaces, one thread, live master seeds. All builds finished before serial timing. Timing mean of condition medians; caustic construction excluded from combined time.

| Stage | Before ms | After ms | Reduction |
|---|---:|---:|---:|
| Integration |0.24997|0.23079|7.67%|
| Combined including seeds, excluding caustic construction |0.46177|0.44644|3.32%|

Reverse order: integration0.25401→0.22813 ms (10.19%); combined0.46720→0.44245 ms (5.30%).

Matched instrumented profile: arc construction0.08636→0.06097 ms (29.4%). Earlier0.077 ms used fewer nested timers; do not mix that number with this new timed decomposition. The50% objective is not achieved. Remaining reduction to half this baseline is approximately0.01779 ms.

Profile children: assembly0.01139→0.00470; outside-pair quartic solve0.01690→0.01038; pair-fallback solve0.02424→0.01622. Pair total0.04861→0.03832 includes the last child and is not additive. Profile clocks add overhead; accepted total runtimes use the uninstrumented binaries.

## Accepted algorithm changes

1. With a square-free certified count of2 or4 and matching previously solved real-root count, Newton-update the existing real roots. Require finite convergence, scaled Horner residual1e-11, conservative root separation, and unchanged complete angular count. Complex seeds remain available for Aberth fallback.
2. If that update fails, factor the depressed quartic using the real cubic resolvent into two quadratics, then polish its real candidates. Apply the same residual/count/separation gates. Difficult/ambiguous candidates fall back to the original warm/cold Aberth solver at this node. No tolerance, sample count, physical endpoint polish or error-estimator gate changed.
3. After successful physical inside-pair transport, a complete certified root count allows returning those arcs directly. Retain root residual, branch/gap and physical-midpoint checks. Avoid re-sorting/merging a heap-backed angular-root list and another physical interval-sign evaluation. Warm-D14 cross-check consumers retain the original path. Direct-arc audit recomputes the incumbent interval builder and compares exact endpoints.

`HOLO_QUARTIC_LEGACY_ARCS` restores the pre-v22 algorithm for matched comparisons. `HOLO_TRANSPORT_DIRECT_AUDIT` checks every direct arc against the incumbent construction.

## Validation

896 initial/fold/deeper cases with direct-arc, count-witness, interval-sign, nearest-anchor and fold-map audits: all complete successfully.754 converged D14 references: zero candidate failures or tolerance violations.142 unresolved D14 references remain excluded from independent acceptance. Before/after maximum relative magnification change1.84e-12; no node-count changes. Maximum reference discrepancy/tolerance0.249.

Existing tests:1608 adaptive +23 frontier +43680 interpolation checks, zero failures. New independent polynomial test:3000 quartics built from known real roots and conjugate pairs with seven coefficient scales; zero failures, maximum angular error3.12e-11, acceptance1e-8. This is sampled validation, not a universal completeness theorem for all physical configurations.

## Trials retained as evidence, not adopted

- Re-enable pair transport after its cumulative trip cap: negligible gain.
- Refresh fallback Aberth seeds from the more recent pair state: negligible gain.
- Use real-only transport everywhere: slower than the accepted hybrid.
- Compute quadratic-factor candidates in long double: no additional gain.
- Match real Newton stop to existing loose seed tolerance: negligible gain; strict1e-13 retained.
- Factor in reciprocal chart selected by larger leading coefficient: no measured gain; reverted.

Rejected implementation branches removed; raw trial binaries/logs/TSVs retained. Trial build scripts record their historical flags; final `build_before.sh`, `build_after.sh`, `build_audit.sh`, `build_tests.sh`, `build_profile_{before,after}.sh` reproduce accepted code. `run_accepted.sh` runs serial matched/reverse timings, profiles and full-suite audits after compilation. `summarize.py` records source snapshots/hashes, checkout identity and results in summary.json/manifest.json.

Remaining main cost is solver work at first/failed pair nodes and the successful pair predictor/corrector/validation itself. The present checkpoint achieves29%, not50%; further removal needs a new measured decomposition rather than dropping correctness gates.
