# v25: reduce radial-support/event preparation

Caustic construction is per-orbit shared preparation and out of scope, per user correction. This pass changes local experimental frontier headers only. Shared master seed implementation and its linked libraries are not rebuilt or edited. Existing derivative state and full precision-ledger paths remain available.

## Accepted changes

- Add a coordinate-only mode to double_event_estimate, used only by the frontier's local physical-fold solver. That caller consumes t_seed, then independently runs the incumbent DD/qf physical fold refinement and acceptance tests. Avoid building the discarded precision ledger, its DD refinement, radial derivative coefficients and conditioning calculations. Other callers default to the full existing precision path.
- For coordinate proposals with a certified complete real count2/4, reuse the already validated factored-quartic/real-Newton kernel. Fall back to incumbent quartic solving if it cannot supply a usable complete set. Physical event refinement/precision thresholds remain unchanged. All candidate roots are ranked with the incumbent score; no derivative-cubic replacement adopted.
- Read event lists by const reference, avoiding copies of RadialEvent and its strings. Read the observed probe list by reference during repair: the called local-event routine does not mutate that list. March count, bisection counts, seed witnesses and integrator error thresholds unchanged.

HOLO_FRONTIER_LEGACY_EVENT_SEED preserves the original coordinate/precision proposal path for ablation. Full pre-pass comparison uses the saved before_include overlay (adaptive_epoch.hpp and seeded_radial_frontier.hpp) so event-copy behavior is also reproduced. Overlay includes the same guarded profiling clocks as after; timers disabled for accepted runtime.

## Matched timing

87 normal `(id,u)` conditions, tol1e-4, live master seeds, fresh source workspaces, one thread,5 repetitions. Arithmetic means of condition medians. Finish every compilation before serial timing. Caustic construction excluded; no cross-epoch geometry reuse assumed for local preparation.

| Stage | Before ms | After ms | Reduction |
|---|---:|---:|---:|
| Radial-support/event preparation |0.11014|0.09830|10.75%|
| Combined including seed construction |0.43522|0.42216|3.00%|

Reverse order: preparation0.10983→0.09859 (10.24%); combined0.43357→0.42410 (2.18%). Integration time remained approximately0.222ms. Seed generation remained approximately0.100ms and is not optimized in this pass.

Instrumented profile (3 repetitions): local-event work0.036736→0.029572ms (19.5%). Count probes0.043680→0.043762ms, unchanged. Counts/witnesses are retained. Combined reductions include eliminated event/string copies not charged to local-event timer. Instrumented times must not be summed with production medians.

## Validation

896 normal/fold/deeper conditions: before/after magnifications and evaluated node counts exactly equal.754 converged D14 references: zero candidate failures/tolerance violations; maximum discrepancy/tolerance0.249.142 unresolved references remain excluded from independent accuracy acceptance. All896 with direct-arc, interval-sign, fold-map, anchor-selection and count-witness audits.

1608 adaptive checks including gradient policies and resource budgets;23 frontier checks;43680 interpolation checks;3000 known-root polynomial checks: zero failures. Full precision/gradient APIs retain default coordinate_only=false. Gradient performance is not newly benchmarked.

## Rejected trials

- Skip12 support-edge count bisections:26 normal failures, wrong fold selection.
- Restrict direct fold to coarse bracket with local fallback: still normal failures; rejected.
- Use derivative-cubic stationary roots as initial coordinates:76 normal failures; rejected.
- Stream only two Sturm polynomials instead of full chain: no demonstrated gain; restored original quartic_sturm.hpp byte-for-byte.

Rejected code paths removed. Historical trial binaries/raw TSVs/logs retained; historical trial build scripts describe their original flags and do not reconstruct removed code from final sources. No speedup from an inaccurate trial is reported as accepted.

## Reproduction

build_before.sh/build_after.sh; build_profile_before.sh/build_profile_after.sh; build_audit.sh and build_tests.sh. Complete builds before run_accepted.sh. summarize.py records summary.json, manifest.json and source snapshots/hashes. before_include provides the source baseline. Final default bench_seeded_radial_master is updated to accepted v25. git diff --check passes. No commit/push performed.

Next outside-integrator costs: seed construction~0.100ms, count-probe planning~0.044ms. Simply removing seed topology witnesses or loosening root/event accuracy has not been accepted.
