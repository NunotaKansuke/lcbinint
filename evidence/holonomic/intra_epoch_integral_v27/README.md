# v27: integral cost within one epoch

## Accepted measurements

Same 87 selected normal `(id,u)` conditions, relative tolerance 1e-4, one thread, five repetitions, mean of per-condition medians. Fresh magnifiers/workspaces for every repetition. Seed construction and radial planning included; caustic construction excluded as previously agreed. No compilation overlaps accepted timing. No source/geometry cache across epochs is introduced.

| Stage | Before (v26), ms | After, ms | Reduction |
|---|---:|---:|---:|
| seed | 0.084186 | 0.083816 | 0.4% (noise) |
| radial planning | 0.090809 | 0.088086 | 3.0% |
| integral | 0.217710 | 0.167476 | **23.1%** |
| entire path excluding caustic construction | 0.395392 | 0.342136 | **13.5%** |

Reverse order: integral 0.216861→0.166063 ms (23.4%); entire path 0.393842→0.340325 ms (13.6%). The goal of less than 0.3 ms is not achieved. About 0.084 ms seed and 0.088 ms planning remain in this measurement, alongside about 0.167 ms integration.

## Changes

1. **Reciprocal fold precision selection.** `topology_event_estimate` previously always called software quad Newton for a finite reciprocal-coordinate seed. The ordinary t chart already tries DD and accepts it under explicit residual, radial derivative and radius-budget tests. Apply the same DD contract in the reciprocal chart (8 Newton iterations maximum, step tolerance 1e-24, scaled residual 1e-14, unchanged derivative and 256-ULP/value-budget screens). Failure runs the existing quad solve from its original seed. `HOLO_RECIPROCAL_EVENT_QF_ONLY` restores the old reciprocal policy. The direct t chart, polynomial seed finder, physical fold equations and requested tolerances remain unchanged.
2. **Optional angular payload.** Ordinary `AdaptiveSample` objects carried 416 bytes of result/error/derivative coefficient arrays for optional angular upgrades. Move these arrays to an on-demand `AdaptiveAngularState`. A const empty view requires no allocation; writes allocate once and use copy-on-write when a prior sample shares the payload. Access is through `angular_data()`. Sample size is **1040→640 bytes**, panel size remains1472. Angular/Jacobian algorithms and state remain available. Persistent payload plus conservative shared-control-block allowance is included in workspace memory checks, including Hermite sidecars and upgrade replacement. Tests cover allocation, sharing, copy-on-write isolation, insufficient budget and sufficient budget.
3. **Compiled value lanes.** `mapped_radius` is now a template with ordinary generic, plain value and uniform-source specializations. Select the plain lane only when derivatives, angular adaptation, Hermite shadow and atlas hooks are absent. Compile away unused derivative/upgrade code; uniform-source specialization also eliminates LD code. The force-cold flags and seed selection match the original dispatch. `HOLO_ADAPTIVE_GENERIC_VALUE_BODY` restores generic dispatch. `HOLO_ADAPTIVE_VALUE_BODY_AUDIT` evaluates both implementations at each sampled radius and requires exact equality of all value/error arrays, crossing count, reliability and reject reason.
4. **Opt-in internal cost clocks.** Physical-node, estimator and scheduler clocks in `adaptive_detail::integrate` are enabled by `AdaptiveConfig.collect_costs`, diagnostic collection, an installed `V2Profile`, or explicit Hermite microtiming. Unrequested cost fields remain zero. Solver control flow does not depend on these durations. Overall epoch/planning/integration timers used for accepted comparisons remain active. Instrumented benchmarks retain inner clocks.

The partial trial sequence was about0.391→0.364 ms with reciprocal DD, then about0.354 ms with optional angular state, then about0.341 ms with value lanes and optional inner clocks. These are short three-repetition exploratory comparisons, not additive isolated estimates of every individual change. Final accepted matched/reverse runs above supersede them.

## Correctness

- All896 normal/fold/deeper conditions: magnifications and node counts exactly equal to v26; maximum relative candidate change0.
- All896 also complete the per-radius generic/value specialization audit; audited and production magnifications exactly equal.
- 754 converged D14 references: no failures/tolerance violations against either the new or saved pre-change reference. Maximum discrepancy/tolerance0.249.142 unresolved references are excluded from independent accuracy acceptance. Successful equivalence does not establish a new universal image-support proof.
- 1614 adaptive checks, including existing gradients, angular upgrades, resource budgets and6 added payload checks: zero failures.
- 23 frontier,43680 interpolation and3000 known-root polynomial checks: zero failures.
- Gradient performance and the opt-in angular adaptation path were not separately benchmarked. Their numerical tests and budget checks pass.

## Diagnostic breakdown

`summary.json` contains nested instrumented fields. These are diagnostic arithmetic means and include timer overhead; do not add them or substitute them for production timings. The before diagnostic log is the saved v26 run on the same host, not a fresh matched profile.

- Event setup0.03312→0.00718 ms, consistent with the DD-only trial's main gain.
- Sample allocation0.01657→0.01022 ms, with the same3.57 allocations on average.
- Arc solve/transport remains about0.061 ms; endpoint evaluation about0.0186 ms; per-node count about0.0227 ms. These remain candidate targets for further work. No further reduction is established here.

## Reproduction and provenance

Run in the existing `experiment/seeded-boundary-events` checkout. `build.sh` uses the saved v26 epoch/radial/atlas overlay for the baseline and the v26 seed-provider object for both routes. `build_audit_profile.sh`, `build_tests.sh` and `build_size.sh` build the other checks. Finish all compilation, then run `run_accepted.sh`. `summarize.py` records measurements, snapshots, hashes, checkout and binary identities. The canonical `bench_seeded_radial_master` binary has been updated to the accepted after binary; its build helper compiles the current local headers.

`pre_final_code/` preserves a completed intermediate measurement before the final generic-audit refactoring and memory-accounting checks. The top-level matched/reverse and after/audit suite outputs are the final accepted measurements. Rejected compiler attempts have build logs and are not timing evidence. The shared root checkout, seeds and caustic construction were not edited in this pass. No commit or push.
