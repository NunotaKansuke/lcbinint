# Seed-position proposals for radial fold preparation (v28)

Research checkout: `/rogue1_8/nunota/lcbinint/experiments/seeded-boundary-events`, branch `experiment/seeded-boundary-events`, HEAD `f3547723b79828d9c501182746028e99f7bad116`, with the existing dirty research changes retained.

## Accepted change

Use image positions already supplied by the unchanged master seed provider to propose local stationary folds before constructing/solving another angular quartic. No extra lens-equation solve and no cross-epoch geometry/source cache.

1. Retain each image seed's radius and position once. Select the seed nearest the proposed fold radius.
2. Form the bounded half-angle chart coordinate directly from the image: `t=Im(z)/(r+Re(z))` on the right half-plane, `u=-Im(z)/(r-Re(z))` on the left. Compute this only for the selected seed.
3. Run the existing long-double coupled `P=Pt=0` Newton and its unchanged polynomial and original-lens-map residual gates.
4. Exterior proposals must lie in the **final existing 12-bisection inner/outer bracket**; internal count-transition proposals must lie in their existing refined bracket. Keep the wider original exterior bracket for the original fallback, since coefficient rounding can move a nearly singular fold beyond the count bracket.
5. Only when this proposal fails or leaves the bracket, obtain the original quartic root proposal and continue the original endpoint fallback. Earlier radius/count probes, bands, events and accepted integration samples remain available.
6. The existing DD/quad endpoint refinement, error budgets, residual thresholds and all quadrature kernels are unchanged.

`HOLO_FRONTIER_LEGACY_IMAGE_START` disables the image proposal. The saved before overlay is the actual v27 frontier, so the accepted A/B baseline also excludes the new seed-position preparation overhead. Profile output appends proposal attempt/accept counts.

This is a local proposal acceleration. The existing `coverage_certified=false` remains; this does not establish universal radial-support completeness.

## Accepted measurements

One thread (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`), fresh epoch objects, same 87 normal `(id,u)` conditions, tolerance `1e-4`, five repetitions, arithmetic mean of per-condition medians. Caustic construction is excluded; master seed preparation is included. All compilation/tests finished before serial matched timing. Reverse order repeated.

| stage | before (ms) | after (ms) | reduction |
|---|---:|---:|---:|
| radial preparation | 0.088327 | 0.079593 | 9.89% |
| integration | 0.168232 | 0.168879 | -0.38% |
| full, including seeds and excluding caustic construction | 0.343407 | 0.334941 | 2.47% |

Reverse order: preparation 0.088045→0.079349 ms; full 0.342740→0.333380 ms. Integration and seed differences are measurement noise; no integration/seed speedup is attributed to this change. The <0.3 ms objective and a 50% preparation reduction are not achieved.

Fresh nested diagnostic profiles: fold proposal/refinement 0.028752→0.020199 ms (29.7% reduction). Count diagnostics 0.035691→0.035667 ms. Timers overlap/includes overhead: do not sum these or replace production timings with them.

For the selected normal profile conditions, 1344/1668 image-position proposals (80.6%) were accepted, omitting their angular quartic proposal solves. This is not a reduction in all polynomial solves or all lens-equation solves.

## Validation

- 896 matched conditions across initial, fold and deeper matrices.
- 754 converged D14 references: zero convergence/tolerance failures, including comparison with the saved before references.
- 142 unresolved D14 references are excluded from accuracy claims.
- One output changes by relative `1.386113752876232e-16`; all other magnifications are identical. **Zero node-count changes** against v27.
- Maximum discrepancy/reference tolerance remains `0.24896766490094166`.
- 1614 adaptive checks, 23 frontier checks, 43680 interpolation checks, 3000 known-root transport checks: zero failures; frontier rebuilt after the final narrowed-bracket change.
- `git diff --check` and shell syntax checks pass.
- Gradient performance is not independently benchmarked. Derivative kernels and their precision policy are unchanged, and the existing adaptive numerical suite passes.

## Rejected bounded trials

- Save and search every radius probe's coefficients/certificate: initial outputs unchanged, but no timing benefit; rolled back.
- Reciprocal scaling for count normalization: no useful production improvement; rolled back.
- Power-of-two count normalization: only ~2% synthetic gain, 3312/200000 changes in certificate metadata/count status on the broad random/degenerate matrix; rolled back. `audit_count.cpp`, `quartic_binary_trial.hpp`, `count_audit.log` retain that experiment, not accepted code.
- Stop preparation-side Newton at a binary64 seed accuracy instead of the original long-double stop: 896 outputs/nodes identical, no useful matched gain; original stopping rule restored.
- Eager chart conversion for all seeds and a wider image-proposal acceptance bracket are saved separately. Final code defers chart conversion and accepts shortcut exterior folds only in the final count bracket; it preserves the original wider fallback.

## Reproduction / installation

Run from this checkout: `bash evidence/holonomic/frontier_reuse_v28/build.sh`, `bash evidence/holonomic/frontier_reuse_v28/build_tests.sh`, then `bash evidence/holonomic/frontier_reuse_v28/run_accepted.sh`, then `python evidence/holonomic/frontier_reuse_v28/summarize.py`. `TMPDIR` is under this checkout. Baseline main/profile binaries in the accepted measurement were copied from the exact v27 binaries; the build script reproduces the corresponding saved frontier overlay and before profile source. Both routes link the same pinned v26 master seed-provider object and the same static libraries.

Source snapshots, baseline overlay, measurements, test logs, manifest and hashes are retained here. The research canonical `build-seeded-events/bench_seeded_radial_master` is updated to the accepted after binary; its build helper compiles current local headers. The shared root checkout is untouched. No commit/push.
