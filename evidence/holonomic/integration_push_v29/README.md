# Single-core integration push v29

Checkout `/rogue1_8/nunota/lcbinint/experiments/seeded-boundary-events`, branch `experiment/seeded-boundary-events`, HEAD `f3547723b79828d9c501182746028e99f7bad116`, preserving the existing dirty research work. Baseline: accepted v28, unchanged master seed provider object from v26.

## Changes

- **Reciprocal symmetric-pair transport.** A source image crossing theta=pi cannot be represented by bounded direct half-angle t pairs. After a cold solve, try the direct chart first; if that cannot represent all inside arcs, try u=-1/t. Reverse quartic coefficients exactly, transform dR coefficients the same way, and reuse the existing center/squared-width predictor and corrector. Return theta=pi+2 atan(u) in the reciprocal chart. Existing root residual, separation, physical inside-midpoint, complete certified real-count and DD/quad endpoint gates remain. A singular reciprocal leading coefficient invalidates that continuation. Failed transport uses the existing quartic cache and original cold solve.
- **Remove transcendental round trips in midpoint checks.** Evaluate the original lens map from the bounded half-angle coordinate using rational sine/cosine identities. This changes rounding, not the physical equation. `HOLO_MV_TRIG_MIDPOINT` restores the trig evaluation. The full matrix and direct-arc audit cover both charts.
- **Remove redundant fmod after 2 atan.** Its range is already [-pi,pi]; retain the original conditional addition of 2pi for negative angles. No remainder is needed.
- **Incremental sample witnesses.** Keep each sample slot's last collected `(radius,count)`. Add it when that pair changes; invalid samples remain eligible later. Sort only the appended suffix, merge with the prior sorted prefix, and unique. This preserves the original accumulated set, including in-place count/radius changes and reliability transitions. All earlier probes/events/panels remain. `HOLO_FRONTIER_REPEAT_WITNESSES` restores repeated collection/full sort.
- **Reuse seed radii in motion initialization.** Use the exact radius already computed while preparing the image starts instead of recalculating hypot across all seeds for each uncovered band.
- **Compile one-component estimation separately.** Value-only estimate loops have a compile-time component count of one. Gradient estimation retains the general path.
- **Single-core SIMD compensated sums.** Four independent Neumaier sums accumulate value/inner/geometry/roundoff channels, preserving each channel's scalar operation order. Small-panel interpolation uses four compensated partial sums followed by a compensated reduction. Gradient interpolation retains its original scalar sums. AVX2 is selected by the compile target; targets without AVX2 retain the original production loops. `HOLO_ADAPTIVE_SCALAR_ESTIMATOR` restores scalar estimation. No multicore integration or cross-epoch source/geometry cache was introduced.

## Accepted performance

One thread (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`), fresh source/geometry objects for each repetition. Same 87 selected normal pairs, tolerance1e-4, five repetitions; arithmetic mean of per-condition medians. Seed generation included; caustic construction excluded. No compilation/tests run concurrently with accepted serial timing. Reverse order repeated.

| stage | before ms | after ms | reduction |
|---|---:|---:|---:|
| master seeds | 0.084192 | 0.083681 | 0.61% |
| radial preparation | 0.080082 | 0.072160 | 9.89% |
| integration | 0.169019 | 0.153222 | 9.35% |
| full excluding caustic construction | 0.336190 | 0.311817 | 7.25% |

Reverse order: full 0.339177→0.316164 ms (6.79%). Integration 0.169982→0.156622 ms. The full result is about0.312–0.316 ms. **The <0.3 ms objective is not achieved.** Seed timing differences are noise, not a claimed seed improvement.

Fresh nested diagnostic profiles are in `summary.json`: arc solve/formation about0.0615→0.0497 ms; estimator about0.0156→0.0107 ms; prediction about0.0104→0.0071 ms. These overlap and include timer overhead; do not add them or substitute them for production timings. Before/after production claims concern the complete combined change, not SIMD alone.

## Correctness / scope

- 896 conditions (initial/fold/deeper matrices).
- 754 converged D14 references: zero tolerance or convergence failures, also compared against saved before references.
- 142 unresolved D14 references excluded from accuracy claims.
- Zero node-count changes. Reciprocal continuation changes301 magnifications by at most relative6.268630929795326e-13. Maximum discrepancy/reference tolerance0.2489676651091549.
- The final SIMD addition changes zero outputs and zero node counts against the saved pre-vector implementation across all896 conditions.
- 1614 adaptive checks (including existing gradient/budget checks),321 frontier checks (including200 exact witness-union/duplicate checks and independent cold-root comparisons for the reciprocal chart),65772 interpolation checks,3000 known-root checks: zero failures.
- `HOLO_TRANSPORT_DIRECT_AUDIT`, fold-map, arc-sign, anchor and count-witness audits enabled in numerical suites. Reciprocal direct-arc audit correctly maps u coordinates back to source angles.
- Final acceptance budgets and physical residual screens are unchanged. Local coverage remains `coverage_certified=false`; these comparisons do not establish universal support completeness.
- Gradient performance was not separately benchmarked. Derivative routines remain available and their existing numerical tests pass.

## Reproduction and provenance

From this checkout: `bash evidence/holonomic/integration_push_v29/build.sh`, `bash evidence/holonomic/integration_push_v29/build_tests.sh`, finish compilation, then `bash evidence/holonomic/integration_push_v29/run_accepted.sh`, then `python evidence/holonomic/integration_push_v29/summarize.py`. Build temporary files remain in project `build-seeded-events`.

Accepted baseline binaries were copied from the exact v28 accepted binaries; the build script reproduces them using the saved epoch/radial/atlas/frontier/radius overlay. Both routes link the same v26 seed object and same static libraries. Snapshots/hashes, inputs, commands and results are retained here. `pre_vector` preserves the completed non-SIMD intermediate result around0.315 ms. `angular_*` and `chart_*` retain bounded intermediate measurements. `prepare_reciprocal.py` records an exploratory source transformation and is not a build prerequisite; final sources are in `source/` and the working checkout.

The canonical research binary `build-seeded-events/bench_seeded_radial_master` has been updated to the accepted after binary, and its build helper uses the new default local headers. Shared root checkout untouched. No commit/push. No active benchmark jobs.
