# Lightweight retained LD moments

## Implemented change

The opt-in adaptive LD grouping path now uses **only points already mapped on the preceding dyadic lattice** to decide whether to interpolate missing odd points. The former candidate sampling and LD interval-map arithmetic were removed. The uniform path and ordinary fixed/default routing retain their existing behavior.

For a 16-cell interval, read the cached even points and their halo. If any needed value is absent or outside the source disk, use the existing scan. For the odd point between two retained even points, use the four-point cubic interpolant with weights `(-1,9,9,-1)/16`. The leading interpolation defect is estimated from the retained fourth difference, multiplied by `3/128`. Sum these absolute defects, in physical flux units, into the existing error remainder. Large defects reject grouping; previously evaluated values remain available to the scan. No global magnification convergence comparison is used.

A cheap real-arithmetic Lipschitz bound checks that the full segment lies inside the source disk:

`|d lens_map/dx| <= 1 + sum_j m_j / min_segment |z-z_j|^2`.

The center distance plus half-width times this bound must remain below rho. A floating roundoff margin is included. This check is conservative in real arithmetic, but **is not a complete floating-point interval certificate**. The fourth-difference estimate is also not a general rigorous error bound. Existing `certified=false` and `complete_error_model=false` remain appropriate. The check runs before the full moment stencil; a missing center cache entry immediately rejects the candidate.

This is a local horizontal interpolation path during existing component refinement. It does not yet retain arbitrary vertical panels or provide cross-call adaptive resume. Actual fine endpoint states and the existing Bennett horizontal/vertical corrections remain in place. Profiles with nonzero second LD coefficient use the original sum.

## Measurements

Final main matrix: 74 geometry/profile combinations at both absolute targets, 148 calls, all estimated targets met, all actual errors passed saved independent VBM/radial-LD references. Five alternating timing batches, one thread, scalar calls including seed and all integration work; comparison is against the unchanged adaptive method.

| Target | Profile | New-map reduction | Time / baseline | Maximum actual error |
|---|---|---:|---:|---:|
| 1e-3 | c=0.5 | 0% | 1.0002 | 8.121e-4 |
| 1e-3 | c=1 | 0% | 0.9999 | 8.267e-4 |
| 1e-4 | c=0.5 | 8.21% | 1.0048 | 9.863e-5 |
| 1e-4 | c=1 | 7.88% | 0.9718 | 9.717e-5 |

At 1e-4 the scalar changes relative to baseline are at most 1.75e-6 and 2.35e-6. Uniform remains exactly equal to baseline and approximately 14% faster on this matrix through its existing grouping path.

Separate existing three-geometry reference matrix: 18 calls, all estimated targets met and all actual errors passed. Pure-LD 1e-4 timing was 1.0323 times baseline there. Thus a universal LD speed improvement has **not** been established. The heavy LD overhead was reduced on the main matrix, but saved map work does not always cover interpolation/cache/check overhead. The prototype remains opt-in via `LCBININT_DIAGNOSTIC_INTERIOR_BLOCKS=1`; no production/default promotion occurred.

No gates were relaxed. `unit_core` and `unit_boundary_support` passed (2/2). The added moment check verifies exact discrete sums through polynomial degree three and the leading-defect constant for degree four. This does not prove the estimator for arbitrary higher-order LD variation.

## Artifacts

- `tests/diagnostics/results/recal2026/ld-retained-moment-v3/summary.json`: final accuracy rows and matched timing samples.
- `tests/diagnostics/results/recal2026/ld-retained-moment-fresh-v1/summary.json`: separate reference matrix.
- `ld-retained-moment-v1` and `v2`: preserved intermediate results and source snapshots.
- `tests/diagnostics/recal2026/local_interior_block_audit.py`: reproduction with `--output UNIQUE_OUTPUT`; `--reference-source adaptive-shared-flux-fresh-v1` selects the separate matrix.
- `tests/diagnostics/recal2026/test_retained_ld_moments.py`: polynomial check.

Use project-local TMPDIR when rebuilding `build-boundary-support`. Run Python through `/home/nunota/.miniconda3/envs/myenv/bin/python` with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LCBININT_PROBE_BUILD=$PWD/build-boundary-support`. The audit toggles the diagnostic path inside the same native module and preserves the baseline's ordinary adaptive behavior. Source/native hashes and HEAD are recorded alongside the final result.
