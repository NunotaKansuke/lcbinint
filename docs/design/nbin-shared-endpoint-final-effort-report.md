# Uniform / linear-LD: retained-row endpoint model refinement

## Scope and result

Experimental overlay only; scalar magnifications and the master production route
are unchanged. Error indicators remain **absolute magnification errors**, with
`certified=false` and `complete_error_model=false`. No tolerance was imposed on
the estimator. The 1e-4 and 1e-3 bands below are reporting scales.

Replayed 547 saved independent-reference comparisons: 175 previous comparisons
and 372 low-resolution comparisons, including overlapping input/resolution pairs.
The low-resolution set starts at nbin=16 and includes 24,32,48,64,96. All scalar
outputs agree exactly with their saved outputs and ordinary scalar calls; all
estimator `extra_map_evaluations` remain zero.

## Changes retained

1. Fit a cubic polynomial to four **squared geometric row widths**, continuing
   the regular endpoint root from the existing three-row quadratic fit. This
   gives a refined endpoint phase and curvature and the missing cubic
   contribution to the t^(5/2) width coefficient. Invalid or steep expansions
   retain the previous model. Newton iterations here solve a tiny local fitted
   polynomial, not the lens equation.
2. Center the old and new signed endpoint estimates and retain half their
   difference as model uncertainty. The same phase/geometry treatment is used
   for uniform and linear LD. LD brightness retains its existing regular t+t²
   model; the brightness model is not replaced by a separate algorithm.
3. At internal regular caps, convert held-row shape disagreement into its
   t^(5/2) quadrature contribution instead of adding a full row-value mismatch.
   This is enabled only within the existing expansion validity range; otherwise
   the original remainder is retained. The factor four is the existing heuristic
   remainder convention, not a new fitted coverage factor. None of these terms
   is a rigorous enclosure.

## Absolute error scale matters

Examples below use **actual error / previous estimate / new estimate**, not a
ratio alone. Changes were assessed against actual error magnitude, including
new deficits and the magnitudes of unnecessary excess estimates.

| Profile | nbin | Actual error | Previous estimate | New estimate |
|---|---:|---:|---:|---:|
| uniform, s=1.0100534161 | 16 | 1.82290e-3 | 1.77811e-3 | 1.83476e-3 |
| uniform, s=2 | 16 | 2.12061e-4 | 1.87713e-4 | 2.12582e-4 |
| uniform, s=1.9125769660 | 16 | 6.92453e-4 | 6.75373e-4 | 6.92447e-4 |
| LD c=.5, s=1.0100534161 | 16 | 1.55371e-3 | 1.52152e-3 | 1.56159e-3 |
| LD c=1, s=1.0100534161 | 16 | 1.14993e-3 | 1.13673e-3 | 1.15378e-3 |
| LD c=.5, s=1.3080280296 | 32 | 1.01836e-5 | 2.27465e-5 | 1.87520e-5 |

Low-resolution mean **absolute difference between estimated and actual error**:

| Profile | Actual-error band | Count | Before | After |
|---|---|---:|---:|---:|
| uniform | [3e-5,3e-4) | 67 | 3.28365e-5 | 3.22785e-5 |
| uniform | [3e-4,3e-3) | 43 | 1.08656e-4 | 1.06629e-4 |
| LD | [3e-5,3e-4) | 123 | 3.36542e-5 | 3.35718e-5 |
| LD | [3e-4,3e-3) | 64 | 1.09225e-4 | 1.08509e-4 |

These are modest improvements: approximately 1.7%,1.9%,0.2%,0.7%, respectively.
This is not evidence that the remaining tails have been solved. For example,
uniform actual error 2.29502e-3 is still estimated at about 1.60489e-3;
the maximum deficit in the low-resolution uniform 1e-3 band remains 6.90e-4.
LD's corresponding maximum deficit remains 2.87e-4. Existing maxima of excess
estimates also remain essentially unchanged. Some formerly accurate predictions
become slightly more conservative; the largest worsening of absolute prediction
error among c=0,.5,1 is about 5.9e-6. A small-error ratio degradation alone is not
used as a rejection rule.

Across all 547 comparisons, uniform deficits exceeding 10% decrease 4→3 and
LD estimates exceeding twice actual error decrease 19→17. No new >10% deficit
appears. These are supplementary ratio diagnostics, not the primary acceptance
criterion or a coverage guarantee.

## Rejected experiments

- Raising the regular LD brightness polynomial from t+t² to t+t²+t³ gave
  negligible benefit; retained the original cheaper brightness model.
- Restricting the coarse fallback to just the failed endpoint created eight new
  >10% deficits. The omitted contribution was not safely local to the endpoint.
  Rejected; kept the existing fallback. This shows why merely dropping terms
  cannot solve the remaining overestimates.
- The four-row geometry model alone produced some new ~8% deficits. The retained
  quadratic/cubic model spread prevents those new failures in this cohort.

Rejected trials and source snapshots are preserved under
`tests/diagnostics/results/recal2026/shared-endpoint-final-effort-v1/`.

## Cost and checks

Same-binary paired timing: five changing pilot geometries, c=0,.5,1,
nbin=16,32,64; eleven alternating pairs per configuration; one thread.
Legacy switches restore the previous quadratic endpoint and raw internal
remainder. Median time ratios new/previous range **0.99782–1.00263**.
No measured material slowdown in this bounded comparison. This does not measure
an automatic adaptive-resolution loop or all geometries.

Analytic cubic-cap checks verify phase and coefficients for nine known regular
caps and reject degenerate/nonfinite inputs. `unit_core` and
`unit_boundary_support` pass (2/2). The full magnification scalar-equality replay
checks the final Python build, not an installed package.

Artifacts:

- `shared-endpoint-final-v1/summary.json`: final 547 comparisons.
- `shared-endpoint-final-effort-v1/absolute-scale-audit.json`: low-grid absolute-scale results.
- `shared-endpoint-cost-final-v1/summary.json`: paired timing samples and native hash.
- `shared-endpoint-final-effort-v1/*audit.log`: retained and rejected trials.

Reproduce with the explicit build, one thread, and a new output name:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LCBININT_PROBE_BUILD=$PWD/build-boundary-support \
/home/nunota/.miniconda3/envs/myenv/bin/python -m tests.diagnostics.recal2026.shared_endpoint_effort_audit --output NEW_OUTPUT
```

## Remaining work

The substantial deficits and large excess estimates still come from outer-row
shape/fallback or possible support coverage outside the retained local model.
The next substantive step requires selectively resolving those ambiguous row
regions while reusing existing rows. Increasing local polynomial order further
has diminishing benefit. No minimum nbin, fitted safety multiplier, global
resolution doubling, or default promotion was introduced.
