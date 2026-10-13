# Low-grid outer error: lattice phase and closed short gaps

## Result and scope

This continues `nbin-shared-endpoint-final-effort-report.md`. Changes remain in
its experimental overlay. Fixed-grid scalar magnifications are unchanged;
`extra_map_evaluations=0`, `certified=false`, `complete_error_model=false`.
No tolerance, minimum nbin, empirical geometry multiplier, or global grid-doubling
rule was added. Error quantities below are **absolute magnification errors**.

The low-grid comparison contains 372 cases at nbin=16,24,32,48,64,96. Its references
were reused, not regenerated to suit the estimator. The larger replay contains
547 cases including 175 previous comparisons, with overlapping geometry/grid
pairs. A further 45 checks use three previously unused geometry inputs.

## Root causes and implementation

### A valid local phase was being rejected

For uniform s=1.3, q=1e-4, nbin=16, a retained component's endpoint fits have
phases approximately 1.035 and 1.021 grid spacings. The width model itself permits
phases up to 1.5, but the outer error implementation rejected every phase above
one. This discarded useful signed information and substituted a second-difference
fallback. A phase above one can represent a missing subcell row; it is not by
itself evidence that the fitted curve is unusable.

Extend the existing moment evaluation using the exact lattice recurrence:

`R_p(delta) = R_p(delta-1) + (delta-1)^p` for `1 < delta <= 1.5`.

Bennett weights still refer to the first **retained** row at delta, so their
adjustments use delta and delta+1. The same recurrence is used for the existing
half-integer powers 1/2,3/2,5/2. Integer Bernoulli moments already have analytic
expressions and now permit the same domain. No lens-map query is needed. The
limit 1.5 is inherited from the existing endpoint root selection, not fitted
against these cases.

### Two endpoints of a short gap were being treated independently

A hole that lasts only a few rows has overlapping endpoint regions. Independent
half-power models, or a coarse fallback at each end, can double count shape
uncertainty and leave large unexplained junction curvature. This affects uniform
and the geometric `(1-c)` part of linear LD.

Recognize isolated, one-to-one gap chains of 3–7 retained rows. Both neighbouring
image rows must geometrically enclose the gap. Fit a concave quadratic to squared
gap widths and require its roots to lie within the inherited 1.5-cell endpoint
window. Integrate that **closed quadratic cap** analytically, rather than adding
two independent infinite-lattice endpoint estimates. Subtract its modelled
second differences at junctions. This reuses the existing crossing fractions,
row graph and areas.

For four or more rows, alternate interior samples provide model sensitivity:
center the two area predictions, retain half their spread, and account for
second differences of retained width residuals. Three-row fits have no independent
held-out width sample. The analytic cap integral is exact for that surrogate;
the surrogate is not an enclosure of the actual lens-map boundary.

The same geometry and signed/remainder bookkeeping is used for uniform and LD.
At c=1 the geometric gap contribution is identically zero, so skip that zero work.
The LD brightness-singularity model is unchanged.

## Absolute-scale evaluation

Reporting bands `[3e-5,3e-4)` and `[3e-4,3e-3)` are centered around the user's
1e-4 and 1e-3 scales; they are not acceptance tolerances or fitted parameters.
The statistic is the mean of **abs(estimated error − actual error)**, not a
ratio average.

| Profile | Actual-error band | Cases | Previous mean discrepancy | New | Reduction |
|---|---|---:|---:|---:|---:|
| uniform | 1e-4 scale | 67 | 3.22785e-5 | 5.90560e-6 | 81.7% |
| uniform | 1e-3 scale | 43 | 1.06629e-4 | 8.75914e-6 | 91.8% |
| LD | 1e-4 scale | 123 | 3.35718e-5 | 1.04808e-5 | 68.8% |
| LD | 1e-3 scale | 64 | 1.08509e-4 | 3.08896e-5 | 71.5% |

At 1e-3 scale, the worst deficit decreases from 6.90e-4 to 2.29e-5 for uniform,
and from 2.87e-4 to 2.00e-5 for LD. The worst excess estimate decreases from
1.24e-3 to 9.14e-5 for uniform, and from 9.00e-4 to 5.38e-4 for LD.

Representative **actual error / previous estimate / new estimate**:

| Profile and nbin | Actual | Previous | New |
|---|---:|---:|---:|
| uniform, s=1.3, 16 | 2.29502e-3 | 1.60489e-3 | 2.38644e-3 |
| uniform, s=.6098031856, 16 | 1.38897e-3 | 1.08465e-3 | 1.39121e-3 |
| LD c=.5, s=1.3, 16 | 1.78504e-3 | 1.49853e-3 | 1.90847e-3 |
| LD c=1, s=1.1072850495, 16 | 2.30807e-4 | 1.29837e-3 | 3.13614e-4 |

One new relative deficit exceeds 10%: LD c=.5, s=1.0468104907, nbin=24.
Actual error is 7.12301e-5; estimate changes 2.83271e-4 → 6.06769e-5.
The remaining absolute deficit is 1.05533e-5, while the previous excess was
2.12041e-4. Retained as a useful overall correction, not rejected solely on its
14.8% ratio deficit. This can still give a wrong acceptance decision if a requested
absolute tolerance lies between those two values; no acceptance guarantee is claimed.

## Independent fresh geometry checks

Holdout-pool indices **9,13,17** were absent from the previous replay. For each,
use c=0,.5,1 and nbin=16,24,32,48,64: 45 comparisons. References use independent
VBMicrolensing uniform areas and two SciPy radial-quadrature settings for LD.
No reference warnings; maximum reported reference stability 5.20e-12. All scalar
outputs agree exactly with ordinary calls, with zero extra estimator maps.

These checks reveal some increased conservatism. Fresh uniform 1e-3-band mean
absolute discrepancy increases 6.52e-6 → 1.27e-5; fresh LD's corresponding mean
increases 9.83e-6 → 1.31e-5. The largest new excess estimate in the fresh set is
7.57e-5, against actual error 4.81e-4 (previous estimate 4.95e-4, new 5.56e-4).
The maximum fresh deficit remains 1.79e-6. Thus the historical large tails improve
substantially, but this is not uniform improvement on unseen geometry.

## Cost and validation

Same-binary timing uses legacy switches for the old unit phase range and gap
fallback. Numerical replay of these switches is checked against all 547 saved
previous outputs. Zero-amplitude gap work is skipped in both timing paths, so
this is a conservative comparison with the optimized previous computation.

Five changing pilot geometries; c=0,.5,1; nbin=16,32,64; eleven alternating paired
measurements; one thread. Median new/previous time ratios **0.99880–1.00167**:
−0.12% to +0.17%, no material measured slowdown in this workload. This is not an
adaptive-resolution end-to-end benchmark.

Direct analytic checks cover shifted power moments and Bennett-weight adjustments,
regular cubic caps through phase 1.25, closed elliptical caps for 3–7 rows, and
invalid/degenerate inputs. Core and boundary-support tests are rebuilt and pass.

## Artifacts and reproduction

Under `tests/diagnostics/results/recal2026/`:

- `outer-phase-short-gap-final-v2/summary.json`: final 547 comparisons and absolute-scale statistics.
- `outer-phase-fresh-low-v1/summary.json`: independent reference generation, provenance and 45 original fresh checks.
- `outer-phase-fresh-short-gap-v2/summary.json`: final-model replay of those independent references.
- `outer-phase-closed-gap-cost-v1/summary.json`: timing samples and native hash.
- `outer-phase-extension-v1/`: snapshots, failed/partial experiment logs, unit logs and legacy replay.
- `outer-fallback-cause-v3/summary.json`: retained-row diagnosis. One trace containing refined trials is explicitly excluded from row interpretation; no cases are excluded from the 547-case numerical replay.

Example final replay, using a new output name:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LCBININT_PROBE_BUILD=$PWD/build-boundary-support \
/home/nunota/.miniconda3/envs/myenv/bin/python -m tests.diagnostics.recal2026.shared_endpoint_effort_audit \
--baseline shared-endpoint-final-v1 --output NEW_OUTPUT
```

## Remaining limitations

Some small-error cases still incur large excess estimates, especially pure LD
at short gaps: the geometric fix has zero amplitude at c=1, while the current
LD single-junction model cannot separate nearby junctions. For example actual
8.41e-6 is still estimated at 9.40e-4. LD at actual 5.91e-4 can still be estimated
at 1.13e-3. These are meaningful unnecessary-refinement risks, not dismissed
because the actual error is small. The next target is the brightness contribution
of overlapping short-gap junctions, reusing the same closed-gap geometry.

No certified area-error bound, default promotion, or adaptive acceptance rule
has been introduced by this change.
