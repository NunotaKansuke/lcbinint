# LD shared-support prototype (2026-10-13)

## Result and scope

The experimental overlay now connects `NestedDiskAreaState` to
`LimbLayerIntegralState`. Production/master routing is unchanged. This is an
estimated-error binary linear-LD prototype, not a globally certified integrator.

One outer trace supplies centre-image seeds and represented image support.
Nested radii reuse these seeds, cached lens-map distances and retained adaptive
area panels. No additional full lens-equation image solve is requested by the
nested disk provider. New radii still require run exploration and boundary
crossing refinement. Initial distance caching recomputes outer lattice distances
because the original trace does not export its distance cache.

Small disks use dyadic local spacing. Existing run-based flood fill replaces the
initial pixel BFS implementation. Repeated queries at an already achieved budget
perform no additional map evaluations. A local interval chord test prevents a
nearby centre seed from duplicating a represented connected image; it does not
prove global coverage. Safeguarded crossing refinement, square-root opening at
image tips and floating-point stopping floors address small-radius failures.

The LD layer identity and coordinate substitution are exact. Numerical error
indicators remain approximate (`certified=false`). The square-root LD coefficient
has analytic unit coverage; the binary numerical pilot below is c=0.5, d=0 only.

## Evidence

Artifacts are in `tests/diagnostics/results/recal2026/`:

- `nested-disk-area-v7`: 20/20 nested disk tests within their area budgets;
  repeat queries incur zero extra lattice/crossing/boundary map evaluations.
- `adaptive-image-area-v7` and `adaptive-image-area-holdout-v4`: 60/60 uniform
  checks at absolute magnification tolerances 1e-5 and 1e-7, no unresolved cases.
- `nested-ld-v5`: five changing binary geometries. All initial 1e-5 requests
  accepted and independently checked within tolerance. Four tight 1e-7 requests
  accepted. One tight request remains rejected due to an unresolved local gap
  turn. Its close numerical value is not grounds for overriding the rejection.
- `ld-uniform-reference-split-v1` and `v2`: uniform VBM disk areas integrated
  with both the native radial quadrature and independent SciPy quadrature.
  Their differences are at most 1.64e-11 in magnification. Accepted tight native
  results differ by 3.4e-9 to 3.9e-9. These are numerical checks, not certified
  bounds on the reference. Per-radius samples also check disk area budgets.
- `nested-ld-v5/corrected-reference-comparison.json`: postprocessed comparison
  against these independent references; earlier raw reference fields retained.

Installed VBM LD is not an adequate 1e-7 reference merely by lowering Tol. Local
source contains a default 100-annulus cap and a fixed 1e-5 inverse cumulative LD
profile stopping threshold. Increasing minannuli alone leaves precision limits.
`nested-ld-reference-v1` records the annulus study and source/native provenance.
The uniform disk reference bypasses the LD profile inversion.

## Accuracy and cost

| s | physical q | initial absolute error | tight absolute error | tight accepted |
|---|---|---|---|---|
|1.3|1e-4|1.21e-6|8.98e-9|no|
|1.1|1e-5|2.90e-7|3.41e-9|yes|
|0.6|0.3|3.03e-7|3.82e-9|yes|
|1.0|1.0|3.36e-7|3.86e-9|yes|
|2.0|0.1|3.39e-7|3.88e-9|yes|

Initial shared LD elapsed times sum to 0.4206 s (one pilot observation per case,
including outer trace and setup). Actual master with overlay OFF, its component
refinement enabled and nbin=512 takes 0.1081 s for the same five inputs (median
of three batches). Both meet 1e-5 on these inputs. This is evidence that the
current shared LD construction is slower; the differing repetition protocol
precludes treating the roughly 3.9 ratio as a precise universal benchmark.
Master nbin=64/128/256/512 meets 1e-5 in 1/1/2/5 of these five cases respectively.
No tested master resolution meets 1e-7 in these cases. This is a bounded pilot,
not a general statement about fixed-resolution accuracy.

Within the experimental routes, sharing helps some cases compared with separate
full area construction per radius, but that is not a master speedup claim.
The initial-stage total_disk_calls is the initial count; provider_calls and
unique_radii fields in the existing audit include subsequent tight refinement.

## Remaining work and next decision

The unresolved case is s=1.3, q=1e-4 at radius/outer-radius about 0.6518. A local
row band cannot resolve its gap turn. Repair should add a local contour/row patch
and retain completed area domains, rather than restart the old full pipeline.

Boundary work per radius dominates the present LD cost. In the difficult case,
more than two million boundary-map evaluations remain despite avoiding new image
solves. Further image-root reduction therefore does not address this cost.
Before optimizing this nested-radius design further, investigate direct weighted
integration on retained image rows with local brightness/error refinement. Keep
the exact LD law and normalization; approximate only the error indicator where
appropriate. The nested-radius route provides a diagnostic/reference path.

There is no proof that centre seeds and sampled outer support discover every
new inner-disk image component. Global mathematical certification, broad LD
coefficients, polar routing and production/default promotion remain outstanding.

## Follow-up: remove nested-radius cost

See `nbin-ld-cheap-error-report.md`. The newer scalar experimental entry keeps
the existing LD flux and adds only same-grid boundary/row indicators. Nested
disk integration is retained as a diagnostic path, not the proposed fast route.
