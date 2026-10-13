# Retained component adaptation with the shared Bennett flux estimator

## Scope and result

An experimental `binary_adaptive_flux` entry now starts at nbin16, discovers
seeds once, and replaces only selected image components with finer Bennett
run-fill results. It supports the same uniform/linear-LD kernel, c=0..1,
Cartesian mode. No standard/master routing or automatic resolution defaults
are promoted. This is separate from the older `binary_adaptive_area` contour
integrator: it retains the existing image-plane flux rule.

The 74 profile comparisons at each absolute target1e-3/1e-4 all meet the actual
error target after corrections. This is a regression matrix built from saved
independent VBM/radial-LD references, not a new random holdout. An additional
three perturbed physical configurations, chosen before adaptive evaluation,
provide nine new uniform/c=.5/c=1 profiles and18 target comparisons; all pass.
Their references use two uniform VBM tolerances and two SciPy radial quadratures,
with maximum stability difference5.88e-12 and no integration warnings.

## Retention and local work

- Initial16 uses the historical support/seed discovery once. Later fills receive
  exact dyadic lifts of the selected component's existing runs. No later image
  quintic solve or global seed scan is performed.
- Component flux, signed error, positive remainder, run intervals and crossing
  fractions are retained. Other components are not reintegrated.
- A flat lattice-value cache retains source-plane squared distances at exact
  physical double coordinates. New and reused points are counted separately.
  No per-cell allocation or duplicate hash lookup is needed. Cache keys never
  approximately match coordinates; geometry changes cannot reuse this context.
- Divide only the chosen component by two. A factor-four jump is now diagnostic
  (`LCBININT_DIAGNOSTIC_ADAPTIVE_JUMP4`); grid phase can invalidate a forecasted
  error coefficient even when its asymptotic order is correct.
- Select the largest predicted error reduction per added-work estimate, using
  retained occupied-cell/row counts. Degree2 forecasts are reduced to1 if the
  child's estimated error falls too slowly. This compares error models, never
  magnification differences between resolutions.
- Fine-grid support must remain private relative to the original coarse foreign
  components. On failure retain the previous result and mark the component
  blocked. A lattice budget exhaustion also returns the last adopted result
  with `estimated_target_met=false`; no whole old-route fallback runs.

This first stage refines whole connected image components, not individual row
panels. Inside a chosen component the finer fill/discovery and LD weights are
recomputed, while lens-map values at surviving lattice points are reused. It
therefore does not yet achieve the smallest possible per-row additional work.

The state survives replacements within one call. There is no public persistent
state/resume API across calls yet; a new call repeats initialization.

## Failures that required structural changes

The first148 comparisons falsely accepted three results: a short uniform image,
a missing subcell centre image, and mixed-resolution LD cancellation.

1. A component with fewer than eight rows cannot form the endpoint stencil.
   Its curvature fallback was too small. Until that stencil is available, use
   the represented component-flux scale as its unresolved local uncertainty.
   This is a conservative scale, not a rigorous enclosure.
2. The existing centre-image witness can disappear during coarse lattice snapping.
   Retain those already solved positions and locate only a missing fine-lattice
   witness. Search a local3x3 patch at dyadic levels, then fill only that image.
   Cached augmented seeds sometimes bypassed centre metadata on a repeated call;
   recover centre witnesses from already retained positions with lens-map
   residual checks. No additional image solve is needed. If a known centre cannot
   be represented before the resolution/budget cap, report unresolved status.
3. Different component grids can upset the cancellation which made a whole-grid
   estimate accurate. For adaptive acceptance sum each component's
   `abs(signed_error)+remainder`. Continue returning the combined signed estimate
   separately (`signed_combined_indicator`). This is the triangle-inequality
   bookkeeping of local estimated errors, not a proof of their remainders.

The cheap model is consequently not a mathematical guarantee: `certified=false`
and `complete_error_model=false` remain. Arbitrary unseen finite-source image
components between existing support probes are still outside the contract.

## Final precision and work

| Absolute target | Existing profiles passed | New profiles passed | Largest actual error, existing |
|---|---:|---:|---:|
| 1e-3 | 74/74 | 9/9 | 8.70518e-4 |
| 1e-4 | 74/74 | 9/9 | 9.89941e-5 |

No falsely accepted or unresolved result remains in these final comparisons.
Largest per-call new lattice-map counts are29,120 and330,139. Across each74-profile
batch, cached lattice-map hits total139,602 and1,801,897 respectively. Some small
images reach512 locally; this is not a global nbin512 calculation.

The final ordinary fixed-grid estimator exactly reproduces all547 previous
results. Analytic flat-cache tests cover100,000 keys, rehashes, read-only misses
and signed zero. Repeated tolerance/geometry order gives identical adaptive
results; budget exhaustion retains the prior complete flux, subsequent calls
remain valid, and invalid requests are rejected. unit_core and
unit_boundary_support pass after rebuilding.

## Matched total time and remaining speed problem

One thread, Release overlay, same inputs/profile/target, seed generation through
final flux included, seven alternating batches. The normal component refinement
heuristic is enabled for ordinary fixed comparisons; it is not disabled by an
environment flag. The adaptive route substitutes model-directed refinement for
that size heuristic.

A reference-informed comparator chooses the minimum passing fixed nbin among
16,32,64,128,256,512 before timing. It is an oracle selector using the truth,
not an available production algorithm and not necessarily the fastest timed
fixed nbin when the normal component heuristic changes work.

| Target | Reference-informed fixed | Current automatic selector | Adaptive |
|---|---:|---:|---:|
| 1e-3,74 profiles | .11972 s | .35603 s,74/74 pass | .14111 s,74/74 pass |
| 1e-4,74 profiles | .15790 s | not an admissible comparison | .49680 s,74/74 pass |

At1e-3, adaptive is60.4% faster than the current automatic selector, while17.9%
slower than the reference-informed fixed choice. At1e-4, adaptive is still3.15x
that reference-informed fixed choice. No production speed claim is made there.

The existing automatic rule requires absolute-only tolerance strictly greater
than1e-4 (`kEmpiricalResolutionMinimumAbsoluteTolerance`); exactly1e-4 with
relative tolerance0 rejects all74 calls before integration. Those failures are
retained in the cost artifact. Their tiny failure-return time is not a baseline
and its computed ratio must not be used as a speed comparison.

The initial retained-cache/4x-jump/largest-error strategy took .17019 s and
1.17271 s against the same reference-informed fixed choice. Flat cache,
minimum dyadic increments and benefit-per-work priorities reduce the latter
cost by57.6%. This is a combined change, not attribution to one optimization.
An intermediate2x-only/largest-error trial spent the1,000,000-map budget on a
large component although cheaper small components could reduce the same total
error; benefit-per-work selection fixes that counterexample.

For1e-4 the principal remaining work is complete fine-component filling and
conservative per-component acceptance, rather than repeated image roots.
Row/strip-level refinement with a consistent mixed-spacing flux rule is the next
optimization; it must retain these whole-component results as a fallback and
preserve the short-image repair. Do not remove the latter to manufacture a speedup.

## API and artifacts

Experimental native entry:

`n.binary_adaptive_flux(x,y,s=s,q=q,rho=rho,limb_darkening=LD,
initial_bins=16,max_bins=512,absolute_tolerance=1e-4,max_lattice_maps=1000000)`

max_bins/initial_bins must be dyadic, maximum4096. The budget covers new lattice
maps, not boundary-refinement evaluations or every seed-stage map. All statuses
and `estimated_target_met` must be checked. Returned history records bins,
normalized acceptance estimate, cumulative new/hit map counts and selected
component. Components record bins, normalized flux, signed error, remainder,
and retained run count.

Under `tests/diagnostics/results/recal2026/`:

- `adaptive-shared-flux-audit-v5/summary.json`: final148 comparisons.
- `adaptive-shared-fresh-references-v1/summary.json` and
  `adaptive-shared-flux-fresh-v1/summary.json`: new independent references/tests.
- `adaptive-shared-flux-cost-v3/summary.json`: final timing, oracle selections,
  actual errors, automatic-selector failures, native hash.
- `adaptive-shared-flux-regression-v3/summary.json`: repeat order/budget/API tests.
- `adaptive-shared-fixed-final-check-v1/summary.json`: unchanged fixed results.
- `adaptive-shared-flux-v1/`: build logs, failed/intermediate audit outputs,
  fresh inputs, analytic cache test, source hashes/commit provenance.

Reproduce using existing myenv Python, `LCBININT_PROBE_BUILD=$PWD/build-boundary-support`,
`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`. Run the diagnostic modules
`adaptive_shared_flux_audit`, `adaptive_shared_flux_regression`, and
`adaptive_shared_flux_cost_audit` with a unique `--output`. For fresh comparisons
use `--reference-source adaptive-shared-fresh-references-v1`.
