# Global error triage and next step — 2026-10-13

## Recommendation

Pause full endpoint-rectangle certification. Prototype an error decomposition
for the actual one-dimensional outer integrals, including existing Bennett
weights, and keep boundary-position error as a separate term. Reuse stored row
or angular-column fluxes; avoid extra image roots and full-grid reruns.

This is a diagnostic/remainder-design step, not permission to lower nbin or
accept an unproved flux. The existing production path stays unchanged.

## Fresh bounded check

Ran original master OFF-overlay build, same saved independent VBM references,
5 near-caustic lens geometries, 2 positions each, uniform/linear c=.5, nbin16/64,
Cartesian/Polar: 80 settings per mode, 320 evaluations total. Methods were
recorded and all calls used the requested inverse-ray grid. All four modes used
the identical native SHA 9b9a28838c9f7de7c534c673edc642a5fed798b2fcfe92357b98816db0b84f58.

| Ablation | Cartesian, 40 settings | Polar, 40 settings |
| --- | --- | --- |
| Disable outer Eq. (15) | 0 values changed | 0 values changed |
| Disable Cartesian boundary-root refinement | all changed; error worsened in 32, improved in 8 | switch does not control this route |
| Disable Cartesian narrow-component refinement | 22 values changed; error worsened in 8, improved in 14 | switch does not control this route |

The misleading historical environment name DISABLE_BENNETT_LIMB actually
selects the initial interpolated boundary fraction instead of its Newton
refinement in this Cartesian run path. Bennett weights remain applied. This
ablation establishes the importance of boundary location, NOT the contribution
of its residual error after Newton refinement. Error changes are not an additive
decomposition, and a few improvements can be cancellation of separate errors.

Outer correction may still matter greatly for clean arcs: a saved independent
single-lens-limit check improved Cartesian absolute error from 4.53e-4 to
7.82e-6 when enabled. Our fresh near-caustic sample does not represent those
clean cases or the full production distribution.

Artifacts: tests/diagnostics/results/recal2026/global-error-triage-20261013-v2/.
Script, inputs/native hashes, source snapshots, route names, flags and provenance
are saved. The VBM references were reused, not recomputed. They were originally
checked at tolerances 1e-7/1e-8, including coordinate-frame checks. Their stability
is evidence, not a certified true-value interval. Timings were one warmed call
per setting and are not moving-geometry speed measurements.

## Existing evidence reviewed

1. The high-resolution seed work saved only about 2.84 ms of root time across
   48 inputs, versus roughly 1 s of Cartesian integration. Those cases queried
   about 124 million map points. Sampling work is the higher-leverage target;
   the magnitude of benefit at smaller nbin remains workload-dependent.
2. The existing single-grid outer curvature indicator covered 15/20 uniform
   errors at nbin16 and 19/20 at nbin64, but underestimated some errors. Its
   median indicator/error ratios were about 2.3 and 2.5. LD combined indicators
   covered the saved 40 errors, but median ratios were about 6.9 and 7.7.
   This separate experimental sample is useful evidence for an outer-integral
   diagnostic, not a certified bound or a matched main-master comparison.
3. Polar angular-resolution evidence already exists: a 118-row reference-qualified
   holdout improved median error from 2.14e-5 to 1.79e-5 when global grid ratio
   changed 4 to 2, at about 1.9 times median time. Near equal grid work, nbin90
   ratio2 versus nbin128 ratio4 had similar median/p90 error and 19% higher median
   time. Global angular oversampling is not the demonstrated win; localized
   angular allocation remains a candidate.
4. Signed endpoint pairing halves the typical bound on the saved two-row
   footprints. It does not supply full endpoint coverage. Larger endpoint
   rectangles still cross the actual image on their sides/bottom and have
   wide intervals. Improving those rectangles alone need not reduce required
   nbin or whole-calculation time.
5. Re-fitting the magnification-only p99 law after Bennett did not consistently
   reduce nbin: safe tight-tolerance candidates increased it. The law's limits
   and reference-censored tails must not be interpreted as formal per-evaluation
   error guarantees.

Relevant records: boundary-error-local-enclosure-report.md,
seed-runtime-cost-regression-report.md, nbin-bennett-pair-error-progress.md,
and the 20261001 handoff and improvement_probe README. All saved data are retained.

## Design direction

For each computation distinguish:

    boundary-position error
    inner/radial brightness quadrature error
    outer row/angular quadrature error
    non-smooth/topology-transition unresolved contribution.

For uniform brightness, geometric area does not require a brightness model.
For LD, the singular limb derivative requires its own basis/remainder; the
boundary brightness-model residual alone is not a geometric error estimate.

Work with the actual flux function F(y) or F(phi) and the actual weights Q.
The useful mathematical object is E[F] = Q[F] - integral(F), not a separately
large area enclosure around every sample. Determine which low-order or limb
basis terms the active stencil integrates exactly; then E[F] = E[F-p] for
those verified terms. Preserve signed contributions within a justified smooth
segment before taking an absolute bound. Use derivative remainder bounds for
that residual, with explicit unresolved handling at folds/cusps/junctions.

This is analogous to estimating the remainder after the parabolic correction
in Bozza's contour integrator, but the coordinate, quadrature and singularities
are different: its estimator cannot be copied as a proof for inverse rays.

Literature checked:
- Bennett 2010, https://arxiv.org/html/0911.2703 (limb-specialized radial rule
  and anisotropic integration grid).
- Bozza 2010, https://arxiv.org/abs/1004.2796 (parabolic correction and error
  control designed for its contour integral).

## Concrete next experiment

1. Cartesian first: instrument existing row fluxes and boundary brackets,
   with production-equivalent weights/refinement and no extra root solves.
   Form candidates for signed outer residual plus boundary-position uncertainty.
   Do not assert that second differences alone are derivative bounds.
2. Use a small matrix spanning a clean arc, fold, cusp, resonant and small
   planetary caustic, including limb contact and both brightness profiles.
   Compare independent references, underestimation, overestimation and estimator
   plus integration time. The investigation may use saved resolution ladders;
   the runtime estimator must not use Richardson/repeated full-grid checks.
3. Decide which derivative remainder is worth certifying. Add interval bounds
   only to the selected residual terms/uncertain local regions; retain earlier
   work for targeted refinement. If those bounds cost more than the grid work
   they save, reject that implementation rather than making proof the objective.
4. Carry the successful outer-residual idea to Polar's angular columns. Then
   consider allocating radial and angular/local resolution separately instead
   of increasing the global nbin or global polar ratio.
5. Only after valid total-error acceptance exists, lower initial nbin. Initial
   geometry-aware guesses and final acceptance remain distinct roles. No empirical
   safety multiplier is promoted to a mathematical guarantee.

This next step changes the error-estimation target, not the integration backend.
Complete global flux certification, useful lower nbin, and measured total speedup
remain unachieved. Routing/point-source/hexadecapole errors are outside this
forced-inverse-ray audit; it is not a full light-curve pipeline error decomposition.
