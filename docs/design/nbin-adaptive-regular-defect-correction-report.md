# Adaptive regular-defect correction: final numerical result

## Result

Selected candidate: initial nbin16, maximum512 on each axis, original seeds, retained directional/row/strip/cap integration, regular model-defect correction. Final artifact is `adaptive-defect-correction-v13`. All406 geometry/profile/tolerance combinations pass their independent absolute-error target; zero false acceptance. This is empirical validation, not a complete mathematical certificate.

This changes the numerical flux by applying a same-lattice model correction. The preceding cache optimization preserved all fluxes; this change deliberately increases approximation accuracy at a given grid. It does not loosen the requested tolerance or use references at runtime.

## Matched full-call timing

21 rotated alternating batches per profile, one thread, changing adjacent geometry including batch wrap; initial nbin16 and absolute1e-3/1e-4. Full scalar calls include normal seeds, fill, corrections, estimation/scheduling and result construction. Compilation and validation runs are excluded. The27/25/22 timed profile sets are matched within profile.

| tolerance | LD c | new / previous adaptive | time reduction | previous / fixed oracle | new / fixed oracle |
|---|---:|---:|---:|---:|---:|
|1e-03|0|0.955|4.5%|1.054|1.007|
|1e-03|0.5|0.955|4.5%|1.083|1.034|
|1e-03|1|0.891|10.9%|1.173|1.045|
|1e-04|0|0.461|53.9%|1.542|0.711|
|1e-04|0.5|0.536|46.4%|1.630|0.873|
|1e-04|1|0.594|40.6%|1.460|0.868|

Sums of the three profile medians (not a median of a combined batch):

- 1e-03: previous0.116794s ->new0.109184s, 6.52% shorter; fixed ratio1.099->1.027.
- 1e-04: previous0.248684s ->new0.129416s, 47.96% shorter; fixed ratio1.546->0.804.

Fixed oracle is the independent-reference-selected minimum passing *original fixed-grid* nbin among16..512, with selection cost excluded. It is not the production automatic selector. Inputs, tolerances and references match; the new adaptive numerical approximation includes additional model corrections.

LD follows I(r)=1-c+c sqrt(1-r^2/rho^2), normalized by pi*rho^2*(1-c/3). Uniform is c=0; c=.5 has nonzero limb brightness; c=1 is the square-root-only profile.

## Why another refinement was being requested

The saved preceding histories show post-reference-passing steps predominantly driven by signed endpoint defects rather than positive remainder alone: uniform15 outer-signed steps and1 outer-remainder step; c=.5 six outer-signed and six inner-signed plus1 outer-remainder; c=1 nine inner-signed, five outer-signed,1 outer-remainder and1 short-component step at1e-4 on the main matrix. Truth is consulted retrospectively for diagnosis only.

For raw flux Q and true flux I, the model separates I-Q = S_regular + S_junction + epsilon. We compute Q_corrected=Q+S_regular, then retain sum_component(|S_junction|+R) as the acceptance indicator. Component signs are not cancelled against each other. If a correction would create a nonpositive/nonfinite component flux, retain the raw flux and its full signed indicator.

This removes a predicted leading defect by improving the computed flux; it does not merely declare the same raw error estimate smaller. All lens-map values, roots, completed row contributions and fallback work remain retained.

## Structural fixes needed at coarse resolution

### Three-node horizontal endpoint model

For g(t)=sqrt(t)*u(t), two inside nodes fit a linear u. A third already available lattice node supplies the quadratic interpolation term. Its exact power-moment defect is added to the signed correction, and the difference from the two-node model remains positive uncertainty. Short rows lacking the third node retain the unresolved contribution in uncertainty. This is a same-grid local fit, not global Richardson convergence or extra lens-equation solving.

### One-node LD rows were missing a meaningful horizontal error

Independent continuous horizontal-slice quadrature exposed one-node rows with a nonzero true error but zero recorded signed defect/remainder. Separate endpoint expansions have no second sample in this case.

Use actual left/right root fractions l,r and source depth D(0) at the one retained node. The finite parabola is P(t)=D(0)*(t+l)*(r-t)/(l*r), and its LD integral is (1-c)*(l+r)+c*sqrt(D(0)/(l*r))*pi*(l+r)^2/8.

Let M bound |D third derivative| in lattice coordinates over the entire interval. Interpolation gives |D-P| <= M*|(t+l)*t*(t-r)|/6. If D(0)/(l*r)>M*max(l,r)/6, positivity throughout the interval follows. Using |sqrt(a)-sqrt(b)|<=sqrt(|a-b|), the integrated LD perturbation is bounded in exact real arithmetic by c*pi*(l+r)^2/8*sqrt(M*max(l,r)/6).

M is calculated cheaply from the binary lens map: |D third derivative| <=2*(F0*F3+3*F1*F2)/rho^2, with Fj bounds on map derivatives over the interval and F0 from the retained node plus F1 times the interval extent. No extra lens solve is used. The native implementation uses floating root fractions and ordinary floating bounds with a roundoff allowance, so it is not an outward-rounded certificate. An invalid/pole-crossing bound retains a finite full-row brightness allowance and ordinary incremental refinement.

### Apply the outer rule to the corrected row function

The y model sees raw_row + signed_horizontal_error/(hx*hy); otherwise separately applying x/y defects retains a mixed tensor error. The LD cap includes its third integer-power term, with the fourth row held out. Fifth/sixth retained width rows check the next squared-width term. All use already stored data.

### Critical/junction models remain unresolved error, not a correction

Full correction of junction/saddle estimates falsely accepted new coarse cases. These models therefore continue to drive refinement. On short critical footprints, two four-row endpoint stencils overlap. A finite closed-cap fit cannot validate two independent infinite-tail corrections.

The closed-cap defect uses Integral(model)-Sum(model). Using Integral(model)-Sum(data) leaves a fit-residual sum in the signed estimate while the curvature allowance covers only its quadrature defect. Model family/root-footprint checks retain uncertainty; invalid finite fits keep local zero-halo curvature instead of applying overlapping endpoint expansions.

A short 1->2->1 row-multiplicity footprint may be an image arm ending, rather than a closed gap. The gap graph alone missed this case. Retain its already available zero-halo geometric curvature allowance while the stencils overlap (3..7 rows). The allowance replaces the accounted junction uncertainty via a maximum; it is not added again to the same term. Only that component receives needed rows, and earlier maps/rows remain available. This inexpensive sampling rule is a conservative model guard, not a general mathematical bound.

## Accuracy and verification

- Original main148 plus separate18 saved-reference conditions, plus two new120-condition matrices:406/406 actual target passes,406 estimated target met, zero false acceptance.
- Each new matrix has12 changed geometries, LD c=0/.2/.5/.8/1 and both tolerances. References use independent uniform VBM areas and SciPy radial quadrature at two accuracy settings. Final new-matrix reference variation is at most3.99e-12. All rows and warnings are retained.
- 35 budget/accounting/repeat checks: preserve positive last-adopted corrected flux, component flux/indicator sums, correction bookkeeping and request isolation. Unsupported nonlinear LD retains the pre-existing API rejection.
- 148 unchanged square adaptive and547 fixed results/indicators exactly reproduce saved values;148 directional histories checked.
- Native unit_core and unit_boundary_support pass. Separate assertions check finite one-node integrals for parabolic/cubic depths by independent quadrature, integer moments through degree4 by finite polynomial sums, half-power degree3 moments, valid/invalid closed-gap fallback behavior.
- Independent exact horizontal-slice checks confirm the finite one-node row errors lie inside their local uncertainty on the diagnosed configurations.

Worst actual errors across all406 conditions at1e-4:

| LD c | maximum actual absolute error |
|---|---:|
|0|9.3306348e-05|
|0.2|5.8896403e-05|
|0.5|4.9047202e-05|
|0.8|3.7919728e-05|
|1|5.932421e-05|

Pointwise indicator underestimation still occurs:9/203 at1e-3 and7/203 at1e-4, all with actual errors below the requested target. The saved `absolute-error-diagnostics.json` distinguishes these from false acceptance and from errors smaller than one tenth of tolerance. Do not interpret a factor ratio alone as a tolerance failure.

The initial unconstrained correction passed much of the saved matrix but failed independent new configurations. v8..v12 still had one1e-3 case at actual0.0010023845 (0.24% over target); these were not accepted as final. The row-multiplicity guard resolves it in v13. All failed variant outputs/logs are preserved.

## Selected path and remaining scope

Experimental opt-in:

```
LCBININT_DIAGNOSTIC_DIRECTIONAL_ADAPTIVE=1
LCBININT_DIAGNOSTIC_LOCAL_ROW_INTERPOLATION=1
LCBININT_DIAGNOSTIC_LOCAL_BOUNDARY_STRIPS=1
LCBININT_DIAGNOSTIC_LOCAL_VERTICAL_CAPS=1
LCBININT_DIAGNOSTIC_DEFECT_CORRECTION=1
```

Production/default routing and seed/discriminant selection remain unchanged. The native result exposes applied_flux_correction and raw_magnification; components contain corrected flux and the remaining signed residual. certified=false and complete_error_model=false remain. Validated scope is binary Cartesian uniform/linear LD with absolute1e-3/1e-4 targets; stricter tolerances and general mathematical certification are not established.

This version was frozen as the final selected numerical implementation at the user request on 2026-10-13. No further packaging or model changes are included in this freeze. The selected switches above define the final path; merging or promoting it to the main branch is deferred.

Artifacts: `tests/diagnostics/results/recal2026/adaptive-defect-correction-v13/` (final timings, accuracy, regressions, provenance and source snapshot), `adaptive-defect-fresh-v1/`, `adaptive-defect-fresh-v2/` (independent references), `adaptive-defect-row-cause-v1/` and `adaptive-defect-row-cause-v3/` (independent row diagnosis).

Reproduce into a new output directory:

```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LCBININT_PROBE_BUILD=$PWD/build-boundary-support /home/nunota/.miniconda3/envs/myenv/bin/python -m tests.diagnostics.recal2026.adaptive_defect_audit --output UNIQUE_OUTPUT --fresh tests/diagnostics/results/recal2026/adaptive-defect-fresh-v1/summary.json --fresh tests/diagnostics/results/recal2026/adaptive-defect-fresh-v2/summary.json --timing
```
