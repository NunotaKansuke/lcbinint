# Low-resolution error audit and quadrature-aware saddle remainder

## Scope and interpretation

The user set an engineering focus on errors of order1e-3 and1e-4. These are scales for evaluating the estimator, not internal tuning constants or fixed user tolerance. Error estimation and acceptance are separate. All code changes below use analytic quadrature moments and local retained data; no tolerance-specific fitted multiplier.

Fixed low grids16,24,32,48,64,96 were audited for62 source/brightness profiles, giving372 combinations. References are existing independent numerical references; inputs include regular and difficult binary geometries, uniform and linear LD. Coefficient/resolution variants are repeated inputs, not independent occurrence samples. No runtime global nbin-doubling/convergence ladder was added.

## Low-grid baseline

| nbin | cases | min estimate/actual | max | deficit>1.1% | over2 |
|---|---:|---:|---:|---:|---:|
|16|62|.6993|126.14|14|13|
|24|62|.9887|11.56|1|8|
|32|62|.4645|18.50|4|4|
|48|62|.9889|31.52|1|10|
|64|62|.9937|2.09|0|1|
|96|62|.9641|6.09|2|3|

The maximum126 occurs at actual error3.52e-6, with estimate4.44e-4 due unresolved outer fallback, not at1e-3 actual error. Multiplicative maxima alone exaggerate some low-error cases but still identify wasted refinement. Low-grid estimation is not universally reliable: nbin16 has30% underestimation, nbin32 has54% underestimation, both pre-existing.

## What caused the remaining2.02?

At the previously reported saddle c=1,nbin96, actual6.48625269e-6 and signed model-6.48627975e-6 nearly match. Outer uncertainty6.54191149e-6 consisted of:

- saddle raw row-value fit discrepancy5.85073523e-6
- unexplained junction second differences6.80902376e-7
- smooth fourth differences9.39483742e-9
- regular LD cap remainder8.79046601e-10

Raw row-value discrepancy was counted as full flux integration error, although its second differences already contribute in the junction pass. A row-function discrepancy does not equal its quadrature defect: constant/linear errors have zero midpoint defect.

## Change

Retain |Delta²(actual-model)|/24 in the junction pass. Replace the extra raw maximum row-value discrepancy with:

1. residual fourth differences with the existing next midpoint coefficient7/5760;
2. sensitivity of the singular quadrature correction to including the two withheld rows.

For2, rank-one least-squares updates reuse the four-dimensional model system. The tlog coefficient change is multiplied by its actual whole-lattice quadrature moment. No second lens integration, new maps or image solves. This is an asymptotic numerical indicator and model sensitivity, not a certified error bound.

A residual-fourth-only trial caused two new low-grid underestimates and was rejected. The final sensitivity version slightly improves those but does not entirely eliminate them. They are:

| input s,c,bins | actual absolute error | estimate | underestimate |
|---|---:|---:|---:|
|1.3378041232,1,48|7.27734e-5|7.04358e-5|3.21%|
|1.1072850495,1,24|1.73087e-3|1.71085e-3|1.16%|

Given the user's clarified emphasis on absolute scale and decision impact, these deficits are reported explicitly rather than treated as automatic rejection by an arbitrary ratio cutoff. The candidate remains experimental, not production-certified. It must not be claimed safe for every possible tolerance: e.g. the first case permits false acceptance for absolute tolerance in[7.04358e-5,7.27734e-5).

On the previous175 conditions, maximum falls2.02036→1.19326. The previously discussed2.02 case falls1.13020. No new>1.1% underestimates appear in that set. All scalar magnifications remain bit equal; additional estimator maps0.

## Scale-based results

For descriptive order-of-magnitude bands, define1e-4 scale as actual error[3e-5,3e-4), and1e-3 scale as[3e-4,3e-3). These are reporting bins, not algorithm parameters.

Low372, absolute magnification error:

| scale | count | median estimate/actual | central80% | maximum underestimation gap |
|---|---:|---:|---:|---:|
|1e-4|190|1.0062|.9972–1.5589|2.435e-5|
|1e-3|107|1.0087|.9873–1.4646|6.901e-4|

Most predictions are close at the scales of interest; tails remain. Worst absolute1e-3-band underestimation is the pre-existing uniform s=1.3,bins16: actual.0022950 vs estimate.0016049. No claim that median accuracy ensures reliable acceptance.

Relative error (divide by absolute reference magnification), same descriptive bands:1e-4 band189 cases, median1.0083, central80%.9949–1.5609;1e-3 band30 cases, median1.0042, central80%.9754–1.3067. Absolute and relative conventions must be stated explicitly.

## Acceptance simulation: separate from estimator construction

Example tolerances1e-2,1e-3,1e-4,1e-5,1e-6,1e-7 were selected by us for diagnostics. These are not user's fixed tolerances or estimator constants.

No simulated false acceptances at these absolute tolerances in either previous175 or low372. Relative diagnostics have three false acceptances in low372, all pre-existing:

- tol1e-3: uniform s=1.3,bins16: estimate7.5328e-4 vs actual1.0772e-3;
- tol1e-5: LD s=1.2370961483,c=.5,bins32: estimate9.7618e-6 vs actual2.1015e-5;
- tol1e-5: LD s=1.912576966,c=1,bins32: estimate9.8154e-6 vs actual1.0060e-5.

Absence at these discrete tolerances is not safety at all thresholds. For every underestimating case, the interval[estimate,actual) identifies tolerances that would falsely accept. This experimental entry does not itself perform tolerance acceptance.

## Independent horizontal decomposition

Validation-only Brent boundary roots and SciPy row integration (analytic LD) on retained runs:

- uniform s=1.3,bins16: boundary-position error3.2945e-7; inner integral error near0; outer/coverage remaining.00229469.
- LD s=1.2370961483,c=.5,bins32: horizontal integral residual-7.24945e-5 vs predicted inner-7.22290e-5; outer/coverage remaining+1.14021e-4 vs predicted outer+9.10830e-5.
- discussed saddle c=1,bins96: actual inner-4.59072e-6, outer remaining-1.89557e-6; sum agrees with total actual signed error.

Thus the severe low-grid deficit is predominantly beyond the horizontal integral, not a table or LD inner-boundary issue. Outer remaining may include missed coverage, because per-retained-run quadrature does not prove complete support. A coarse/fine support diagnostic was invalidated when refined trace levels appeared; no coverage conclusion is drawn from it.

## Cost and state

True same-binary legacy/candidate comparison, five changing pilot geometries,11 alternating paired batches, bins16/32/64 andc0/.5/1. The legacy diagnostic branch skips new inverse-system columns, fourth residual and sensitivity updates. LD total timing differences range-.24% to+.39%, at the measurement-noise scale. No material slowdown detected on this workload.

Temporary eight-part diagnostic arrays/accumulation were removed from final runtime source. Before/diagnostic/final source snapshots and hashes are saved. Experimental overlay only; main master unchanged. certified=false and complete_error_model=false remain.

Final replay reproduces547 saved comparisons (175 prior plus372 low, with overlap) with identical scalar magnifications and estimator predictions to the diagnostic sensitivity build. Unit core/boundary support2/2 pass; standalone saddle analytic units pass.

## Next work

The next priority is a local validity check for outer integration at low resolution, followed by limited additional rows only where the current shape model is unsupported or misses curvature. Reuse retained rows and brackets. Do not globally raise the starting grid to64, add a global safety multiplier, or tune against tolerance thresholds. Distinguish unresolved outer fallback (can over- or underestimate) from resolved-but-inaccurate row models. Coverage remains a separate uncertainty where retained-run evidence is insufficient.

## Artifacts and reproduction

Results under tests/diagnostics/results/recal2026:

- shared-flux-low-resolution-v1:372 baseline low-grid comparisons.
- shared-flux-low-resolution-parts-v1 and shared-flux-saddle-resolution-parts-v1:temporary uncertainty attribution.
- shared-flux-low-horizontal-cause-v1:independent row decomposition.
- shared-flux-saddle-quadrature-remainder-v1:rejected fourth-only trial.
- shared-flux-saddle-sensitivity-v1 and shared-flux-saddle-sensitivity-final-v1:final547-comparison replay.
- shared-flux-saddle-remainder-cost-v1:same-binary paired costs.
- ld-low-resolution-cause-v1:source snapshots/hashes, scale/acceptance audits, build/unit logs and invalidated support-diagnostic note.

Use OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LCBININT_PROBE_BUILD=$PWD/build-boundary-support and Python /home/nunota/.miniconda3/envs/myenv/bin/python. Build target lcbinint_python test_core with-j2. Replay modules under tests.diagnostics.recal2026:shared_flux_low_resolution_audit, shared_flux_saddle_remainder_audit (both --output), shared_flux_low_horizontal_cause, saddle_remainder_cost_audit. Diagnostic-part outputs belong to temporary diagnostic source, not final runtime. Use exclusive new output names; preserve failed trials.
