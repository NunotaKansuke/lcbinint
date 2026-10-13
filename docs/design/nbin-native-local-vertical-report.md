# Native retained vertical cores and physical-cap refinement

## Result

Implemented and measured the live binary Cartesian adaptive route, starting from initial nbin16, maximum512 on each axis, map budget1e6, axis ratio4. This is no longer saved-row replay. An eligible component retains its previously computed coarse interior integral and refines only its two physical caps. Uniform and linear LD share the same code. Experimental opt-in: `LCBININT_DIAGNOSTIC_LOCAL_VERTICAL_CAPS=1` together with the existing directional, row-interpolation and boundary-strip flags. Production/default routing and the seed/discriminant selection are unchanged.

166/166 actual errors pass saved independent references,166/166 estimated targets are met, zero false acceptance (148 main +18 separate-geometry calls). At1e-4, worst actual errors including separate cases: uniform9.12390e-5, c=.5 9.53495e-5, c=1 9.86504e-5. This finite-matrix result is not a general mathematical certificate; `certified=false`, `complete_error_model=false` remain appropriate.

## Matched full-call runtime

Baseline is the previously selected adaptive algorithm: directional refinement + smooth intermediate-row interpolation + horizontal boundary strips. Fixed comparator is the reference-selected minimum passing fixed nbin from16,32,64,128,256,512, with the original component heuristic; selection cost is excluded. It is an optimistic fixed oracle, not the production automatic nbin selector.

One thread, Release build, nine rotated alternating batches for fixed/baseline/local per profile. All inputs and tolerances match within each comparison. Every adjacent input changes(s,q), including batch wrap. Full scalar calls include seeds, integration, adaptive scheduling, indicators and Python result conversion. No builds or numerical audits run concurrently with timing. Counts per profile are27/25/22 distinct main geometries at each tolerance; profile geometry sets differ.

| tolerance | LD c | new / baseline | time reduction | new / fixed oracle | accepted cap trials / attempted |
|---|---:|---:|---:|---:|---:|
| 1e-03 | 0 | 0.9651 | +3.49% | 1.0780 | 15/16 |
| 1e-03 | 0.5 | 1.0045 | -0.45% | 1.1134 | 2/3 |
| 1e-03 | 1 | 0.9989 | +0.11% | 1.2158 | 0/0 |
| 1e-04 | 0 | 0.8720 | +12.80% | 1.6678 | 13/65 |
| 1e-04 | 0.5 | 0.9559 | +4.41% | 1.8076 | 8/10 |
| 1e-04 | 1 | 1.0030 | -0.30% | 1.6142 | 0/4 |

Aggregate is the sum of three profile medians, not the median of a combined batch:

| tolerance | fixed seconds | baseline seconds | new seconds | reduction | baseline/fixed → new/fixed |
|---|---:|---:|---:|---:|---:|
|1e-3|0.107598|0.123071|0.121670|1.14%|1.1438 →1.1308|
|1e-4|0.161796|0.294533|0.274312|6.87%|1.8204 →1.6954|

At1e-4 uniform improves12.8%, c=.5 improves4.4%, pure LD slows0.3% and is effectively unchanged at this scale. Fixed-like1e-4 latency remains unresolved. The earlier replay core-node savings must not be equated with full-call acceleration.

## Integration and error accounting

Eligible components have at least25 contiguous, overlapping single-interval rows and no unresolved horizontal boundary. Refinement must be y-only. Horizontal error must already consume no more than half the component's allocated target. Four coarse rows at each physical cap become a fine band; the retained core uses trapezoid half weights at its two interfaces. Seven cached smooth row integrals at each interface provide first/third derivatives and their five-versus-seven-stencil variation. Smooth values remove the old horizontal Bennett phase: F_j = hx·raw_row_j + signed_horizontal_error_j / hy.

Fine bands use the existing row integration and actual Bennett physical-tip correction. Per-tip applicability is inherited from the parent; aspect tests on an artificially shortened band cannot silently change that selection. A clipped core/band interface is explicitly excluded from physical-tip correction and cap error fitting. Unresolved masked cap fits fail closed with infinite uncertainty, rather than using a curvature fallback containing the artificial endpoint.

For coarse core spacing hc and fine cap spacing hf, the shared interface residual is

(hc²−hf²)(F′lower−F′upper)/12 − (hc⁴−hf⁴)(F‴lower−F‴upper)/720.

The uncertainty contains derivative-stencil variation, propagation of the cached horizontal-row uncertainty through the derivative coefficients, and the first omitted Euler-Maclaurin term. This is an asymptotic smoothness model, not a rigorous bound on arbitrary interior derivatives. Horizontal signed residuals and positive remainders are reweighted with the actual coarse/fine row widths. The old whole-component vertical estimate is replaced by the physical-cap estimates plus the retained-core/interface model, not added a second time.

Actual admission still requires abs(sum signed residuals)+sum positive remainders≤the allocated local target. A successful local candidate may be revisited if other components keep the global total above tolerance. Its old support representation and fine cap data remain available; it is not permanently blocked merely because a local target was met. Diagnostics `mixed_component_grids` distinguish horizontal bins, coarse core y bins and fine cap y bins; component maximum bins include the cap resolution.

## Incremental fallback and clip extension

- Failed cap candidates retain mapped points and their completed raw row integrals, boundary residuals, fractions and core metadata. The ensuing ordinary refinement checks those exact fine rows first, then the old coarse/interpolated data, and computes only missing work. Raw stored rows have no outer cap weights baked in, avoiding double application.
- If the physical cap reaches the outer clip edge, extend that edge by four fine rows. Reuse the prior band before adding the missing rows. Only the artificial interface remains clipped. Checked integer arithmetic guards the extension.
- If a band splits into multiple components, its completed rows are retained for ordinary refinement; the single-core candidate is rejected.
- Budget exhaustion leaves the last adopted flux and error intact. There is no new cross-call resume API.

Forced-extension tests on uniform/c=.5/c=1 exercised8/4/2 extensions and146/36/37 reused rows. Magnification and indicator reproduced the default-padding calculation exactly in all three cases. Invalid diagnostic padding rejects cleanly, and a later default call shows no leaked request state. Default timing uses padding4 and does not enable the forced-extension probe.

## Why LD gains remain limited

On the baseline1e-4 main matrix:

| c | all refinements | y-only | fewer than25 rows | horizontal error exceeds local half-share | eligible before core smoothness checks |
|---|---:|---:|---:|---:|---:|
|0|213|126|47|0|79|
|.5|190|84|41|29|14|
|1|145|33|24|3|6|

Most pure-LD work is x refinement or simultaneous x/y refinement. Most remaining y-only components are short; six reach the core smoothness checks, and no pure-LD cap trial is accepted in the timed main matrix. Vertical localization alone therefore cannot yield the replay's apparent large general LD saving. Further effort should focus on horizontal/interior LD moment work and its allocation, preserving the same shared integration/error framework.

## Validation and rejected variants

- All166 independent-reference calls pass; all failures from previous experiments remain saved.
- Unchanged routes:148 square-adaptive and547 fixed results/indicators reproduce saved values exactly. Directional history and original budget behavior validate.
- New partial-cap budget tests across uniform and both LD profiles preserve the last adopted positive flux. Ample-budget repeats reproduce magnification, indicator, mapped-point counts, components, history and cap counters exactly.
- Native unit_core/unit_boundary_support:2/2 pass.
- Mixed-spacing interface correction reproduces fine-trapezoid equivalence for constant through quintic analytic polynomials.
- Retaining eight instead of four coarse cap rows was safe but less useful for c=.5; the smaller band uses the same derivative/cap-stencil requirements across profiles.
- A scheduling gate based on h^(3/2) cap scaling removed useful candidates and reduced measured benefits. It was rejected; final admission uses actually recomputed cap data.
- A local candidate is not allowed to freeze an unaccepted global sum. Failed work is reused instead of restarting a legacy calculation.

## Artifacts and reproduction

Final directory: `tests/diagnostics/results/recal2026/native-local-vertical-v8/` contains summary/accuracy, all nine timing samples, native/source hashes and source snapshots, native regression, budget regression, forced-extension regression, eligibility explanation, interface test and unit logs. v1/v2/v4/v5/v6/v7 retain earlier measured variants; v3 was a build stage.

```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LCBININT_PROBE_BUILD=$PWD/build-boundary-support /home/nunota/.miniconda3/envs/myenv/bin/python -m tests.diagnostics.recal2026.native_local_vertical_audit --output UNIQUE_RESULT_NAME
```

Source/build identity and thread settings are in provenance.json. No default promotion, commit, push or new worktree.
