# Retained row cores and local boundary strips

## Result

The experimental directional adaptive path now retains coarse row interiors and refines horizontal boundary strips. Smooth intermediate rows reuse six neighboring rows. Production/default routing is unchanged. Both uniform and linear LD share the same retained-run machinery. Initial16, max512, axis ratio4, map budget1e6. All166 independent-reference calls pass (148 main,18 separate geometries), with no false acceptance. This is finite-matrix evidence, not a general error guarantee.

| tolerance | profile | local / previous adaptive | local / fixed oracle | new maps / previous |
|---|---|---:|---:|---:|
| 0.001 | c=0 | 0.975 | 1.120 | 0.775 |
| 0.001 | c=0.5 | 1.004 | 1.112 | 0.943 |
| 0.001 | c=1 | 1.028 | 1.223 | 0.927 |
| 0.0001 | c=0 | 0.914 | 1.864 | 0.848 |
| 0.0001 | c=0.5 | 0.970 | 1.908 | 0.819 |
| 0.0001 | c=1 | 0.905 | 1.621 | 0.752 |

Timing: one thread, nine rotated alternating batches per profile; geometry changes between adjacent calls. Full scalar calls including seeds, integration, adaptation, indicators and result conversion. Fixed oracle uses independently reference-selected minimum passing nbin, selection cost excluded. Profiles contain27/25/22 geometries and are not identical paired sets. No compilation or other audit during timings.

| tolerance | fixed oracle seconds | previous adaptive seconds | local seconds | aggregate local / fixed |
|---|---:|---:|---:|---:|
| 0.001 | 0.106688 | 0.122386 | 0.122464 | 1.148 |
| 0.0001 | 0.159868 | 0.311388 | 0.289225 | 1.809 |

Aggregates are sums of profile medians, not a combined-batch median. At1e-4 aggregate runtime falls7.1%, fixed ratio1.948 ->1.809. At1e-3 aggregate unchanged within0.1%; pure LD profile slows2.8%. Fixed-like high-precision runtime remains unresolved.

## Retained horizontal core

A completed coarse row supplies the integral between two interior anchors, leaving four coarse cells on each boundary strip. The immutable core stores its physical endpoints, spacing, trapezoid integral, endpoint derivative differences and uncertainty. Only the boundary strips are resampled when x resolution increases. The core survives subsequent adopted refinements; y-only reuse changes its physical row weight, not its horizontal integral.

For coarse spacing hc and fine spacing hf, use the Euler-Maclaurin difference:

T_f ≈ T_c − (hc²−hf²)(g′(b)−g′(a))/12 + (hc⁴−hf⁴)(g‴(b)−g‴(a))/720.

Seven already evaluated coarse points at each anchor estimate derivatives; five-vs-seven point differences and the next Euler-Maclaurin term supply positive uncertainty. That uncertainty participates in the normal acceptance sum. This is a smooth-core model, not a bound on arbitrary lens configurations. Unsupported/short/unresolved cores use the ordinary scan, reusing their already evaluated lattice points. LD d!=0 uses the ordinary scan.

## Narrow support check: the missed-gap failure

Before the core can be reused, confirm its complete horizontal interval lies inside the source disk. Linear interpolation of the lens map over a coarse interval has norm defect at most hc² sup|f″|/8. For the binary map, sup|f″| ≤2Σm_j/d_j³, with d_j the minimum image-plane distance from that core to lens j. Thus b=hc²Σm_j/(4d_j³). Cached coarse source-distance squares must all satisfy maxD² +2rho*b+b² <rho². This uses convexity of the source disk. The bound is computed once per immutable core. Floating evaluations use a roundoff margin, not a complete outward-interval implementation; certified and complete_error_model remain false.

The first strip prototype missed an interior source-outside gap between coarse samples. Untimed complete fine-row checks found two uniform gap rows and one c=.5 gap row. One uniform case falsely accepted at1e-4 (actual1.0010554469230826e-4 in strips-only,1.0024190688673329e-4 with interpolation). The support check rejects the affected cores. This is a structural fix, with no per-case exception or fitted tolerance multiplier.

## Intermediate rows: remove Bennett grid phase before interpolation

Interpolating the raw Bennett-corrected row is invalid for LD: changes in endpoint phase, including the guarded delta_c rule, are not smooth geometry changes. The first raw-row interpolation had large fine-row differences even though all global tolerance checks happened to pass.

For old row contribution S, width W and horizontal signed error E in physical flux units, interpolate the smooth moment M=S+E/(hx*hy)−(1−c)W. On the new row, evaluate the actual endpoint fractions and model residual Enew, then reconstruct Snew=Minterp+(1−c)Wnew−Enew/(hx*hy). Carry old positive model uncertainty and the local cubic fourth-difference defect; add new endpoint uncertainty. Uniform is the zero-LD limit, for which the endpoint-model defect vanishes. Tip/gap/junction stencils and simultaneous x changes keep ordinary scans. The interpolated row is tagged as approximate, so a later core cannot infer an exact coarse interior from it.

The maximum LD fine-row difference fell from5.4345e-5 to7.2997e-7 (c=.5) and9.1734e-5 to8.0388e-7 (c=1). These are single-row flux differences normalized by source flux, not global reference errors or rigorous bounds.

## Validation

- Main148 + separate18 independent saved VBM/radial-LD references: all estimated targets met and actual errors pass; no failure omitted.
- Final untimed full-row audit (same native hash as final timing): all accepted retained cores and interpolated rows checked on their complete fine lattice.
- c=0: 3176 core rows, 13371 interpolated rows; gaps 0.
- c=0.5: 3724 core rows, 5362 interpolated rows; gaps 0.
- c=1: 6415 core rows, 783 interpolated rows; gaps 0.
- Unchanged routes: 148 square adaptive and 547 fixed values/indicators exactly reproduce saved results.
- Two native unit suites pass; polynomial tests confirm cubic interpolation, quartic leading defect, derivative stencils, and Euler-Maclaurin fine-trapezoid equivalence through degree5.
- Both baseline and local budget exhaustion retain the previously adopted positive flux. A subsequent ample-budget call exactly reproduces the full result, with no leaked request state.

## Rejected variants and remaining work

- Omitting entire horizontal rows below a small per-row error share retained only a few rows and was slower for pure LD. Its diagnostic path is preserved, not selected.
- A sample-count setup gate was conservative: it removed useful core reuse and reduced the speed gains. LCBININT_DIAGNOSTIC_STRIP_SETUP_GATE stays unset.
- Vertical interpolation under a simultaneous x-grid change is disabled; the retained data belong to the old x spacing. Those new rows use normal evaluation.
- All rejected runs and the missed-gap counterexample remain in result directories. No default route promotion, commit or push.
- The vertical lattice still grows over an entire selected component. Even reused/interpolated rows must be registered and traverse the fill machinery. Reaching fixed-like1e-4 cost requires truly local outer-row panels with consistent Bennett endpoint weights, rather than relaxing error factors or thinning global grid axes.

## Artifacts

Final: tests/diagnostics/results/recal2026/local-row-interpolation-v10/{summary.json,accuracy.json,regression.json,local-budget-regression.json,ctest.log,polynomial-test.log,provenance.json,source/}. Full-row audit: boundary-strip-difference-v4/summary.json. Earlier variants: local-row-interpolation-v1..v9, local-boundary-strips-only-v1, boundary-strip-difference-v1..v3. Scripts: local_row_interpolation_audit.py and boundary_strip_difference_probe.py. Final selected flags and source/native SHA256 are in provenance.json.
