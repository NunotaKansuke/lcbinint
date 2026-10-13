# Local vertical panels: retained-row replay

## Result and status

A shared uniform/linear-LD local vertical quadrature prototype is implemented in `tests/diagnostics/recal2026/local_vertical_panel_replay.py`. Native diagnostic extraction provides physical raw row integrals and horizontal boundary errors for each adopted adaptive stage. Native flux, estimator, lattice-map counts, component records and history reproduce the previous selected implementation exactly for all 166 saved independent-reference calls, even with extraction enabled.

The prototype preserves eight existing rows at each physical tip and all rows in gaps/junctions or unresolved horizontal boundaries. Only contiguous, single-interval interior bands use adaptive dyadic panels. The normal Bennett quadrature at the tips is retained. The smooth row integral is raw horizontal integral plus its signed horizontal residual divided by row spacing, removing the horizontal Bennett grid phase before interpolation.

This is an **offline replay**, not a native mixed-grid integration path. All fine rows have already been computed to supply replay samples. It measures potential demand for interior row values, not saved lens maps, total traversal cost or wall-clock acceleration. Tip rows and ineligible components are additional costs not included in the core-node ratio. This remains an uncertified local smoothness model, with no improvement to image-component coverage.

## Budget correction

The first trial assigned one eighth of tolerance to local panels in addition to the previous native error. Two calls at 1e-4 exceeded actual tolerance: uniform maximum1.03002496e-4, pure LD maximum1.08813984e-4. Neither was falsely accepted by the combined estimator. All failures remain in `local-vertical-panels-v1/replay.json`.

The second trial uses only available slack: max(0,tolerance-native_indicator)/(8*component_count). It refines only the panel with largest local defect, retaining previously requested nodes. At the finest saved spacing it falls back to the original trapezoidal integral in that interval. It does not restart the component or discard its samples. No reference value enters scheduling or admission.

The local indicator combines the embedded Simpson defect |S_fine-S_coarse|/15 with spacing²|right slope-left slope|/12 for replacing the saved lattice trapezoid. This compares quadratures within one local panel; it never checks convergence of whole-source magnifications at successive nbin. Neither term is a general upper bound near arbitrary unresolved structure.

Second trial:166/166 actual errors pass,166/166 estimated targets met, zero false acceptance on saved main148 and separate18 cases. This is finite-matrix evidence, not a general certificate.

| tolerance | profile | eligible calls / total | interior nodes / full core rows | potential core-row reduction |
|---|---|---:|---:|---:|
| 1e-3 | uniform |30/30|3866/6238|38.0%|
| 1e-3 | c=.5 |24/28|2649/2921|9.3%|
| 1e-3 | c=1 |24/25|2593/2849|9.0%|
| 1e-4 | uniform |30/30|16388/26508|38.2%|
| 1e-4 | c=.5 |25/28|9184/10692|14.1%|
| 1e-4 | c=1 |25/25|4695/4871|3.6%|

The safe slack rule sharply reduces LD savings. Simply adding another error allocation to the existing estimate duplicates caution rather than replacing the vertical contribution already being estimated. The next native design needs separate accounting for retained physical caps and replaced interior quadrature; dropping the previous whole vertical indicator without that separation would be unjustified. Increasing a heuristic safety factor is not the proposed solution.

## Verification and artifacts

- Constant through cubic analytic integrals, quartic defect, zero-budget equivalence to the retained lattice and non-dyadic tail checked in `test_local_vertical_panels.py`.
- Native `unit_core` and `unit_boundary_support`:2/2 pass.
- Diagnostic row extraction is opt-in via `LCBININT_DIAGNOSTIC_ROW_PANEL_TRACE`; no row-export allocations in ordinary calls.
- `local-vertical-panels-v1/{build.log,ctest.log,replay.json,summary.json}` retains initial failures.
- `local-vertical-panels-v3/{replay.json,summary.json,source/}` retains the final revised matrix and source/native hashes; v2 retains the preceding equivalent accumulator variant. No native-speed claim or default promotion.
- The next implementation should evaluate requested interior rows directly, store panels independently of the uniform run lattice, and keep endpoint weights at physical tips. It must preserve old row samples on budget exhaustion and account for mixed spacing in component privacy checks. Reusing the old uniform-grid outer estimator on mixed rows is invalid.
