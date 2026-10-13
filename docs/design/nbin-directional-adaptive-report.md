# Directional adaptive refinement with retained rows

## Result

Implemented an experimental component-level directional adaptive path, with exact reuse of existing horizontal row contributions when only vertical spacing changes. It is **not yet arbitrary boundary-panel refinement**: the selected component still receives new samples along the selected direction. Production/fixed routing and the default experimental square-adaptive method remain unchanged.

Final main matrix: 74 geometry/profile combinations at each of absolute targets 1e-3 and 1e-4, 148 calls. Independent saved VBM uniform/radial-LD references: 148/148 actual passes and 148/148 estimated passes, no false accepts. Separate three-geometry, nine-profile matrix: 18/18 actual and estimated passes. These finite matrices do not establish a general guarantee; `certified=false`, `complete_error_model=false` remain unchanged.

## Matched timing, changing geometry every call

Seven alternating timing batches, one thread, scalar end-to-end calls including seeds, all integration/refinement, error evaluation and native result conversion. Geometry changes between adjacent calls inside each profile batch. Each profile uses the same inputs, tolerance, precision and native binary for all three methods. Profile sets differ: uniform27, c=0.5 has25, c=1 has22.

The fixed comparator chooses the smallest passing nbin among16,32,64,128,256,512 using independent reference truth, then excludes selection cost from timing. It retains the ordinary component-refinement heuristic. This is a reference-informed oracle comparator, not a deployable initial selector or a rigorous fastest-fixed lower bound.

| Target | Profile | Time / square adaptive | Time / reference-informed fixed |
|---|---|---:|---:|
| 1e-3 | uniform | 0.898 | 1.150 |
| 1e-3 | LD c=0.5 | 0.938 | 1.107 |
| 1e-3 | LD c=1 | 0.873 | 1.186 |
| 1e-4 | uniform | 0.530 | 2.179 |
| 1e-4 | LD c=0.5 | 0.774 | 1.990 |
| 1e-4 | LD c=1 | 0.614 | 1.828 |

Sum of the three profile medians, explicitly not a median of a single combined batch:

- 1e-3: fixed0.107642s; square0.136726s; directional0.123399s. Directional/fixed1.146, directional/square0.903.
- 1e-4: fixed0.151732s; square0.500637s; directional0.306196s. Directional/fixed2.018, directional/square0.612.

Thus the target of fixed-comparable speed is approached at1e-3 but **not met at1e-4**, where approximately a factor of two remains. Whole74-case batches in `directional-adaptive-cost-v2` interleave profiles and can reuse same-geometry seed state, so use the profile batches above for the user's changing-geometry workload. At1e-4 the ordinary empirical automatic selector rejects all74 configurations before integration; that failure timing is not a valid speed comparator.

At1e-4 new lattice maps relative to square adaptive fell by60.8% uniform,39.9% c=0.5,51.0% c=1. Maximum actual errors were9.124e-5,9.541e-5,8.702e-5, respectively. At1e-3 maxima were9.150e-4,7.847e-4,7.821e-4. Speed was not obtained by loosening acceptance.

## Algorithm and shared mathematics

A retained component now has independent horizontal and vertical source-bin counts. Physical coordinates use hx=rho/bins_x, hy=rho/bins_y, and area/error terms use hx*hy. Horizontal Bennett crossing fractions use hx; vertical boundary fractions use hy. The common uniform/linear-LD kernel and error decomposition are retained.

The heuristic uses already available inner/horizontal and outer/vertical residuals and remainders:

1. If one directional magnitude exceeds twice the other, double only that direction.
2. If neither dominates, double both.
3. A short-component guard indicates insufficient rows for the outer stencil, so prefer vertical refinement.
4. If the previous same-direction refinement reduced the error poorly (existing estimator-order fallback), add both directions next.
5. Keep the axis resolution ratio at most4, and respect max512 separately on each axis.

This is a general directional policy, not a lens-configuration lookup table. The existing error-reduction-per-predicted-work score still chooses the component. No whole-grid magnification-difference/Richardson stopping criterion is used.

Existing inside run seeds are lifted with separate x/y factors; no additional image quintic solves are performed for refinement. Footprint privacy projects each axis separately. Small-image repair remains incremental and uses already solved center witnesses.

### Exact row reuse

If hx is unchanged, old rows lie at identical physical positions with identical horizontal crossings, brightness sums and Bennett horizontal corrections. Reuse their retained maximal runs, `run_areas`, horizontal error terms and crossing fractions. Horizontal physical-flux errors scale by new_hy/old_hy; outer weights and outer error estimation are rebuilt on the new row lattice. New rows are explored normally. The run registry/frontier still processes reused runs, preserving connectivity traversal.

Old coordinates/maps remain in the dyadic physical cache. A rejected trial or exhausted new-map budget preserves adopted flux; no fallback restarts the full solve. At1e-4 the main matrix reused12,422 uniform rows,5,642 c=0.5 rows,902 c=1 rows. Horizontal-only LD refinement still sums new row brightness samples; arbitrary local LD interval moments are not yet retained as a replacement for that sum.

## Verification

Final build:

- `unit_core` and `unit_boundary_support`:2/2 pass.
- Directional seed scaling: negative coordinates, separate axes, overflow and invalid factor checked.
- Retained-row test: identical runs/areas and fewer classified cells.
- Square adaptive148 results exactly match saved scalar magnifications, error indicators, map counts, histories, components and acceptance.
- Ordinary fixed547 scalar magnifications/error indicators exactly match saved outputs.
- Directional grid histories obey dyadic increments and per-axis caps.
- New-map budget exhaustion retains a positive partial flux and reports not met; subsequent ample-budget call reproduces the successful result.

The rectangular experimental route rejects the optional square-grid interval coverage-audit mode, whose geometry assumes one spacing. The normal adaptive API does not request it. No certification claim is inferred from the performance tests.

## Rejected experiments

Both are retained; neither changes the selected settings:

- Predicted4x jumps:148+18 accuracy checks passed, but1e-4 maps increased from2.238m to3.838m, and a bounded three-batch timing pilot was1.372 times slower. Keep ordinary2x increments.
- Axis ratio8:1e-3 passed74/74;1e-4 passed73/74. Uniform case s=1.3106493282130236,q=0.00012191596814834001,x=-0.5583732912541342,y=0.0026610337480946463,rho=0.03181757256303529 produced actual1.1912e-4 with estimated3.2174e-5. Its component grids were(32,256) and(16,128). The separate18 checks passed, which does not erase this counterexample. Keep ratio4.

The ratio8 failure establishes that excessive horizontal coarsening can leave error unseen by the outer estimator. It does not by itself establish whether the missing contribution is tip location, a missed subcell strip, or another boundary effect. Do not fit an exception to this case.

## Remaining work

The measured improvement supports directional refinement. Reaching fixed-like1e-4 speed still requires replacing component-wide directional scans with retained local boundary/row panels and a lightweight indicator for insufficient cross-direction resolution. The ratio8 counterexample is a useful regression case for that design. Simply raising the ratio or jumping farther has already failed the speed/accuracy checks.

## Artifacts and reproduction

Final code remains opt-in:

```sh
LCBININT_DIAGNOSTIC_DIRECTIONAL_ADAPTIVE=1
```

Leave `LCBININT_DIAGNOSTIC_ADAPTIVE_JUMP4`, `LCBININT_DIAGNOSTIC_ADAPTIVE_AXIS_RATIO`, and `LCBININT_DIAGNOSTIC_INTERIOR_BLOCKS` unset for the selected path. The default ratio is4. The optional ratio diagnostic accepts2,4,8,16;8 is retained for experimentation, not endorsed.

- `directional-adaptive-audit-v4/summary.json`: final148 accuracy rows, axis histories, reused-row counts.
- `directional-adaptive-fresh-v3/summary.json`: separate18 rows.
- `directional-adaptive-profile-cost-v3/summary.json`: final changing-geometry timing, raw seven-batch samples and native hash.
- `directional-axis-ratio-v1/`: final build/ctest/regression logs, rejected ratio8 results and final provenance.
- `directional-adaptive-v1`, `v2`: prior steps, seed/row unit checks, jump pilot.
- `directional-adaptive-cost-v2`: reference-informed fixed selections and all accuracy checks.

All result directories are under `tests/diagnostics/results/recal2026/`. Scripts: `adaptive_shared_flux_audit.py`, `directional_adaptive_regression.py`, `directional_adaptive_cost.py`; the latter accepts `--output UNIQUE --source directional-adaptive-cost-v2`. Use the existing probe build and project-local TMPDIR for compilation, and run with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LCBININT_PROBE_BUILD=$PWD/build-boundary-support` through the project Miniconda Python. No commit, production promotion or default seed-route change was made.
