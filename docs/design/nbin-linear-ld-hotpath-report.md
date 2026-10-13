# Analytic linear-LD hot path optimization

## Result

Retain the exact analytic profile instead of introducing a new approximation. In accepted Cartesian disk cells, mapped distance squared is nonnegative and the cell has already passed the source-disk test. Evaluate `1-c*(1-sqrt(max(0,1-r²)))` in the hot weight lambda, avoiding the generic source profile call, its redundant upper clamp and its two-coefficient dispatch. This preserves the same operation order as the analytic baseline. Endpoint/general-profile calls retain their general checks.

The previous experiment also left table-only coarea work enabled in the ordinary linear-LD error path. The boundary polish no longer requests inverse-gradient metadata for d=0. The outer row model has an optional `collect_table_density` flag; the ordinary linear-LD path disables table-only density aggregation and inverse-square-root moments. Defaults preserve diagnostic/legacy caller behavior; the accepted-refined outer call still uses that default, so it retains a small unused table-density calculation. Its outputs do not enter linear-LD errors. This remaining cleanup is not a numerical problem and is not claimed removed.

No approximate square root, discontinuous brightness table, new error floor, geometry calibration, lens mapping, or root solve is introduced. Uniform/LD integration and residual assembly stay shared.

## Validation

Replay the same 82 saved numerical-reference conditions. All magnifications and error indicators are bit equal to the analytic baseline (`shared-flux-exact-ld-v1`). Ordinary scalar equals indicator scalar, no additional error maps, no brightness-table remainder. Unit core/boundary support pass 2/2. Thus the previous distribution and worst underestimation remain unchanged: >=1.1 in15/82, >=2 in2/82, max2.4314, minimum ratio.993650. No claim of a universal mathematical bound.

## Timing

Five changing geometries, one thread, bins64/128/256/512, seven alternating scalar/indicator batches. Saved same-host comparisons to `shared-flux-error-exact-ld-cost-v2` (analytic generic profile) and `shared-flux-error-v14` (nearest-bin table):

| bins | indicator time / generic analytic | indicator time / former table | indicator overhead / new scalar |
|---|---:|---:|---:|
|64|.96755|.99897|6.36%|
|128|.95395|1.01039|6.91%|
|256|.94367|1.02356|6.29%|
|512|.94012|1.04231|5.12%|

First inline-only run independently showed the same direction (indicator time 2.8–6.2% below generic analytic). Final cleanup run is 3.2–6.0% below generic analytic. Sequential saved-baseline comparisons are not a randomized old/new binary experiment; small differences near1% should not be interpreted as established gains. The old approximately11% increase at512 is now approximately4%. Exact analytic brightness remains modestly slower than the old nearest-bin table at large grids. This is not a default promotion or an end-to-end adaptive-nbin speed claim.

## Remaining factor2

The worst geometry c=1,nbin128 has |signed residual|8.578169e-6 and model remainder9.302426e-6 for actual error7.353993e-6. Table contribution is exactly zero. Inspection finds the outer model includes a four-row fitted-cap remainder, a held-out LD cubic remainder, internal junction differences and smooth-row remainders. Their separate contributions have not yet been numerically isolated for this worst case. Do not lower the common remainder coefficients to fit it. This optimization leaves the error distribution unchanged; the remaining factor2 is still unresolved.

## Reproduction

Artifacts: `tests/diagnostics/results/recal2026/shared-flux-inline-ld-v1/` (before/after source, hashes,82-row summary, build/audit/unit logs). Timing outputs `shared-flux-inline-ld-cost-v1` (inline-only) and `shared-flux-inline-ld-cost-v2` (final cleanup).

Build: `cmake --build build-boundary-support --target lcbinint_python test_core -j 2`.

Use `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LCBININT_PROBE_BUILD=$PWD/build-boundary-support /home/nunota/.miniconda3/envs/myenv/bin/python -m tests.diagnostics.recal2026.shared_flux_exact_ld_audit --output shared-flux-inline-ld-v1`.

Timing: same environment, `-m tests.diagnostics.recal2026.shared_flux_error_audit --output shared-flux-inline-ld-cost-v2`, with a new exclusive output name on rerun. Tests: `ctest --test-dir build-boundary-support -R 'unit_core|unit_boundary_support' --output-on-failure`.
