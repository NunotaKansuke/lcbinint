# Cartesian row residual investigation

## Implemented

The experimental overlay records the existing boundary-refinement brackets, including floating residual-zero exits. No extra lens-map queries or image roots are introduced by bracket recording. These brackets are numerical diagnostics, not certified enclosures: floating signs and uniqueness have not been proved. The ordinary master kernel is unchanged.

The trace now exposes signed brightness-model residuals and limb darkening. Saved runs reconstruct final flux from the actual Bennett-weighted row contributions. Diagnostic recording on/off produced exactly equal values for all 40 conditions.

## Bounded evidence

Artifacts: `tests/diagnostics/results/recal2026/cartesian-residual-diagnostic-20261012-v1/`. Five geometries, two source positions, uniform/limb-darkened brightness, nbin 16/64. Independent VBM references were reused, with their recorded stability. No runtime repeated-grid convergence check is proposed.

| profile | nbin | absolute second difference covers | grouped signed candidate covers |
| --- | --- | --- | --- |
| uniform | 16 | 6/10 | 6/10 |
| uniform | 64 | 10/10 | 10/10 |
| limb darkened | 16 | 8/10 | 7/10 |
| limb darkened | 64 | 10/10 | 10/10 |

This first table uses unweighted row contributions. Grouping reduces the estimate by only about 2–15% on average and can worsen underestimation. Differences of final Bennett-weighted contributions are also saved in `corrected-row-summary.json`: coverage is 6/10, 10/10, 8/10, 10/10 for both absolute and grouped variants. Applying the midpoint remainder coefficient to corrected contributions has no mathematical justification; this is a diagnostic comparison only.

The numerical boundary-position indicator divided by actual error has median 0.040 at nbin16 and 0.102 at nbin64 for uniform brightness. This does not bound total boundary error or establish what causes the remaining error. Limb darkening requires a flux-weighted position remainder and is excluded from this comparison.

Only 36/40 trace results agree with the saved production-route values. All four discrepancies belong to the same s=1.1, q=1e-5, position2 geometry. The trace constructs augmented seeds directly with infinite caustic distance; the production route passes through its own distance/seed selection. The precise cause of the discrepancy remains unresolved. Consequently these results characterize the experimental trace, not a fully matched production benchmark.

## Next step and acceptance gates

Do not lower nbin or promote either indicator. First reconcile the trace seed inputs with the production route. Then split the actual one-dimensional integrand into smooth regions and endpoint contributions. At a generic image boundary turn, the width has a square-root leading term; test analytic integration of that term against Bennett endpoint weights and estimate only the residual. Degenerate turns, cusp behavior, and limb-darkened flux need separate treatment. Sampled row-count stability alone cannot certify smoothness.

Reuse saved traces for candidate screening before adding map evaluations. Any eventual acceptance rule must account for omitted components and boundary-location uncertainty as well as quadrature remainder. Report added work and total runtime separately from trace allocation cost.

## Follow-up: matched scheduling and endpoint basis (2026-10-13)

The four discrepancies were isolated by running the experimental production wrapper: it agreed exactly with the experimental trace, excluding the wrapper/trace transformation as the cause. Restoring master scheduling (offer all seeds before draining the frontier) removed all four discrepancies. The overlay now defaults to master scheduling; `LCBININT_SEQUENTIAL_SEED_FILL` explicitly retains the archived sequential experiment. Seed order affects discovery/run partition and therefore quadrature; equality of visited cells alone is insufficient to establish equal integrals. The precise run-partition mechanism was not separately instrumented.

After recompilation, `cartesian-residual-diagnostic-20261013-v2/summary.json` records 40/40 master matches and 40/40 exact diagnostic on/off matches. The old v1 results remain as provenance. Uniform position-indicator medians are now 0.067 and 0.102 of actual error at nbin16/64. The grouped raw-row candidate covers 6/10, 10/10, 8/10, 10/10; it is still unsuitable for acceptance.

A separate algebraic audit of the implemented Eq15 weights shows exact constant and linear moments, but nonzero square-root moments. This explains why a generic smooth midpoint residual formula cannot describe a turning endpoint. Fit the local basis `a*t^(1/2)+b*t^(3/2)` from two existing samples at a known endpoint, then integrate that basis analytically and subtract its actual Eq15 contribution. A third sample can diagnose model mismatch; it does not certify the omitted remainder.

`outer-endpoint-basis-20261013-v1/summary.json`: analytic unit-circle endpoint panel, eta .01/.05/.15/.3/.5/.75/.99 and spacing 1/16,1/64, 14 conditions. The maximum remaining absolute panel error divided by the original panel error is 0.000876. This favorable result assumes the endpoint location is known exactly; it is neither a microlensing benchmark nor a global error bound. No lens solves are needed for this audit.

Next: obtain endpoint location and eligibility from existing run geometry, without global regridding. First assess the uncertainty from estimating that location; reject ambiguous branching/degenerate panels rather than silently fitting them. Verify eligible lens panels with independent local area references before introducing any runtime correction or acceptance rule.

Reproduce (single thread, from repo root):

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LCBININT_PROBE_BUILD=/rogue1_8/nunota/lcbinint/build-boundary-support /home/nunota/.miniconda3/envs/myenv/bin/python -m tests.diagnostics.recal2026.cartesian_residual_diagnostic
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/nunota/.miniconda3/envs/myenv/bin/python -m tests.diagnostics.recal2026.outer_endpoint_basis_audit
```

Scripts intentionally refuse to overwrite existing result directories. Use a new output suffix when repeating.

## Consolidated follow-up

See `nbin-single-grid-endpoint-model-report.md` for the native implementation, internal-gap model, local107-panel audit, separate-position global checks, 60-digit reference checks, timing, and remaining acceptance limitations.
