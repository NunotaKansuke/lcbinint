# Paired LD singularity model for short closed gaps

## Scope and result

Continue `nbin-short-gap-and-phase-report.md`. All modifications are in the
experimental, default-off boundary-support overlay. Ordinary scalar
magnifications and the main production path are unchanged. No extra lens-map
queries or root solves are introduced. This is a local error estimate, not a
mathematical enclosure: `certified=false`, `complete_error_model=false` remain.
All errors below are absolute magnification errors.

The model improves five of the 547 retained comparisons numerically (four smaller
estimates and one larger estimate). Its benefit is local, not evidence of a
population-wide 20% improvement. Previous references are retained independently
of the estimator. Additional 90 geometry checks and 124 unseen-grid checks
exercise unchanged fallback results; they do not independently validate adoption
of the new local model.

## Mechanism

A short closed image-plane gap has nearby endpoints whose LD singularities cannot
be treated as independent infinite-range models. The quadratic local level-set
model gives `lambda(y)=(y-a)(y-b)`. Integrating square-root brightness across a
row produces a `lambda log|lambda|` singular term. Reuse the same closed-gap
geometry as the uniform estimator, with no new geometry queries.

For isolated gaps of 3–7 rows, retain three smooth rows on each side. Fit a
quartic regular part plus `lambda log|lambda|` and
`t lambda log|lambda|`, with `t=(y-(a+b)/2)/(b-a)`. The latter permits smooth
amplitude variation across the two endpoints. Seven coefficients use 9–13
retained rows, withholding the two smooth footprint ends. The regular quartic
uses the same fourth-difference convention as uniform. The geometry must lie
outside outer endpoint panels and both neighbouring image intervals must enclose
the gap.

Compute singular lattice moments together, sharing logarithms across powers.
Residual fourth differences and withheld-row coefficient updates contribute to
the remainder. Compare the old and proposed local remainders using retained row
second differences, including the same smooth fourth term. Adopt only the smaller
local uncertainty. This comparison never reads reference errors or tolerances.
Old isolated-contact calculations are reused, including on rejection. The total
indicator can nevertheless rise when its signed contributions cancel differently.

Rejected trials (independent endpoint amplitudes, a symmetric paired amplitude,
and ungated asymmetric fits) are retained in the diagnostic result directories.
They reduced selected worst tails but worsened mean discrepancies at the target
absolute scales. Smooth held-out ends, asymmetric amplitude, consistent junction
coverage, and reuse of the already evaluated model were needed together.

## Absolute-scale comparison

Reporting bands are `[3e-5,3e-4)` and `[3e-4,3e-3)`, not acceptance tolerances.
Statistics use the 372 low-grid rows at nbin=16,24,32,48,64,96.

| Profile / actual-error scale | Count | Previous mean abs(estimate−actual) | New | Reduction |
|---|---:|---:|---:|---:|
| uniform / 1e-4 | 67 | 5.90560e-6 | unchanged | 0% |
| uniform / 1e-3 | 43 | 8.75914e-6 | unchanged | 0% |
| LD / 1e-4 | 123 | 1.04808e-5 | 8.40122e-6 | 19.8% |
| LD / 1e-3 | 64 | 3.08896e-5 | 2.64659e-5 | 14.3% |

LD maximum excess above the actual error decreases from `3.93e-4` to `1.37e-4`
in the first band, and `5.38e-4` to `2.11e-4` in the second. Maximum deficits
remain `2.23e-5` and `2.00e-5`. No meaningful additional deficit appeared in any
of the 761 comparisons; floating reconstruction differences are below 1e-18.

Examples at nbin=16:

| s / c | Actual error | Previous estimate | New estimate |
|---|---:|---:|---:|
| 1.3378041232 / 1 | 5.90646e-4 | 1.12837e-3 | 7.72252e-4 |
| 1.3378041232 / .8 | 1.12480e-4 | 5.05424e-4 | 2.49631e-4 |
| 1.3132706269 / 1 | 8.41375e-6 | 9.39876e-4 | 4.88181e-4 |
| 1.3132706269 / .5 | 4.17968e-4 | 5.32282e-4 | 6.29399e-4 |

The final example worsens by `9.71e-5` despite lower local model uncertainty;
signed/global cancellation and positive remainders must be distinguished.
The small-error pure-LD example is still substantially overestimated. Neither
is resolved by this work. Do not add geometry-specific coefficients to force
these examples to fit.

## Verification and speed

- 547 saved-reference replays: scalar exactly unchanged; zero extra map evaluations.
- Setting `LCBININT_DIAGNOSTIC_DISABLE_PAIRED_SADDLE=1` restores the previous
  estimator exactly in all 547 cases (`disabled-baseline-check.json`).
- 45 previous unused-geometry checks plus 45 new unused-geometry checks.
- 62 profiles at unseen nbin=20,40: 124 checks; estimates unchanged.
- Analytic synthetic paired-row tests and independent large-cutoff singular
  moment tests passed; `unit_core` and `unit_boundary_support` passed.
- Timing uses the same binary, disabled/enabled model, changing geometries,
  one thread, 11 alternating pairs, nbin=16,32,64, c=0,.5,1.
  Five pilot geometries give candidate/previous medians 0.9974–1.0016.
  Two geometries where the new model is adopted give 0.9962–1.0069; LD-only
  ratios are 1.0006–1.0068. No substantial slowdown was observed in this bounded
  workload. Millisecond batches and timing noise preclude a precise sub-percent
  population claim. This does not measure adaptive end-to-end speed.

## Reproducible artifacts

All under `tests/diagnostics/results/recal2026/`:

- `paired-ld-final-replay-v1/summary.json`
- `paired-ld-final-fresh-replay-v1/summary.json`
- `paired-ld-final-new-fresh-replay-v1/summary.json`
- `paired-ld-final-intermediate-grid-v1/summary.json`
- `paired-ld-short-gap-cost-v1/summary.json`
- `paired-ld-short-gap-affected-cost-v1/summary.json`
- `paired-ld-short-gap-v1/`: intermediate source snapshots, build logs,
  analytic executables, disabled baseline check, final source hashes.

Replay with `shared_endpoint_effort_audit` using `--baseline` and unique
`--output`. For the new fresh dataset add `--before-field before_estimate` and
`--low-offset 0`. Run `paired_ld_intermediate_grid_audit --output UNIQUE`.
Run `paired_ld_short_gap_cost_audit --affected --output UNIQUE` for affected
profiles. Use `LCBININT_PROBE_BUILD=$PWD/build-boundary-support`,
`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and the project Python environment.

Next work should inspect the signed cancellation versus retained positive
remainder in the remaining large excess, before adding model order or broader
geometry coverage. The current model is retained as experimental, not promoted.
