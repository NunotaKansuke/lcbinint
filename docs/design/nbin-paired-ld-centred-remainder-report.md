# Centre retained LD models instead of adding a one-sided full spread

## Scope

Follow-up to `nbin-paired-ld-short-gap-report.md`, experimental overlay only.
Fixed-grid magnifications and uniform estimates are unchanged; zero extra
lens-map evaluations or root solves. The local indicator remains asymptotic,
not a mathematical bound. No tolerance or default policy is changed.

## Diagnosis and change

The paired short-gap model fits rows while withholding its two smooth ends,
then enriches the fit with those ends using already available rank-one updates.
Previously its signed prediction used the original fit and its positive
remainder included the entire change in the singular quadrature moment.

Use the midpoint of these two models for the signed prediction, regular fourth
term, and row predictions. Keep half the singular-moment spread as uncertainty,
and recompute residual fourth differences around the midpoint. Junction second
differences therefore use the same centred model. This shares the outer endpoint
model's existing centring convention, rather than choosing a geometry multiplier.
Skip the original residual pass whose result would otherwise be discarded.
`LCBININT_DIAGNOSTIC_LEGACY_PAIRED_CENTRE=1` retains the exact previous route.

This is a representation of local model sensitivity, not proof that the actual
error lies between the two surrogate models. The actual/reference error is never
used to select a model at runtime.

## Absolute errors

547 comparisons reuse independent saved references. 372 low-grid comparisons
use nbin=16,24,32,48,64,96. Reporting bands are `[3e-5,3e-4)` and
`[3e-4,3e-3)`; these are analysis bands, not acceptance tolerances.

| LD actual-error scale | Count | Previous mean abs(estimate−actual) | Centred | Improvement |
|---|---:|---:|---:|---:|
| 1e-4 | 123 | 8.40122e-6 | 8.04056e-6 | 4.3% |
| 1e-3 | 64 | 2.64659e-5 | 2.49599e-5 | 5.7% |

Maximum excess above actual error decreases `1.37150e-4 → 9.65831e-5` in
first band and `2.11430e-4 → 1.82000e-4` in second. Maximum deficits remain
`2.22554e-5` and `2.00155e-5`. Five comparisons change; all five total estimates
decrease without creating an additional underestimated case. These counts are
saved comparisons, not a random population estimate.

At nbin=16, s=1.3378041232, c=1, actual error is `5.90646e-4`:
- signed prediction: `-5.95959e-4 → -5.88638e-4`;
- positive remainder: `1.76293e-4 → 1.25275e-4`;
- total estimate: `7.72252e-4 → 7.13913e-4`.

Mixed LD at s=1.3132706269, c=.5, actual `4.17968e-4`, improves its
estimate `6.29399e-4 → 5.99968e-4`. This still exceeds the earlier independent
paired-model baseline `5.32282e-4`; do not present it as fully repaired.

The difficult pure-LD small-error case at the same s has actual signed error
`+8.41375e-6`. Its signed estimate worsens `-3.59905e-5 → -1.03814e-4`, while
its positive remainder decreases `4.52190e-4 → 3.18195e-4`. The total improves
`4.88181e-4 → 4.22009e-4`, but the mismatch is unresolved. Much of the apparent
improvement here is tighter sensitivity bookkeeping, not more accurate signed
physics. Its sign error deserves a separate retained-model/geometry diagnosis.

As an illustration only, comparing indicator decisions at 1e-4 and 1e-3 gives
zero false accepts before and after in these 547 comparisons. False rejects
remain eight and zero respectively. No adaptive nbin reduction is demonstrated.

## Verification, timing, and provenance

- Legacy-centre flag exactly reproduces all 547 previous estimates.
- Ordinary scalar values exactly match in all replay checks; extra maps zero.
- Two independent unused-geometry groups of45 and an unseen-grid group of124
  are unchanged. They support fallback stability, not new-model adoption.
- Analytic paired-row and moment tests; unit_core and unit_boundary_support.
- Same-binary legacy/new timing on two changing geometries with adopted paired
  models, c=0,.5,1 and nbin=16,32,64; 11 alternating pairs, one thread.
  Initial clean timing medians span0.9974–1.0065 (LD0.9984–1.0065).
  Final optimized-source verification/timing is recorded separately below.
  Millisecond batches have measurable timing noise; no population-wide
  sub-percent speed claim is made.

Artifacts under `tests/diagnostics/results/recal2026/`:

- `paired-ld-centred-replay-v1/summary.json` and legacy-check-v1.
- `paired-ld-centred-fresh-v1/summary.json`, new-fresh-v1,
  and intermediate-v1 (unseen nbin20/40).
- `paired-ld-centred-affected-cost-v2/summary.json` is the initial clean timing.
  v1 overlapped test compilation and is excluded from timing claims.
- `paired-ld-centred-v1/`: before-source snapshot, build logs, analytic test,
  threshold crossings, source hashes and final verification records.

Replay: `python -m tests.diagnostics.recal2026.shared_endpoint_effort_audit
--baseline paired-ld-final-replay-v1 --output UNIQUE`.
Timing: `python -m tests.diagnostics.recal2026.paired_ld_short_gap_cost_audit
--affected --legacy-flag LCBININT_DIAGNOSTIC_LEGACY_PAIRED_CENTRE --output UNIQUE`.
Use the existing myenv Python, `LCBININT_PROBE_BUILD=$PWD/build-boundary-support`,
`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`.

Further work should distinguish the local level-set approximation and remaining
positive junction residuals in the small-error sign-mismatch case. Reducing
remainders by a tuned factor would conceal this issue and is not justified.

## Final optimized source check

Removing the discarded residual pass preserves all547 centred estimates exactly.
The rebuilt unit_core and unit_boundary_support and analytic paired-row test pass.
Final clean same-binary timing is `paired-ld-centred-final-cost-v1/summary.json`;
ratios span 0.9953698116259043–1.0041691273209423.
These short batches still cannot resolve a precise sub-percent speed effect.
Final native identity is in `paired-ld-centred-final-check-v1/summary.json`,
with source hashes refreshed in `paired-ld-centred-v1/source-hashes.json`.
