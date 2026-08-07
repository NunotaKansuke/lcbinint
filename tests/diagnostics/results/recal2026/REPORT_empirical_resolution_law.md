# Empirical Nbin law with a common max-budget policy

August 2026 recalibration of the current binary-lens Cartesian and polar
inverse-ray ladders. This is the paper-facing handoff for the resolution
calibration. It records the data split, the fitting procedure, the absolute
and relative laws, the mixed-tolerance rule, the independent holdout result,
and the reproducibility commands.

The central result is simple:

> Every numerical route uses the same tolerance semantics. The route-specific
> convergence law may differ, but absolute and relative tolerances are
> alternative allowances and the less demanding branch is sufficient.

The calibration is complete and the mixed rule passes the holdout test. The
production C++ selector is intentionally not changed by this report; wiring
this offline policy into `nbin="auto"` is a separate implementation step.
Until that follow-up lands, the current runtime must not be described as
already using this max-budget policy; this document is the final calibration
and handoff record for the change.

## 1. Policy

For a magnification \(A\), define the scale

\[
 S(A)=\max(|A|,1).
\]

The two allowed error branches are

\[
 B_{\rm abs}=a_{\rm tol},\qquad
 B_{\rm rel}=r_{\rm tol}S(A).
\]

The effective acceptance budget is the larger allowance,

\[
 B_{\rm eff}=\max(B_{\rm abs},B_{\rm rel}),
 \qquad
 \varepsilon_{\rm rel}=\max\left(\frac{a_{\rm tol}}{S(A)},r_{\rm tol}\right).
\]

Equivalently, the calculation is accepted when

\[
 E\le a_{\rm tol}\quad\text{or}\quad
 E\le r_{\rm tol}S(A).
\]

This is the same logical structure as the VBMicrolensing stopping rule: the
calculation continues only while both tests fail. It is not the additive
budget \(a_{\rm tol}+r_{\rm tol}S(A)\), and it is not the stricter minimum of
the two allowances.

The resolution selector uses the corresponding two empirical branches:

\[
 N_{\rm abs,g}(a_{\rm tol})
 =\left\lceil C^{\rm abs}_g
 \left(\frac{a_{\rm tol}}{10^{-3}}\right)^{-\beta^{\rm abs}_g}\right\rceil_{\mathcal N},
\]

\[
 N_{\rm rel,g}(r_{\rm tol})
 =\left\lceil C^{\rm rel}_g
 \left(\frac{r_{\rm tol}}{10^{-3}}\right)^{-\beta^{\rm rel}_g}\right\rceil_{\mathcal N},
\]

where \(g\in\{\text{Cartesian},\text{polar}\}\), and
\(\lceil\cdot\rceil_{\mathcal N}\) rounds upward to the measured ladder

\[
 \mathcal N=\{4,6,8,10,12,16,24,32,40,50,64,80,100,128,160,200,256,320,400\}.
\]

For a mixed request, the adopted rule is

\[
 \boxed{N_{\rm mix,g}=\min\left(N_{\rm abs,g}(a_{\rm tol}),
                                  N_{\rm rel,g}(r_{\rm tol})\right).}
\]

The `min` is not an arbitrary optimization. On a fixed resolution ladder,
increasing the allowed error from either branch to \(B_{\rm eff}\) can only
move the first persistent crossing earlier. Therefore the required mixed
resolution is the minimum of the two pure-branch requirements. The empirical
p99 laws still need an explicit mixed holdout check because each branch law is
calibrated from a marginal distribution.

Special cases are immediate: `atol=0` uses only the relative branch,
`reltol=0` uses only the absolute branch, and both zero is invalid.

## 2. Data and convergence label

The campaign contains 6,000 discovery rows and 2,200 independent holdout rows.
Each row has Cartesian and polar ladders evaluated at the same source position,
with the resolutions listed above. The reference is the high-resolution result
stored with its uncertainty.

For each requested budget, the required resolution is the first ladder bucket
for which the result and every finer measured bucket satisfy

\[
 |A(N)-A_{\rm ref}|\le B_{\rm eff}(A_{\rm ref}).
\]

This persistent-crossing definition rejects an isolated lucky crossing. A row
is used only when the reference uncertainty is at most 10% of the requested
normalized budget. Discovery data determine the fit; the holdout is never used
to choose coefficients or safety offsets.

The usable record counts are:

| branch | discovery records | holdout records |
|---|---:|---:|
| relative | 99,678 | 37,050 |
| absolute | 80,317 | 30,088 |

The absolute branch is smaller at tight tolerances because a fixed absolute
accuracy is harder to resolve relative to the uncertainty of high-
magnification references. This is a data-availability qualification, not a
change in the definition of the rule.

## 3. Fitting procedure

For each grid and each branch:

1. Compute the required persistent-crossing bucket for every discovery row and
   tolerance level.
2. Summarize the required bucket distribution at each level by its p99.
3. Fit the p99 values in base-two logarithms with a one-variable power law.
4. Choose the smallest discovery-side upward offset that reaches 99% coverage
   overall and at every available tolerance level.
5. Round the continuous prediction upward to the next supported bucket.
6. Apply the frozen law to the independent holdout.

The logarithmic form is

\[
 \log_2N_{99}=\log_2C+\beta\log_2(10^{-3}/\tau),
\]

where \(\tau=a_{\rm tol}\) for the absolute branch and
\(\tau=r_{\rm tol}\) for the relative branch. Base two is used because one
step in the ladder is naturally a resolution refinement, and the slope has a
direct interpretation: halving the requested tolerance multiplies the p99
resolution by \(2^\beta\).

No \(A_{\rm point}\), \(d/\rho\), cusp-distance, or topology hinge is included
in the Nbin law. Those quantities remain available for route selection and
diagnostics, but they did not produce a stable common reduction in validated
work in this calibration. Adding them would make Cartesian and polar follow
different exception rules without improving the holdout guarantee.

## 4. Fitted laws

The discovery fits and their independent holdout coverage are:

| grid | \(C^{\rm rel}\) | \(\beta^{\rm rel}\) | relative coverage | \(C^{\rm abs}\) | \(\beta^{\rm abs}\) | absolute coverage |
|---|---:|---:|---:|---:|---:|---:|
| Cartesian | 45.32 | 0.4767 | 99.67% | 140.47 | 0.1103 | 99.67% |
| polar | 94.57 | 0.5952 | 99.81% | 201.05 | 0.2286 | 99.75% |

The relative branch is the efficient main calibration. At the common default
\(r_{\rm tol}=10^{-3}\), the p99 initial buckets are Cartesian 50 and polar
100. The absolute branch is intentionally conservative: at
\(a_{\rm tol}=10^{-3}\), its p99 buckets are Cartesian 160 and polar 200.
That difference is the measured cost of asking for an absolute error that does
not relax as the magnification grows.

### 4.1 Relative branch table

| \(r_{\rm tol}\) | Cartesian bucket | coverage | polar bucket | coverage |
|---:|---:|---:|---:|---:|
| \(1.0\times10^{-2}\) | 16 | 99.58% | 32 | 100.00% |
| \(5.0\times10^{-3}\) | 24 | 99.58% | 40 | 99.86% |
| \(3.0\times10^{-3}\) | 32 | 99.67% | 50 | 100.00% |
| \(2.0\times10^{-3}\) | 40 | 99.86% | 64 | 99.86% |
| \(1.0\times10^{-3}\) | 50 | 99.81% | 100 | 99.77% |
| \(5.0\times10^{-4}\) | 64 | 99.57% | 160 | 99.76% |
| \(3.0\times10^{-4}\) | 100 | 99.71% | 200 | 99.46% |
| \(2.0\times10^{-4}\) | 100 | 99.44% | 256 | 99.54% |
| \(1.0\times10^{-4}\) | 160 | 99.83% | 400 | 100.00% |

### 4.2 Absolute branch table

| \(a_{\rm tol}\) | Cartesian bucket | coverage | polar bucket | coverage |
|---:|---:|---:|---:|---:|
| \(1.0\times10^{-2}\) | 128 | 99.85% | 128 | 99.60% |
| \(5.0\times10^{-3}\) | 128 | 99.45% | 160 | 99.75% |
| \(3.0\times10^{-3}\) | 128 | 99.38% | 160 | 99.59% |
| \(2.0\times10^{-3}\) | 160 | 100.00% | 200 | 99.89% |
| \(1.0\times10^{-3}\) | 160 | 99.83% | 256 | 99.89% |
| \(5.0\times10^{-4}\) | 160 | 99.58% | 256 | 99.76% |
| \(3.0\times10^{-4}\) | 200 | 99.86% | 320 | 99.86% |
| \(2.0\times10^{-4}\) | 200 | 99.46% | 400 | 99.54% |
| \(1.0\times10^{-4}\) | 200 | 99.50% | 400 | 100.00% |

The absolute table should be read as a safe initial estimate, not as an
efficiency claim. Its median work versus the measured requirement is 256 for
Cartesian and 178 for polar, compared with 16 and 64 for the relative branch.
The large factor is accepted here because the absolute-only mode is a
conservative fallback, not the normal default.

## 5. Mixed-tolerance validation

The mixed test used all \(9\times9=81\) positive pairs from the levels in the
two tables. For each pair and each holdout row, the script did two independent
things:

* It measured the required bucket directly with the effective budget
  \(\max(a_{\rm tol},r_{\rm tol}S(A_{\rm ref}))\).
* It predicted `min(N_abs, N_rel)` using coefficients fixed before looking at
  the holdout row.

The direct mixed required bucket agreed with
\(\min(N_{\rm required,abs},N_{\rm required,rel})\) in every comparable case.
The coverage result is:

| grid | mixed pairs | minimum coverage | median coverage | pairs at or above 99% |
|---|---:|---:|---:|---:|
| Cartesian | 81 | 99.44% | 99.81% | 81/81 |
| polar | 81 | 99.46% | 100.00% | 81/81 |

The identity check covered 132,341 Cartesian and 132,137 polar row-pair cases;
both had zero mismatches. Thus the less-demanding-branch rule is not merely a
logical description of the tolerance budget: it is also the measured
resolution composition on the independent holdout.

The heatmap in the PDF shows all 162 coverage values. The worst cells are
Cartesian at \(a_{\rm tol}=2\times10^{-4},r_{\rm tol}=2\times10^{-4}\),
99.44%, and polar at \(a_{\rm tol}=3\times10^{-4},r_{\rm tol}=3\times10^{-4}\),
99.46%. Both remain above the predeclared 99% target.

## 6. Figures

[`figures/empirical-resolution-law.pdf`](figures/empirical-resolution-law.pdf)
contains five pages:

1. Relative-branch box-and-whisker distributions of required Nbin. The box is
   Q1--Q3, the thick vertical bar is the central 68% interval (p16--p84), the
   thin whisker is p5--p95, the red diamond is p99, and the open square is the
   fitted supported bucket.
2. The same figure for the absolute branch.
3. A Cartesian/polar heatmap of all 81 mixed holdout coverages.
4. A representative holdout convergence curve, showing \(A(N)\) and
   \(|A(N)-A_{\rm ref}|\) against Nbin for both grids.
5. The fitting recipe, equations, fitted constants, and minimum mixed
   coverage in one page suitable for inclusion in a methods supplement.

The representative convergence case is holdout case 48 with the linear
limb-darkening profile, \(s=3\), \(q=10^{-5}\), and
\(\rho=1.49005\times10^{-3}\). At \(r_{\rm tol}=10^{-3}\), its measured
required buckets are Cartesian 10 and polar 12. This is an illustration, not
the calibration sample used to choose the law.

## 7. What this result does and does not claim

This report establishes an empirical p99 initial-resolution rule against the
stored high-resolution reference. It does not establish that the runtime
embedded estimator \(E\) is a certified upper bound for the true error. The
separate estimator audit found route-dependent underestimation, so the
resolution law and the estimator certificate must remain separate claims.

It also does not claim that a fixed p99 bucket is the fastest answer for every
row. The rule deliberately pays for the hard tail; an eventual runtime
implementation can use the common policy for acceptance and refine only when a
route-specific indicator requires it, provided that the fail-closed behavior
and the holdout coverage are preserved.

The study covers binary Cartesian and polar inverse-ray ladders. It does not
fit source-plane quadrature here because its production resolution coordinate
is not stored as the same complete \(A(N)\) ladder. That route should adopt the
same budget semantics but receive its own calibrated resolution law.

Finally, \(A_{\rm ref}\) appears in the offline definition only because it is
the independent accuracy label. At runtime, the same budget is evaluated from
the current estimate or from the route's convergence state; the reference is
not an extra runtime input.

## 8. Reproduction

From the repository root, after the calibrated discovery and holdout ladders
are present:

```sh
PYTHONPATH=. python -m tests.diagnostics.recal2026.error_budget_law \
  --discovery tests/diagnostics/results/recal2026/discovery \
  --holdout tests/diagnostics/results/recal2026/holdout \
  --output tests/diagnostics/results/recal2026/error_budget_law --no-plots

PYTHONPATH=. python -m tests.diagnostics.recal2026.absolute_error_law \
  --discovery tests/diagnostics/results/recal2026/discovery \
  --holdout tests/diagnostics/results/recal2026/holdout \
  --output tests/diagnostics/results/recal2026/absolute_error_law

PYTHONPATH=. python -m tests.diagnostics.recal2026.mixed_error_law \
  --discovery tests/diagnostics/results/recal2026/discovery \
  --holdout tests/diagnostics/results/recal2026/holdout \
  --relative-report tests/diagnostics/results/recal2026/error_budget_law/error_budget_law.json \
  --absolute-report tests/diagnostics/results/recal2026/absolute_error_law/absolute_error_law.json \
  --output tests/diagnostics/results/recal2026/mixed_error_law.json

PYTHONPATH=. python -m tests.diagnostics.recal2026.error_budget_percentiles \
  --output tests/diagnostics/results/recal2026/figures/empirical-resolution-law.pdf

pytest -q tests/diagnostics/recal2026/test_empirical_law.py
```

The compact machine-readable mixed result is
[`mixed_error_law.json`](mixed_error_law.json). The law constants and selector
functions are in
[`empirical_law.py`](../../recal2026/empirical_law.py), and the
regression tests are in
[`test_empirical_law.py`](../../recal2026/test_empirical_law.py).
The large per-row ladders and CSVs remain ignored working data; the report,
source record, compact mixed result, PDF, and tests are the committed handoff.

## 9. Paper-ready summary

For a methods section, the result can be stated as follows:

> We calibrated the required inverse-ray resolution from independent
> high-resolution reference ladders using the first persistent tolerance
> crossing and a discovery-side 99th-quantile power law. Absolute and relative
> tolerance requests share a common acceptance policy,
> \(B=\max(a_{\rm tol},r_{\rm tol}\max(|A|,1))\), while Cartesian and polar
> integration retain separate empirical convergence constants. For a mixed
> request, the initial resolution is the smaller of the absolute-only and
> relative-only predictions. Across all 81 positive tolerance pairs, this rule
> covered 99.44--100.00% of the independent holdout for Cartesian and
> 99.46--100.00% for polar, exceeding the predeclared 99% target in every
> cell.
