# Empirical resolution law for finite-source magnification

August 2026 recalibration of the current certified binary-lens Cartesian and
polar inverse-ray algorithms. This document is the paper-facing statement of
the result. It is an offline calibration record; it does not yet change the
C++ runtime selector.

## 1. The common principle

Every numerical route is judged against one error budget. For a magnification
value \(A\), define

\[
 S(A)=\max(|A|,1),\qquad
 B(A)=a_{\rm tol}+r_{\rm tol}S(A),\qquad
 \varepsilon(A)=\frac{B(A)}{S(A)}.
\]

The dimensionless quantity \(\varepsilon\) is the common axis of the
calibration. It makes the Cartesian and polar results comparable without
pretending that their convergence constants are identical. The route may have
its own estimator and its own resolution coordinate, but it should answer the
same question: does its estimated error stay within \(B\)?

There are two different values of \(A\), and keeping them separate is
important:

* In this offline study, \(A=A_{\rm ref}\), the high-resolution reference used
  to label the data.
* At runtime, \(A=\hat A(N)\), the value produced by the current evaluation.
  The runtime computes the budget from that value and uses the route-specific
  embedded error estimate to decide whether to accept or refine.

The reference value is therefore a calibration instrument, not an additional
runtime input.

## 2. Data and convergence label

The current algorithm was evaluated on 6,000 discovery rows and 2,200 rows from
an independent holdout. Each row contains a Cartesian and a polar resolution
ladder, with the offline ladder

    4, 6, 8, 10, 12, 16, 24, 32, 40, 50, 64, 80, 100,
    128, 160, 200, 256, 320, 400.

For a target budget, the required resolution is the first bucket \(N\) such
that the result at \(N\) and every finer measured bucket satisfy

\[
 |A(N)-A_{\rm ref}|\le B(A_{\rm ref}).
\]

This persistent-crossing rule rejects an isolated lucky crossing. A row is
used only when the stored reference uncertainty is at most 10% of the target
normalized budget. Discovery data determine the coefficients; holdout data are
used only for the reported coverage.

## 3. The fitted law

The simplest law supported by the data is a grid-specific p99 power law in the
single common variable \(\varepsilon\):

\[
 N_{99,g}(\varepsilon)=
 \left\lceil C_g\left(\frac{\varepsilon}{10^{-3}}\right)^{-\beta_g}
 \right\rceil_{\mathcal N},
\]

where \(\lceil\cdot\rceil_{\mathcal N}\) means rounding upward to the next
measured bucket. The fitted form is linear in base-two logarithms:

\[
 \log_2 N_{99,g}
 =\log_2 C_g+\beta_g\log_2\left(\frac{10^{-3}}{\varepsilon}\right).
\]

The coefficients are:

| grid | \(C_g\) | \(\beta_g\) | independent holdout coverage |
|---|---:|---:|---:|
| Cartesian | 45.32 | 0.4767 | 99.67% |
| polar | 94.57 | 0.5952 | 99.81% |

The exponent has a direct interpretation. Halving the normalized budget
requires approximately \(2^{0.477}=1.39\) times as many Cartesian bins and
\(2^{0.595}=1.51\) times as many polar bins at the p99 level. The common
principle is therefore shared, while the two numerical paths retain their
measured prefactor and exponent.

## 4. Paper-facing bucket table

The table below is the actual supported rule after upward bucket rounding. The
coverage columns are measured on the independent holdout.

| normalized budget \(\varepsilon\) | Cartesian \(N_{99}\) | Cartesian coverage | polar \(N_{99}\) | polar coverage |
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

For the common default \(a_{\rm tol}=0,\ r_{\rm tol}=10^{-3}\), this gives
Cartesian \(N=50\) and polar \(N=100\) as the p99 initial values. These are
population-level upper-quantile settings, not a claim that every individual
case needs that many bins. The median work ratios of the resulting p99 rule
relative to the measured requirement are about 16 for Cartesian and 64 for
polar, because a one-shot p99 setting deliberately pays for the hard tail. A
validated runtime estimator and upward retry can recover efficiency on easy
cases.

## 5. Why no \(A_{\rm point}\) or \(d/\rho\) correction is included

Both variables remain useful diagnostics, and \(A_{\rm point}\) is still used
for the separate Cartesian/polar route choice. They do not earn a term in the
common Nbin law:

\[
 \log_2N=a_g+b_g\log_2(10^{-3}/\varepsilon)
 +\gamma_g\max\left(0,\log_2(A_{\rm point}/A_{0,g})\right).
\]

Discovery selected weak, grid-dependent candidates: \(\gamma=+0.068\) above
\(A_0=4\) for Cartesian and \(\gamma=-0.119\) above \(A_0=64\) for polar.
Holdout coverage was 99.77% and 99.85%, but the Cartesian median work ratio
rose from 16 to 25 and the polar ratio did not improve from 64. A correction
whose sign changes between methods and does not reduce validated work is not a
good common rule. The production calibration therefore sets \(\gamma=0\).
The same reasoning rejects \(d/\rho\) as an additional Nbin branch in this
dataset.

## 6. What \(a_{\rm tol}\) does, and what it does not do

The absolute tolerance is an additive allowance in the same budget. It is not
a minimum Nbin, a separate strictness mode, or a promise that an absolute-only
request has a cheap universal initial rule. With \(r_{\rm tol}>0\), it simply
changes the normalized target to

\[
 \varepsilon=r_{\rm tol}+\frac{a_{\rm tol}}{S(A)}.
\]

The relative law is the useful primary calibration because it removes the
trivial magnification scale. A mixed \(a_{\rm tol}\) cross-check at
\(a_{\rm tol}=10^{-4}\) gave \(C=44.7,\beta=0.538\) for Cartesian and
\(C=92.5,\beta=0.507\) for polar, with 99.69% and 99.76% holdout coverage.
This supports the normalization, but it is not a dedicated validation campaign
for arbitrary absolute tolerances.

The separate \(r_{\rm tol}=0\) experiment is a useful negative result. A raw
absolute-only fit gave:

| grid | \(C^{\rm abs}\) | \(\beta^{\rm abs}\) | holdout coverage | median predicted Nbin |
|---|---:|---:|---:|---:|
| Cartesian | 140.5 | 0.110 | 99.67% | 160 |
| polar | 201.0 | 0.229 | 99.75% | 200 |

The coverage is adequate, but the rule is much too conservative and the usable
reference population shrinks as \(a_{\rm tol}\) becomes small. Therefore:

1. Use the relative-budget law when \(r_{\rm tol}>0\).
2. Evaluate the common runtime budget after the first pass and let the
   method-specific estimator trigger an upward retry if needed.
3. Treat the special \(r_{\rm tol}=0\) mode as requiring a measured first pass
   or a future scale-aware calibration; do not force the raw absolute-only fit
   into the common initial selector.

## 7. The runtime estimator is a separate claim

The p99 law above is calibrated from the observed reference error
\(\lvert A(N)-A_{\rm ref}\rvert\). It must not be confused with a proof that
the runtime-reported estimator \(E\) is always an upper bound.

A separate audit of 2,046 rows from the holdout measured

\[
 R=\frac{E}{\lvert A(N)-A_{\rm ref}\rvert}.
\]

On the reference-consistent subset at \(r_{\rm tol}=10^{-3}\), the
method-stratified finite-\(R\) results were:

| method | median \(R\) | p05 \(R\) | p95 \(R\) | fraction with \(R<1\) |
|---|---:|---:|---:|---:|
| Cartesian inverse ray | 1.52 | 0.087 | 21.0 | 37.5% |
| polar inverse ray | 1.78 | 0.184 | 11.4 | 34.1% |
| source-plane quadrature | 2.83 | 2.64 | 4.40 | 1.1% |

The grid estimators are therefore useful convergence indicators, but this
audit does not support calling them certified upper bounds. The resolution law
and its independent holdout coverage are the established empirical result;
turning the estimator into a standalone guarantee requires a separate
calibration or an additional fail-closed reference check. This distinction is
why the present commit records the law without silently changing runtime
acceptance semantics.

## 8. Status and limits of the claim

This result is established for the current binary Cartesian and polar
image-plane ladders. It is not a license to mix in the old
finite-source-auto-20260716 campaign, which used a different algorithm.
Source-plane/chord quadrature is not yet fitted because its public diagnostic
record does not preserve the complete production \(A(N)\) ladder. It should
reuse the same budget definition and persistent-crossing criterion, with its
own calibrated resolution coordinate and coefficients.

The current C++ branch still has its existing runtime one-shot selector and
embedded error controller. This commit records and tests the new calibration;
applying this p99 law to runtime nbin='auto' is a separate, reviewable change.

## 9. Reproduction

From the repository root:

    PYTHONPATH=. python -m tests.diagnostics.recal2026.error_budget_law \
      --discovery tests/diagnostics/results/recal2026/discovery \
      --holdout tests/diagnostics/results/recal2026/holdout \
      --output tests/diagnostics/results/recal2026/error_budget_law --no-plots

    PYTHONPATH=. python -m tests.diagnostics.recal2026.absolute_error_law \
      --discovery tests/diagnostics/results/recal2026/discovery \
      --holdout tests/diagnostics/results/recal2026/holdout \
      --output tests/diagnostics/results/recal2026/absolute_error_law

    PYTHONPATH=. python -m tests.diagnostics.recal2026.error_budget_percentiles \
      --output tests/diagnostics/results/recal2026/figures/empirical-resolution-law.pdf

    pytest -q tests/diagnostics/recal2026/test_empirical_law.py

The machine-readable calibration record is
tests/diagnostics/recal2026/empirical_law.py. The PDF contains the requested
box-and-whisker distributions, a representative Nbin convergence example, and
a one-page description of the fitting procedure. The larger per-case CSV
outputs remain ignored because they are reproducible campaign data rather than
source-controlled paper metadata.
