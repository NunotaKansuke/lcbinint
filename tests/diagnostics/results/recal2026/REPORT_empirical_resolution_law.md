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

The tolerance semantics and the mixed-rule identity are established, but the
reference-quality audit below changes how the empirical law must be stated.
Rows whose reference is not precise enough are now retained as lower-censored
observations (`Nbin >= N_finest` for that row) instead of being silently
removed. Consequently,
the scalar p99 fits are conditional on reference-certified rows, and a global
99% absolute-error law is not identifiable from this campaign without sharper
references. An `Apoint` diagnostic reduces conditional work, but does not
remove that limitation.

The production C++ selector is intentionally not changed by this report;
wiring any offline policy into `nbin="auto"` is a separate implementation
step. Until that follow-up lands, the current runtime must not be described as
already using these empirical laws.

The handoff decision is fail-closed. The calibrated law may select an initial
resolution only inside a validity domain supported by the reference campaign.
An out-of-domain request must return an explicit unsupported-tolerance status;
it must not silently receive an extrapolated `Nbin`. The fitted extrapolation
is retained in the machine-readable calibration and figures as a diagnostic
for planning the next campaign, but it is not a production fallback or a
99%-coverage claim. In particular, the current evidence does not certify an
absolute-only request at `a_tol=1e-4`.

## 1. Policy

For a magnification $A$, define the scale

$$
 S(A)=\max(|A|,1).
$$

The two allowed error branches are

$$
 B_{\rm abs}=a_{\rm tol},\qquad
 B_{\rm rel}=r_{\rm tol}S(A).
$$

The effective acceptance budget is the larger allowance,

$$
 B_{\rm eff}=\max(B_{\rm abs},B_{\rm rel}),
 \qquad
 \varepsilon_{\rm rel}=\max\left(\frac{a_{\rm tol}}{S(A)},r_{\rm tol}\right).
$$

Equivalently, the calculation is accepted when

$$
 E\le a_{\rm tol}\quad\text{or}\quad
 E\le r_{\rm tol}S(A).
$$

This is the same logical structure as the VBMicrolensing stopping rule: the
calculation continues only while both tests fail. It is not the additive
budget $a_{\rm tol}+r_{\rm tol}S(A)$, and it is not the stricter minimum of
the two allowances.

The resolution selector uses the corresponding two empirical branches:

$$
 N_{\rm abs,g}(a_{\rm tol})
 =\left\lceil C^{\rm abs}_g
 \left(\frac{a_{\rm tol}}{10^{-3}}\right)^{-\beta^{\rm abs}_g}\right\rceil_{\mathcal N},
$$

$$
 N_{\rm rel,g}(r_{\rm tol})
 =\left\lceil C^{\rm rel}_g
 \left(\frac{r_{\rm tol}}{10^{-3}}\right)^{-\beta^{\rm rel}_g}\right\rceil_{\mathcal N},
$$

where $g\in\{\text{Cartesian},\text{polar}\}$, and
$\lceil\cdot\rceil_{\mathcal N}$ rounds upward to the measured ladder

$$
 \mathcal N=\{4,6,8,10,12,16,24,32,40,50,64,80,100,128,160,200,256,320,400\}.
$$

For a mixed request, the adopted rule is

$$
 \boxed{N_{\rm mix,g}=\min\left(N_{\rm abs,g}(a_{\rm tol}),
                                  N_{\rm rel,g}(r_{\rm tol})\right).}
$$

The `min` is not an arbitrary optimization. On a fixed resolution ladder,
increasing the allowed error from either branch to $B_{\rm eff}$ can only
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

For each requested budget, the exact required resolution is the first ladder
bucket for which the result and every finer measured bucket satisfy

$$
 |A(N)-A_{\rm ref}|\le B_{\rm eff}(A_{\rm ref}).
$$

This persistent-crossing definition rejects an isolated lucky crossing. A
reference is exact evidence only when its stored relative uncertainty obeys

$$
 u_{\rm ref}\le 0.1\,B_{\rm eff}/S(A_{\rm ref}).
$$

If that gate fails, the row is not deleted. It is recorded as a lower-censored
observation, `Nbin >= N_finest`, because the stored ladder cannot establish a finite
crossing. Invalid reference records remain excluded as invalid data. Discovery
data determine fits; the holdout is never used to choose coefficients or safety
offsets.

The resulting record counts are:

| branch | grid | discovery total | discovery exact | discovery censored | holdout total | holdout exact | holdout censored |
|---|---|---:|---:|---:|---:|---:|---:|
| relative | Cartesian | 51,840 | 49,846 | 1,994 | 19,296 | 18,528 | 768 |
| relative | polar | 51,840 | 49,832 | 2,008 | 19,296 | 18,522 | 774 |
| absolute | Cartesian | 51,840 | 40,183 | 11,657 | 19,296 | 15,053 | 4,243 |
| absolute | polar | 51,840 | 40,134 | 11,706 | 19,296 | 15,035 | 4,261 |

This distinction matters for a p99 claim. In the absolute branch, even at
$a_{\rm tol}=10^{-2}$, 7.2% of the discovery records are censored; the
population p99 is therefore only known to be at least 400. At
$a_{\rm tol}=10^{-4}$, the censored fraction is about 54%.

## 3. Fitting procedure

For each grid and each branch:

1. Compute the required persistent-crossing bucket for every discovery row and
   tolerance level, retaining reference-limited rows as lower-censored
   `Nbin >= N_finest` observations.
2. Summarize exact rows by p99 and report the lower-censored fraction and
   lower-bound p99 separately.
3. Fit the p99 values in base-two logarithms with a one-variable power law.
4. Choose the smallest discovery-side upward offset that reaches 99% coverage
   overall and at every available tolerance level on exact rows.
5. Round the continuous prediction upward to the next supported bucket.
6. Apply the frozen law to the independent holdout.

The logarithmic form is

$$
 \log_2N_{99}=\log_2C+\beta\log_2(10^{-3}/\tau),
$$

where $\tau=a_{\rm tol}$ for the absolute branch and
$\tau=r_{\rm tol}$ for the relative branch. Base two is used because one
step in the ladder is naturally a resolution refinement, and the slope has a
direct interpretation: halving the requested tolerance multiplies the p99
resolution by $2^\beta$.

No geometry feature is included in the scalar Nbin law. Those quantities remain
available for route selection and diagnostics. In particular, the absolute
branch has a useful diagnostic candidate,

$$
 \log_2 N_{99,g}^{\rm abs}
 =\alpha_g+\beta_g\log_2\left(\frac{10^{-3}}{a_{\rm tol}}\right)
 +\gamma_g\log_2\max(A_{\rm point},1),
$$

which is evaluated below. It improves conditional work, but it cannot turn a
reference-limited lower bound into an exact observation, so it is not yet the
population-level production law.

## 4. Fitted laws

The scalar discovery fits and their independent holdout coverage on exact,
reference-certified rows are:

| grid | $C^{\rm rel}$ | $\beta^{\rm rel}$ | relative exact coverage | $C^{\rm abs}$ | $\beta^{\rm abs}$ | absolute exact coverage |
|---|---:|---:|---:|---:|---:|---:|
| Cartesian | 45.32 | 0.4767 | 99.67% | 140.47 | 0.1103 | 99.67% |
| polar | 94.57 | 0.5952 | 99.81% | 201.05 | 0.2286 | 99.75% |

The relative branch is the efficient main calibration on the certified subset.
At the common default
$r_{\rm tol}=10^{-3}$, the p99 initial buckets are Cartesian 50 and polar
100. The absolute branch is intentionally conservative: at
$a_{\rm tol}=10^{-3}$, its p99 buckets are Cartesian 160 and polar 200.
That difference is the measured cost of asking for an absolute error that does
not relax as the magnification grows.

### 4.1 Relative branch table

The percentages in this table are coverage of exact rows. Lower-censored rows
are shown separately in the figures and machine-readable JSON.

| $r_{\rm tol}$ | Cartesian bucket | coverage | polar bucket | coverage |
|---:|---:|---:|---:|---:|
| $1.0\times10^{-2}$ | 16 | 99.58% | 32 | 100.00% |
| $5.0\times10^{-3}$ | 24 | 99.58% | 40 | 99.86% |
| $3.0\times10^{-3}$ | 32 | 99.67% | 50 | 100.00% |
| $2.0\times10^{-3}$ | 40 | 99.86% | 64 | 99.86% |
| $1.0\times10^{-3}$ | 50 | 99.81% | 100 | 99.77% |
| $5.0\times10^{-4}$ | 64 | 99.57% | 160 | 99.76% |
| $3.0\times10^{-4}$ | 100 | 99.71% | 200 | 99.46% |
| $2.0\times10^{-4}$ | 100 | 99.44% | 256 | 99.54% |
| $1.0\times10^{-4}$ | 160 | 99.83% | 400 | 100.00% |

### 4.2 Absolute branch table

These are also exact-row conditional coverages. They must not be read as a 99%
claim for the full holdout population.

| $a_{\rm tol}$ | Cartesian bucket | coverage | polar bucket | coverage |
|---:|---:|---:|---:|---:|
| $1.0\times10^{-2}$ | 128 | 99.85% | 128 | 99.60% |
| $5.0\times10^{-3}$ | 128 | 99.45% | 160 | 99.75% |
| $3.0\times10^{-3}$ | 128 | 99.38% | 160 | 99.59% |
| $2.0\times10^{-3}$ | 160 | 100.00% | 200 | 99.89% |
| $1.0\times10^{-3}$ | 160 | 99.83% | 256 | 99.89% |
| $5.0\times10^{-4}$ | 160 | 99.58% | 256 | 99.76% |
| $3.0\times10^{-4}$ | 200 | 99.86% | 320 | 99.86% |
| $2.0\times10^{-4}$ | 200 | 99.46% | 400 | 99.54% |
| $1.0\times10^{-4}$ | 200 | 99.50% | 400 | 100.00% |

The absolute table is conditional, not a population-wide guarantee. Its median
work versus the measured exact requirement is 256 for Cartesian and 178 for
polar, compared with 16 and 64 for the relative branch. The lower-censored
audit shows that the finite p99 is not identifiable across the full population
with the current references.

### 4.3 Apoint diagnostic for the absolute branch

Fitting the two-feature candidate above on the same discovery/holdout split
gives:

| grid | $C_g$ at $A_{\rm point}=1$ | $\beta_g$ | $\gamma_g$ | exact holdout coverage | lower-bound coverage | median predicted Nbin |
|---|---:|---:|---:|---:|---:|---:|
| Cartesian | 56.46 | 0.4265 | 0.3412 | 99.49% | 89.22% | 100 |
| polar | 117.52 | 0.5338 | 0.2458 | 99.57% | 95.32% | 160 |

The `Apoint` term is therefore real and useful: the required resolution grows
with point-source magnification, and the conditional median prediction drops
relative to the scalar absolute law. However, the lower-bound coverage still
falls below 99% because the reference-limited rows only provide lower bounds.
This candidate is retained as a diagnostic until the reference ladder is made
precise enough to identify the population p99.

### 4.4 Why the current `a_tol=1e-4` result is not a supported law

At `a_tol=1e-4`, the reference gate requires an absolute reference
uncertainty of at most `1e-5`, because the calibration deliberately keeps a
factor-of-ten margin between reference uncertainty and requested budget. The
stored campaign does not meet that requirement for most rows:

| grid | holdout records at `a_tol=1e-4` | reference-uncertainty censored | actual ladder-limit censored |
|---|---:|---:|---:|
| Cartesian | 2,144 | 1,151 | 0 |
| polar | 2,144 | 1,151 | 10 |

The corresponding discovery counts are 3,120 reference-uncertainty-censored
Cartesian rows and 3,120 reference-uncertainty-censored plus 22
ladder-limited polar rows, out of 5,760 rows per grid. Thus the large apparent
failure population is primarily a reference-resolution problem, not evidence
that thousands of runs require more than `Nbin=400`. Increasing the ladder is
still appropriate for the small set of genuine ladder-limit rows, but it does
not repair the reference floor.

## 5. Mixed-tolerance validation

The mixed test used all $9\times9=81$ positive pairs from the levels in the
two tables. For each pair and each holdout row, the script did two independent
things:

* It measured the required bucket directly with the effective budget
  $\max(a_{\rm tol},r_{\rm tol}S(A_{\rm ref}))$.
* It predicted `min(N_abs, N_rel)` using coefficients fixed before looking at
  the holdout row.

The direct mixed required bucket agreed with
$\min(N_{\rm required,abs},N_{\rm required,rel})$ in every comparable,
reference-certified case. The coverage result is therefore conditional on
that comparable subset:

| grid | mixed pairs | minimum coverage | median coverage | pairs at or above 99% |
|---|---:|---:|---:|---:|
| Cartesian | 81 | 99.44% | 99.81% | 81/81 |
| polar | 81 | 99.46% | 100.00% | 81/81 |

The identity check covered 132,341 Cartesian and 132,137 polar row-pair cases;
both had zero mismatches. Thus the less-demanding-branch rule is not merely a
logical description of the tolerance budget: it is also the measured
resolution composition on the independent holdout.

The heatmap in the PDF shows all 162 conditional coverage values. The worst cells are
Cartesian at $a_{\rm tol}=2\times10^{-4},r_{\rm tol}=2\times10^{-4}$,
99.44%, and polar at $a_{\rm tol}=3\times10^{-4},r_{\rm tol}=3\times10^{-4}$,
99.46%. Both remain above the predeclared 99% target.

## 6. Figures

[`figures/empirical-resolution-law.pdf`](figures/empirical-resolution-law.pdf)
contains five pages:

1. Relative-branch box-and-whisker distributions of lower-bound required Nbin.
   The box is
   Q1--Q3, the thick vertical bar is the central 68% interval (p16--p84), the
   thin whisker is p5--p95, the red diamond is p99 lower bound, the open square
   is the exact-row fitted bucket, and the purple triangle marks the p99 lower
   bound among censored rows.
2. The same lower-bound figure for the absolute branch.
3. A Cartesian/polar heatmap of all 81 mixed holdout coverages.
4. A representative holdout convergence curve, showing $A(N)$ and
   $|A(N)-A_{\rm ref}|$ against Nbin for both grids.
5. The fitting recipe, equations, fitted constants, and minimum mixed
   coverage in one page suitable for inclusion in a methods supplement.

The Apoint-binned diagnostic is
[`figures/absolute-apoint-boxplots.pdf`](figures/absolute-apoint-boxplots.pdf).
It shows the monotonic increase of required Nbin with Apoint and the onset of
the `400+` reference-limited region.

The representative convergence case is holdout case 48 with the linear
limb-darkening profile, $s=3$, $q=10^{-5}$, and
$\rho=1.49005\times10^{-3}$. At $r_{\rm tol}=10^{-3}$, its measured
required buckets are Cartesian 10 and polar 12. This is an illustration, not
the calibration sample used to choose the law.

## 7. What this result does and does not claim

This report establishes conditional empirical p99 initial-resolution rules
against the stored high-resolution reference. It does not establish a
population-wide 99% law where the lower-censored fraction exceeds 1%; that
requires a sharper reference campaign or an explicitly restricted validity
domain. It also does not establish that the runtime embedded estimator $E$ is
a certified upper bound for the true error. The
separate estimator audit found route-dependent underestimation, so the
resolution law and the estimator certificate must remain separate claims.

### 7.1 Operational handoff policy

The next runtime implementation must apply the following policy:

1. `Nbin=auto` uses the common max-budget semantics and the route-specific
   Cartesian/polar law only when the requested branch is inside its validated
   calibration domain.
2. An unsupported request fails closed with a structured status such as
   `unsupported_tolerance`; the message must distinguish a reference-quality
   limitation from an actual resolution-ladder limit.
3. An explicit `Nbin` remains an allowed expert override, but the result is
   reported as having no empirical p99 guarantee when it lies outside the
   validated domain.
4. The scalar and `Apoint` extrapolations remain available to diagnostics and
   future calibration scripts. They are never an automatic fallback for
   `Nbin=auto`, and no paper claim may count them as validated coverage.

For the current campaign, absolute `a_tol=1e-4` is explicitly outside the
supported automatic domain. The absolute coefficients in §4 are retained as
conditional calibration numbers, not as authorization to accept that request.
The supported domain should be widened only after a sharper reference campaign
has reduced the censored fraction enough to identify the target p99, followed by
a frozen-policy holdout test.

The recommended next campaign is targeted rather than a blind rerun: improve
the reference uncertainty to `<=1e-5` for stratified hard cases, and
extend the ladder beyond 400 only for the 10 polar holdout / 22 polar discovery
rows that actually reached the present ladder limit. The holdout remains
reserved for final validation after the policy is frozen.

It also does not claim that a fixed p99 bucket is the fastest answer for every
row. The scalar rule deliberately pays for the hard tail, while the Apoint
diagnostic shows a possible conditional reduction. An eventual runtime
implementation can use the common policy for acceptance and refine only when a
route-specific indicator requires it, provided that the fail-closed behavior
and the holdout coverage are preserved.

The study covers binary Cartesian and polar inverse-ray ladders. It does not
fit source-plane quadrature here because its production resolution coordinate
is not stored as the same complete $A(N)$ ladder. That route should adopt the
same budget semantics but receive its own calibrated resolution law.

Finally, $A_{\rm ref}$ appears in the offline definition only because it is
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

MPLBACKEND=Agg PYTHONPATH=. python -m tests.diagnostics.recal2026.plot_absolute_lower_bounds \
  --output tests/diagnostics/results/recal2026/figures/absolute-error-boxplot.pdf

MPLBACKEND=Agg PYTHONPATH=. python -m tests.diagnostics.recal2026.plot_absolute_apoint_bins \
  --output tests/diagnostics/results/recal2026/figures/absolute-apoint-boxplots.pdf

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
> $B=\max(a_{\rm tol},r_{\rm tol}\max(|A|,1))$, while Cartesian and polar
> integration retain separate empirical convergence constants. Reference
> records that cannot resolve the requested budget are retained as lower-
> censored observations rather than discarded. The scalar laws and mixed
> holdout figures therefore describe the reference-certified subset; a
> population-wide 99% law requires a sharper reference campaign. For the
> absolute branch, an additional diagnostic law using
> $\log_2\max(A_{\rm point},1)$ reduces conditional work, but does not remove
> the lower-censoring limitation.
