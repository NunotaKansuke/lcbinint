# Final binary resolution-law handoff

This file is the short entry point for the final calibration. The complete
paper-facing report, machine-readable result, and PDF figure are:

- [`REPORT_final_apoint_calibration.md`](final_apoint_validation/REPORT_final_apoint_calibration.md)
- [`final_apoint_validation.json`](final_apoint_validation/final_apoint_validation.json)
- [`final-apoint-calibration.pdf`](figures/final-apoint-calibration.pdf)

## Frozen production rule

For $S(A)=\max(|A|,1)$, all binary routes use the common max-budget contract

$$
B=\max(a_{\rm tol},r_{\rm tol}S(A)).
$$

Absolute and relative tolerances are alternative allowances. The initial
mixed resolution is therefore

$$
N_{\rm mix}=\left\lceil\min(N_{\rm abs},N_{\rm rel})\right\rceil,
$$

with a positive-integer `ceil` and the `max_source_bins` cap. Automatic route
selection is polar for $A_{\rm point}\ge200$ and Cartesian otherwise.

The final laws are

$$
N_{\rm rel,g}=C_{\rm rel,g}
\left(\frac{r_{\rm tol}}{10^{-3}}\right)^{-\beta_{\rm rel,g}},
$$

$$
N_{\rm abs,g}=C_{\rm abs,g}
\left(\frac{a_{\rm tol}}{10^{-3}}\right)^{-\beta_{\rm abs,g}}
\max(A_{\rm point},1)^{\gamma_g}.
$$

| grid | $C_{\rm rel}$ | $\beta_{\rm rel}$ | $C_{\rm abs}$ | $\beta_{\rm abs}$ | $\gamma$ |
|---|---:|---:|---:|---:|---:|
| Cartesian | 49.59298071 | 0.4767022 | 138.06382198 | 0.4265493 | 0.3411985 |
| polar | 105.29723706 | 0.5952071 | 396.47500161 | 0.5337642 | 0.2458039 |

The supported automatic domains are $10^{-4}\le r_{\rm tol}\le10^{-2}$ and
$2\times10^{-4}\le a_{\rm tol}\le10^{-2}$. Absolute $10^{-4}$ is retained as
a diagnostic level but fails closed in automatic mode because the current
reference campaign cannot certify it.

## Validation status

The final calibration uses 6,000 discovery rows and 2,200 independent holdout
rows, each with the same 19-level Cartesian/polar ladder. The formal grid is
9 relative levels, 8 supported absolute levels, and all $8\times9=72$ positive
mixed pairs.

On exact/reference-certified holdout rows, every formal level and all 72 mixed
pairs meet the predeclared 99% coverage target. The minimum exact coverage is
99.216% for Cartesian/auto-route and 99.461% for polar. The mixed identity
`N_required,mix = min(N_required,abs, N_required,rel)` had zero mismatches.

These are conditional coverage numbers. Rows whose reference uncertainty is
too large are retained as lower-censored observations; lower-bound coverage is
below 99% in the affected tails. The report does not claim population-wide
99% coverage beyond what the reference campaign can identify.

The Apoint absolute law is deliberately conservative: its safety envelope was
chosen from discovery before holdout evaluation. The larger constants are the
cost of making the adopted rule and its integer-ceil implementation agree on
the independent holdout without adding a runtime probe or hidden fallback.

## Reproduction

```sh
cmake -S . -B build -DLCBININT_BUILD_PYTHON=ON
cmake --build build --target test_core lcbinint_python -j8
PYTHONPATH=. python -m tests.diagnostics.recal2026.final_apoint_validation \
  --discovery tests/diagnostics/results/recal2026/discovery \
  --holdout tests/diagnostics/results/recal2026/holdout \
  --output-dir tests/diagnostics/results/recal2026/final_apoint_validation
```
