# Final Apoint-dependent resolution calibration

Status: **PASS on exact/reference-certified holdout rows; no population-wide 99% claim is made for lower-censored rows.**

This report freezes the current automatic binary selector. The discovery set is used for fitting and safety selection; the holdout set is independent and is not used to choose coefficients.

## Frozen policy

For $S(A)=\max(|A|,1)$, the common dimensional budget is

$$B=\max(a_{\rm tol}, r_{\rm tol}S(A)).$$

The mixed initial resolution is $N_{\rm mix}=\min(N_{\rm abs}, N_{\rm rel})$, rounded upward with `ceil` and capped at `max_source_bins=400`. Automatic routing uses polar for $A_{\rm point}\ge200$ and Cartesian otherwise. Absolute $a_{\rm tol}=10^{-4}$ is retained as a diagnostic level but is outside the production domain.

The fitted laws are

$$N_{\rm rel,g}=C_{\rm rel,g}(r_{\rm tol}/10^{-3})^{-\beta_{\rm rel,g}},$$

$$N_{\rm abs,g}=C_{\rm abs,g}(a_{\rm tol}/10^{-3})^{-\beta_{\rm abs,g}}\max(A_{\rm point},1)^{\gamma_g}.$$

| grid | $C_{\rm rel}$ | $\beta_{\rm rel}$ | $C_{\rm abs}$ | $\beta_{\rm abs}$ | $\gamma$ |
|---|---:|---:|---:|---:|---:|
| cartesian | 49.592981 | 0.4767022 | 138.06382 | 0.4265493 | 0.3411985 |
| polar | 105.29724 | 0.5952071 | 396.475 | 0.5337642 | 0.2458039 |

## Campaign and censoring

Discovery: **6000 rows**; holdout: **2200 rows**. Each row has the same Cartesian/polar 19-level ladder `[4, 6, 8, 10, 12, 16, 24, 32, 40, 50, 64, 80, 100, 128, 160, 200, 256, 320, 400]`. The formal grid is 8 absolute levels and 9 relative levels; the direct three-point auto sweep is only an anchor check, not the fit grid.

A row whose reference uncertainty cannot resolve the requested budget is retained as a lower-censored observation. Therefore the exact coverage below is conditional on reference-certified rows; the lower-bound column is the honest population-level diagnostic. Invalid reference records are excluded from both columns and are counted explicitly.

| branch | grid | dataset | valid records | exact | censored | invalid records | exact coverage | lower-bound coverage |
|---|---|---|---:|---:|---:|---:|---:|---:|
| relative | cartesian | discovery | 51840 | 49846 | 1994 | 2160 | 99.097% | 82.743% |
| relative | polar | discovery | 51840 | 49832 | 2008 | 2160 | 99.381% | 93.906% |
| absolute | cartesian | discovery | 46080 | 37543 | 8537 | 1920 | 100.000% | 90.694% |
| absolute | polar | discovery | 46080 | 37516 | 8564 | 1920 | 100.000% | 99.618% |
| relative | cartesian | holdout | 19296 | 18528 | 768 | 792 | 99.216% | 82.743% |
| relative | polar | holdout | 19296 | 18522 | 774 | 792 | 99.461% | 93.004% |
| absolute | cartesian | holdout | 17152 | 14060 | 3092 | 704 | 99.864% | 89.412% |
| absolute | polar | holdout | 17152 | 14052 | 3100 | 704 | 100.000% | 99.440% |

## Independent holdout result

The following table reports the minimum exact-row coverage over the formal tolerance levels, and the minimum lower-bound coverage over the same levels.

| route | relative exact | relative lower bound | absolute exact | absolute lower bound |
|---|---:|---:|---:|---:|
| cartesian | 99.216% | 82.743% | 99.864% | 89.412% |
| polar | 99.461% | 93.004% | 100.000% | 99.440% |
| auto route | 99.216% | 83.442% | 99.864% | 89.412% |

### Exact-row coverage by tolerance

| branch | tolerance | Cartesian | polar | auto route |
|---|---:|---:|---:|---:|
| relative | 0.01 | 99.579% | 100.000% | 99.906% |
| relative | 0.005 | 99.579% | 99.860% | 99.906% |
| relative | 0.003 | 99.438% | 100.000% | 99.860% |
| relative | 0.002 | 99.485% | 99.859% | 99.813% |
| relative | 0.001 | 99.812% | 99.765% | 99.859% |
| relative | 0.0005 | 99.572% | 99.762% | 99.572% |
| relative | 0.0003 | 99.216% | 99.461% | 99.216% |
| relative | 0.0002 | 99.440% | 99.542% | 99.440% |
| relative | 0.0001 | 99.256% | 100.000% | 99.256% |
| absolute | 0.01 | 100.000% | 100.000% | 100.000% |
| absolute | 0.005 | 100.000% | 100.000% | 100.000% |
| absolute | 0.003 | 100.000% | 100.000% | 100.000% |
| absolute | 0.002 | 100.000% | 100.000% | 100.000% |
| absolute | 0.001 | 100.000% | 100.000% | 100.000% |
| absolute | 0.0005 | 99.940% | 100.000% | 99.940% |
| absolute | 0.0003 | 99.864% | 100.000% | 99.864% |
| absolute | 0.0002 | 100.000% | 100.000% | 100.000% |

The mixed validation uses all $8\times9=72$ supported positive absolute/relative pairs. The identity check compares the direct mixed persistent crossing with the minimum of the two pure required resolutions.

| route | pairs | minimum exact coverage | pairs >=99% | minimum lower-bound coverage | identity mismatches |
|---|---:|---:|---:|---:|---:|
| Cartesian | 72 | 99.216% | 72/72 | 86.194% | 0 |
| polar | 72 | 99.461% | 72/72 | 93.004% | 0 |
| auto route | 72 | 99.216% | 72/72 | 86.894% | 0 |

## Interpretation

The formal exact-row holdout result clears the predeclared 99% target for every branch, formal level, and mixed pair. This is an empirical validation statement for reference-certified rows, not a claim that the unresolved reference tail is covered: lower-bound coverage is below 99% where the campaign reference floor is too coarse.

The relative constants are the earlier discovery fits multiplied by fixed
log2 safety offsets 0.13 (Cartesian) and 0.155 (polar), selected to clear 99%
at every discovery tolerance. The Apoint constants use the corresponding raw
conditional fits with discovery-side offsets 1.28997 and 1.75437; these offsets
cover every exact discovery observation at every supported absolute level.

The absolute Apoint law is intentionally conservative. Its safety factor was fixed from discovery data before holdout evaluation, and the resulting larger C values are the cost of retaining a single paper-defensible automatic rule rather than a hidden route-specific fallback. The unsupported absolute boundary at $10^{-4}$ remains fail-closed in the C++ API.

## Reproduction

```sh
cmake -S . -B build -DLCBININT_BUILD_PYTHON=ON
cmake --build build --target test_core lcbinint_python -j8
PYTHONPATH=. python -m tests.diagnostics.recal2026.final_apoint_validation \
  --discovery tests/diagnostics/results/recal2026/discovery \
  --holdout tests/diagnostics/results/recal2026/holdout \
  --output-dir tests/diagnostics/results/recal2026/final_apoint_validation
```
