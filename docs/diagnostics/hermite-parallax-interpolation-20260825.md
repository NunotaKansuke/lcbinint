# Hermite parallax-table experiment

Branch: `hermite-parallax-interpolation`
Date: 2026-08-25

The Earth ephemeris now uses cubic Hermite interpolation with the tabulated
Cartesian velocity as its node tangent. Spacecraft tables keep their public
`(JD, RA_deg, Dec_deg, distance_AU)` format; after Cartesian conversion, node
tangents are estimated with a three-point Lagrange derivative. Native and JAX
both retain those tangents when an ephemeris is restricted by `t_lim`.

## Accuracy

Measured with `tests/diagnostics/benchmark_parallax_interpolation.py`:

| Reference | Linear | Hermite | Change |
| --- | ---: | ---: | ---: |
| Known cubic, max position error | `8.79e-1` | `6.22e-15` | exact to float precision |
| Earth table, max position error vs local degree-7 reconstruction (AU) | `3.19e-5` | `3.82e-10` | ~`8.3e4` lower |
| Smooth spacecraft trajectory, max position error (AU) | `2.24e-4` | `3.18e-5` | ~`7.0` lower |

The Earth reference is a local degree-7 Lagrange reconstruction of the same
daily table. The spacecraft reference is analytic and the table is sampled at
one-day intervals.

## Speed

The native measurement used 100,000 epochs, Release builds, and
`OMP_NUM_THREADS=1`:

| Path | Linear | Hermite | Ratio |
| --- | ---: | ---: | ---: |
| Annual parallax | 28.97 ms | 35.59 ms | 1.23x |
| Spacecraft parallax | 34.39 ms | 41.99 ms | 1.22x |

An isolated JAX x64 measurement on the same 100,000-epoch geometry gave
approximately `23.16 → 25.61 ms` for annual parallax (`1.11x`) and
`3.52 → 6.00 ms` for spacecraft parallax (`1.70x`). JAX timing is more
sensitive to CPU scheduling; rerun the diagnostic script on the target host
before using these numbers for capacity planning.

## Verification

- `ctest --test-dir build --output-on-failure`: 1/1 passed.
- Hermite, higher-order, time-window, spacecraft, and cross-backend parallax
  tests: 27 passed.
- The full pytest run reached 455 passed and 3 failed. The parallax failure was
  the intentional comparison against jacscanomaly's old linear interpolator
  and is updated in this branch. The other two failures reproduce with the
  pre-existing site-packages extension and are unrelated to parallax:
  `test_hybrid_keeps_calibrated_tiny_high_magnification_polar_path` and
  `test_refinement_leaves_geometries_without_a_thin_component_alone[planetary]`.
