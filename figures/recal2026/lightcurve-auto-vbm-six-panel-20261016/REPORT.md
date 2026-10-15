# LightCurve auto vs VBM: six-panel comparison

Four six-panel figures compare the actual public binary `LightCurve.info` auto path against the saved matched VBM direct-binding measurements, for uniform and linear limb darkening at relative targets 1e-3 and 1e-4.

The 7,216 geometries are grouped by case, impact-parameter bin, and brightness profile. Each full group of sampled epochs is passed together to `LightCurve.info`, with one untimed warm batch followed by three measured batches. The median batch wall time is divided by the number of epochs in that batch for the per-epoch speed ratio. `LightCurve` construction and the epoch array are outside the timer. The auto route keeps its default `inverse_ray_grid=auto`, `nbin=auto`, 512-bin cap, and default point-source/hexadecapole short-circuits; only the coordinate frame and requested tolerance are fixed to match the saved cohort. The VBM measurements are the fresh matched direct `BinaryMag2`/`BinaryMagDark` timings in the Polar six-panel campaign (one warm and three timed calls per epoch). Since VBM uses a direct kernel binding while the numerator uses the public LightCurve API, this speed ratio includes different wrapper and batching boundaries and is an empirical comparison, not a perfectly symmetric kernel-only benchmark.

The upper panels show the cell median `t_VBM / t_LightCurve,auto` (values above one favor the LightCurve auto path). The lower panels show the cell p95 relative difference from the saved RelTol=1e-6 reference after the documented three reference adjudications. This is an empirical agreement diagnostic, not a formal error certificate. All conditions, including non-convergence and actual-error failures, remain in `joined.tsv`.

All four speed maps share the same reciprocal factor-of-two color bins, with boundaries `R = 2^k` for `k = -6,...,6`. The lower tail is `<1/64` and the upper tail is `>=64`; blue denotes `R<1` (VBM faster) and red denotes `R>1` (LightCurve auto faster).

| profile | target | conditions | converged | reference pass | false accepts | faster than VBM | median VBM / auto | median auto ms/epoch | p95 error | max error | methods |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| uniform | 0.001 | 3608 | 3607 | 3599 | 8 | 0/3608 | 0.267x | 0.2526 | 0.0003154 | 0.001624 | `{"inverse_ray_cartesian": 1756, "inverse_ray_polar": 1849, "point_source": 3}` |
| uniform | 0.0001 | 3608 | 3598 | 3601 | 6 | 164/3608 | 0.377x | 0.3382 | 3.204e-05 | 0.001624 | `{"inverse_ray_cartesian": 1674, "inverse_ray_polar": 1934}` |
| linear | 0.001 | 3608 | 3607 | 3604 | 3 | 1928/3608 | 1.085x | 0.2884 | 0.0002652 | 0.001175 | `{"inverse_ray_cartesian": 1655, "inverse_ray_polar": 1950, "point_source": 3}` |
| linear | 0.0001 | 3608 | 3599 | 3602 | 5 | 3141/3608 | 3.834x | 0.3758 | 3.063e-05 | 0.001104 | `{"inverse_ray_cartesian": 1577, "inverse_ray_polar": 2031}` |

Reference differences above the requested tolerance by selected method:

- uniform 0.001: {"inverse_ray_cartesian": 9}; among these, estimator-accepted rows were {"inverse_ray_cartesian": 8}.
- uniform 0.0001: {"inverse_ray_cartesian": 7}; among these, estimator-accepted rows were {"inverse_ray_cartesian": 6}.
- linear 0.001: {"inverse_ray_cartesian": 4}; among these, estimator-accepted rows were {"inverse_ray_cartesian": 3}.
- linear 0.0001: {"inverse_ray_cartesian": 6}; among these, estimator-accepted rows were {"inverse_ray_cartesian": 5}.

The q-rho cohort was designed for parameter-space coverage, not as an iid speed sample. Cells with fewer than eight finite rows are masked. The saved VBM timing identity, LightCurve/native identity, raw per-condition results, and checksums are recorded alongside the figures. Neither the cohort nor the empirical error comparison establishes a general speed guarantee or certified error bound.

Reproduce with `python tests/diagnostics/recal2026/lightcurve_auto_six_panel_20261016.py`.
