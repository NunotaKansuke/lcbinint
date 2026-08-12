# Paper-facing report: cache-warm finite-source kernel speed comparison

Date: 2026-08-11
Primary dataset: [`pure_kernel_eps3_eps4_20260809`](../results/recal2026/pure_kernel_eps3_eps4_20260809/merged/report/REPORT_pure_kernel.md)
Benchmark commit: `ed621f8d56d2358e0bf730edb69929f0dcede61b`
Timing mode: `pure_kernel_cache_warm`

Implementation note (2026-08-12): the benchmark harness has since gained a
direct-source timing path, `_evaluate_preplanned_xy(source_x, source_y, ...)`,
which bypasses per-epoch trajectory reconstruction.  The preplanned Cartesian
kernel also reuses the point-source magnification already obtained while
building image seeds instead of repeating the point-lens solve for its walk
hint.  The dataset and summary tables below predate that change and are kept
as the historical 2026-08-09 measurement; they should not be relabeled as a
direct-XY rerun.

## Executive conclusion

The defensible paper-level conclusion is not a universal speed advantage for
lcbinint. It is profile- and tolerance-dependent:

- For a uniform source, VBM is faster in every measured case in this corpus.
- For linear limb darkening with `c=0.5`, lcbinint wins about 62--67% of the
  measured epochs at external tolerances `1e-3` and `1e-4`, respectively.
- The clearest operational predictor is the Nbin required by the lcbinint
  calculation. `d/rho` gives a trend, but does not define a reliable
  one-dimensional crossover law by itself.
- These are cache-warm, one-epoch integrator results. They are not a claim
  about the total cost of a fresh `LightCurve` construction or a complete
  production light curve.

The result should therefore be stated as: **lcbinint is competitive, and often
faster, for sufficiently demanding linearly limb-darkened finite-source
integrals when the certified grid remains at low-to-moderate Nbin; VBM remains
the clear choice for uniform sources in this test.**

## What was compared

The speed ratio is

```text
R = t_VBM / t_LCB-in
```

Thus `R > 1` means that lcbinint is faster. The comparison is restricted to
epochs for which the production route is a finite-source grid route. Point
source, hexadecapole, and source-plane quadrature shortcuts are excluded.

For each epoch, Cartesian and polar lcbinint grids were both evaluated. The
faster grid satisfying the certified resolution requirement was retained. The
Nbin value was read from the existing warm-up/speed-discovery corpus; Nbin
search was not included in the timing.

## Timing protocol

The lcbinint worker evaluates two identical epochs through the native
`_evaluate_preplanned` path. The first epoch constructs the `LensModel` and
caustic cache. Only the second epoch's native integration time is recorded.
VBM is warmed once and then its direct finite-source call is timed at the
requested external tolerance.

Consequently, the following are outside the reported lcbinint time:

- `LensModel` construction;
- first-use caustic-cache construction;
- Nbin search;
- `LightCurve.info()` and the later geometry annotation used for plots.

This isolates the repeated finite-source integration kernel. It intentionally
does not measure the public scalar `binary_ray_shooting()` call including its
per-call object construction.

## Dataset

- 160 binary-lens cases, seed `20260809`;
- `q >= 1e-4`;
- requested distance factors `0, 0.25, 0.5, 0.8, 0.95, 1.0, 1.1, 1.35,
  1.7, 2.0`;
- requested `d/rho < 2.01`;
- uniform source and linear limb darkening with `c=0.5`;
- external tolerances `1e-3` and `1e-4`;
- reference epochs `(0, 7, 15, 23)`;
- two timing repeats, using the median;
- point timeout: 300 s; job timeout disabled;
- 854 jobs and 3,416 reference epochs in total.

The raw merged result is [`results.json`](../results/recal2026/pure_kernel_eps3_eps4_20260809/merged/results.json).

## Main results

| profile | target | measured | lcbinint win rate | median `R` | p10--p90 `R` | median Nbin |
|---|---:|---:|---:|---:|---:|---:|
| uniform | `1e-3` | 1,132 | 0.0% | 0.091 | 0.018--0.195 | 24 |
| uniform | `1e-4` | 512 | 0.0% | 0.038 | 0.006--0.102 | 50 |
| linear LD | `1e-3` | 1,122 | 61.8% | 1.351 | 0.195--3.036 | 16 |
| linear LD | `1e-4` | 646 | 66.7% | 1.576 | 0.350--6.560 | 50 |

The two unresolved linear-LD epochs at each target are retained in the raw
record and excluded only from the measured speed ratio. No unresolved epochs
occur for the uniform profile.

## Dependence on geometry and Nbin

The refined `actual d/rho` used in the plots is the nearest caustic distance
computed from the lens geometry divided by `rho`. It is not the requested
sampling-factor label and it is computed after timing as a geometry annotation.

For linear LD, smaller actual `d/rho` is associated with a higher lcbinint win
rate, but the separation is broad. At `1e-4`, the win rates are 82.3% for
`0 <= d/rho < 0.3`, 49.6% for `0.3 <= d/rho < 0.8`, and 30.8% for
`0.8 <= d/rho < 1.05`. This is a trend, not a usable `d/rho`-only decision
boundary.

The stronger practical split is the selected Nbin. For linear LD at `1e-4`,
the median speed ratio changes from approximately 2.96--4.93 for Nbin up to
24, to 1.53 for Nbin 24--50, and to 0.97 for Nbin 50--101. In other words,
once the lcbinint grid has to become large, its kernel cost catches up with and
then exceeds VBM.

The full stratification and figures are in
[`REPORT_rho_d_over_rho_speed.md`](../results/recal2026/pure_kernel_eps3_eps4_20260809/merged/rho_d_over_rho_speed/REPORT_rho_d_over_rho_speed.md):

- [`R` versus actual `d/rho`, split by `rho`](../results/recal2026/pure_kernel_eps3_eps4_20260809/merged/rho_d_over_rho_speed/figures/R_vs_actual_d_over_rho_by_rho.pdf);
- [`R` versus `rho`, colored by actual `d/rho`](../results/recal2026/pure_kernel_eps3_eps4_20260809/merged/rho_d_over_rho_speed/figures/R_vs_rho_colored_actual_d_over_rho.pdf).

## Interpretation for the paper

The speed ratio is controlled by two costs, not one geometric scalar:

```text
geometry and source profile -> required Nbin -> lcbinint integration cost
local binary geometry       -> VBM adaptive integration cost
```

The ratio of those two costs produces scatter even at similar `d/rho`. The
source profile is essential: limb darkening makes the VBM calculation more
expensive while the lcbinint grid remains competitive at the tested resolutions.
For a uniform source, that advantage is absent in this corpus.

The data do not support a claim that `A_finite` or `d/rho` alone determines the
winner. A future production switch should use the retained warm-up result or a
multivariate predictor including at least the selected Nbin, source profile,
and the relevant binary/source geometry.

## Reproduction

The corrected benchmark harness is
[`bench_grid_vs_vbm_pure_kernel.py`](bench_grid_vs_vbm_pure_kernel.py). The
merged result can be regenerated from its five part results with
[`merge_benchmark_parts.py`](merge_benchmark_parts.py), and the compact report
with [`report_pure_kernel.py`](report_pure_kernel.py).

To regenerate the geometry-stratified figures from the merged result:

```bash
MPLBACKEND=Agg python \
  tests/diagnostics/recal2026/plot_pure_kernel_rho_d_over_rho.py \
  --results \
  tests/diagnostics/results/recal2026/pure_kernel_eps3_eps4_20260809/merged/results.json \
  --output \
  tests/diagnostics/results/recal2026/pure_kernel_eps3_eps4_20260809/merged/rho_d_over_rho_speed
```

The measurement used the native extension at
`build/lcbinint/_lcbinint.cpython-310-x86_64-linux-gnu.so`; its SHA-256 was
`c8ca2226277ec51e902bd55a1fd34eb66d8df13ceacd2efe96a49a1f6dda7fee`.

## Scope and limitations

This is a controlled kernel comparison, not an end-to-end inference benchmark.
The result does not include Nbin discovery, does not include the one-time
caustic-cache construction, and does not include Python/API overhead on the
lcbinint side. The measured corpus is broad in binary geometry but is not an
independent parameter holdout. The current result is therefore suitable as a
paper diagnostic and mechanism study, not as a universal runtime law.

The previous cache-cold and exploratory speed directories are deliberately not
used as primary evidence here because they mix a one-time lcbinint setup cost
into each scalar epoch. They should remain historical audit material rather
than being combined with this table.
