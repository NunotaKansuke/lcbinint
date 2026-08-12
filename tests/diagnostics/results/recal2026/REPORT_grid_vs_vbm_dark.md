# Fair grid-integrator timing against direct VBMicrolensing

This benchmark compares the integration kernels themselves.  It does not use
VBMicrolensing's `BinaryLightCurve` production dispatcher: uniform sources use
`BinaryMag`, and linear limb darkening uses `BinaryMagDark` directly.

The `BinaryMagDark` call is deliberately written as

```text
BinaryMagDark(s, q, -x, y, rho, 1e-12)
```

with `a1=c` configured on the VBM object.  The sixth argument is the absolute
accuracy; it is not the limb-darkening coefficient.

## Design

- 112 blocks: four geometries from each of the 14 existing d/rho strata
  `(0, 0.25, 0.5, 0.8, 0.95, 1, 1.1, 1.35, 1.7, 2, 3, 5, 10, 30)`.
- Both `uniform` and `linear (c=0.5)` source profiles.
- Relative targets `1e-2`, `1e-3`, and `1e-4`.
- Four reference epochs per block, at indices `(0, 7, 15, 23)`.
- The existing Nbin ladder supplies candidates.  Each candidate is re-evaluated
  in the current build and the smallest passing Nbin is retained independently
  at each reference epoch.  Cartesian and polar are both timed, and the faster
  passing grid is selected per epoch.
- VBM sweeps `RelTol` from the target down to `1e-7` and selects the fastest
  direct-integrator setting that passes the same reference budget.
- Timings are marginal integration timings after the object/cache warm-up;
  two repeats are reduced by their median.

The reported ratio is `VBM seconds / lcbinint seconds`; values above one mean
that lcbinint is faster.  Rows whose stored reference floor was too large for a
target were excluded.  One endpoint of the `case=97, d/rho=0.8` block remains
outside the Nbin ladder (Nbin 400); it is retained in the raw output as
unqualified rather than silently counted.

## Results

Each cell is `median ratio / lcbinint win rate`, where the win rate is the
fraction with ratio greater than one.  “Inner” means d/rho < 1 and “outer”
means d/rho >= 5.

| profile | target | all | inner | outer | selected Nbin median |
|---|---:|---:|---:|---:|---:|
| uniform | 1e-2 | 0.058 / 0% | 0.031 / 0% | 0.096 / 0% | 16 |
| uniform | 1e-3 | 0.075 / 0% | 0.047 / 0% | 0.170 / 0% | 24 |
| uniform | 1e-4 | 0.050 / 0.9% | 0.064 / 0% | 0.059 / 0% | 50 |
| linear | 1e-2 | 0.118 / 0.5% | 0.163 / 1.5% | 0.097 / 0% | 16 |
| linear | 1e-3 | 0.230 / 18.1% | 0.601 / 32.8% | 0.181 / 0% | 16 |
| linear | 1e-4 | 0.354 / 25.2% | 0.787 / 43.6% | 0.139 / 0% | 50 |

The direct comparison therefore supports the following reading:

- Uniform-source VBM remains faster in this integration-only measurement,
  including the high-magnification inner region.
- Linear limb darkening changes the crossover substantially.  At `1e-3` and
  `1e-4`, lcbinint wins a sizeable fraction of inner/crossing epochs, and its
  advantage grows as the requested accuracy tightens.
- At d/rho >= 5, VBM remains faster for both source profiles; the expensive
  grid route is not competitive far from the caustic.

## Focused linear-LD branch

The broad sample hides a strong branch.  We therefore re-ran every usable
linear-LD block in the existing corpus with `d/rho < 1` and the tested source
radii `0.01 <= rho < 0.1` (the actual sampled radii reach about 0.0755).
This was 59 blocks and 235 usable reference epochs at `1e-4`, plus 192 jobs at
`1e-3` and `1e-2`.

| additional condition | target | points | lcbinint win rate | median ratio |
|---|---:|---:|---:|---:|
| all focused blocks | 1e-4 | 235 | 86.4% | 3.26 |
| `q >= 1e-5`, `d/rho < 0.8` | 1e-4 | 140 | 96.4% | 5.19 |
| `q >= 0.1`, `d/rho < 0.8`, `A_point > 10` | 1e-4 | 36 | 100% | 10.81 |
| `q >= 0.1`, `d/rho < 0.8`, `A_point > 10` | 1e-3 | 48 | 89.6% | 3.70 |
| `q >= 0.1`, `d/rho < 0.8`, `A_point > 10` | 1e-2 | 48 | 41.7% | 0.96 |

The opposite branch is also informative: in the focused sample, `q < 1e-5`
had 0% wins at `1e-3` and a median ratio of 0.43 at `1e-4`.  Thus q should be
part of the split; d/rho alone is not enough.

This suggests a candidate runtime policy for linear limb darkening: consider
the LCB grid path first only for high-precision requests (`reltol <= 1e-3`),
`d/rho < 0.8`, `rho` in the tested finite-source range, `q >= 1e-5`, and
preferentially `A_point > 10`.  The stricter `q >= 0.1` branch is the strongest
measured pocket, but it needs more independent geometries before being treated
as a universal law.  The warm-up should continue to measure Cartesian and
polar rather than hard-code one grid; in this focused sample polar won more
often (91.7% versus 78.0% at `1e-4`).

The machine-readable per-epoch measurements are in `results.json` beside this
report.  The reproducible harness is
`tests/diagnostics/recal2026/bench_grid_vs_vbm_dark.py`.

## Publication-ready regime maps

The regime-map script is
`tests/diagnostics/recal2026/plot_vbm_regime_maps.py`.  It plots every usable
reference epoch as a scatter point and uses the same ratio throughout:

```text
R = t_VBM / t_lcbinint
```

The diverging colour scale is centered exactly at `R=1`: red means that
lcbinint is faster, blue means that VBM is faster, and white means equal
timing.  The overview figure projects `A_point` against `d/rho`, `rho`, and
`q`, with one column per requested tolerance.  The orange band marks the
high-magnification cut `A_point >= 10`.

The high-magnification zoom uses the focused linear-LD measurements
(`0.01 <= rho < 0.1`, `d/rho < 1`).  It shows `d/rho` against `q`, colours by
the timing ratio, and scales point area by `A_point`.  The guide lines mark the
candidate pocket `d/rho < 0.8`, `q >= 1e-5`; they are deliberately shown as a
measured regime hypothesis, not as a production boundary.

For the focused high-magnification points, the measured summary is:

| target | points | lcbinint win rate | median `R` |
|---:|---:|---:|---:|
| `1e-2` | 196 | 12.2% | 0.389 |
| `1e-3` | 196 | 69.9% | 1.53 |
| `1e-4` | 167 | 91.6% | 4.90 |

Within that zoom, the proposed two-parameter split is visible numerically as
well:

| target | `A_point>=10`, `d/rho<0.8`, `q>=1e-5` | `q<1e-5` |
|---:|---:|---:|
| `1e-2` | 120 points, 19.2% wins, median `R=0.433` | 12 points, 0% wins, median `R=0.035` |
| `1e-3` | 120 points, 75.0% wins, median `R=1.62` | 12 points, 0% wins, median `R=0.245` |
| `1e-4` | 104 points, 99.0% wins, median `R=6.56` | 11 points, 18.2% wins, median `R=0.529` |

These maps are appropriate for explaining where the crossover occurs, but the
boundaries must be validated on an independent parameter holdout before being
used as a hard-coded runtime rule.  The generated files are
`figures/vbm_regime_map_linear_ld.pdf` and
`figures/vbm_regime_map_highmag_linear_ld.pdf`; PNG previews and a compact
machine-readable summary are written next to them.

## Exploratory q-split visualisation

For diagnosis during development, the q-split pair is produced by
`tests/diagnostics/recal2026/plot_vbm_crossover.py`:

1. `figures/vbm_crossover_map_linear_ld.pdf` is the primary map. Its axes are
   the physically interpretable pair `(d/rho, A_point)`, its colour is the
   median speed ratio per geometry, and its two rows separate `q>=1e-5` from
   `q<1e-5`.
2. `figures/vbm_ratio_distributions_linear_ld.pdf` is the supporting one-
   dimensional view. It uses all usable reference epochs and bins only the
   caustic proximity. The dashed `R=1` line makes the crossover visible
   without requiring the reader to infer it from a colour scale.

In the focused high-magnification branch (`A_point>=10`, `0.01<=rho<0.1`,
`d/rho<1`), the `q>=1e-5` rows give `13.0%`, `74.5%`, and `96.8%` lcbinint
wins at targets `1e-2`, `1e-3`, and `1e-4`, respectively. The corresponding
median ratios are `0.43`, `1.58`, and `5.35`. The `q<1e-5` rows stay VBM-led:
`0%`, `0%`, and `18.2%` wins, with median ratios `0.035`, `0.245`, and
`0.529`.

This pair was useful for identifying the low-q tail, but the final paper-facing
presentation below removes q as an explanatory variable.

## Final q-agnostic presentation

For the final paper-facing version, the low-mass-ratio tail is excluded with
the explicit measurement cut `q >= 1e-4`; `q` is not used as a plotted
explanatory variable.  The figures are generated by
`tests/diagnostics/recal2026/plot_vbm_final_maps.py`:

- `figures/vbm_crossover_map_linear_ld_qge1e-4.pdf` shows the two-dimensional
  relation between `d/rho` and `A_point`, coloured by the epoch-median speed
  ratio.
- `figures/vbm_ratio_distributions_linear_ld_qge1e-4.pdf` has two rows: the
  upper row bins high-magnification points by `d/rho`, and the lower row bins
  near-caustic points by `A_point`.

The `A_point` result is not a simple monotonic law by itself.  It is best used
as a high-magnification gate (`A_point >= 10`), after which the LCB advantage
is clear for `1e-3` and especially `1e-4`.  Within the near-caustic sample at
`1e-4`, the median ratio rises from `2.12` in `A_point=1--3` to `7.11` in
`10--30` and `9.64` in `30--100`; the `A_point>=100` bin returns to `5.73`
with only eight points.  This non-monotonic tail is why `A_point` should be
reported as a regime selector, not fitted as a standalone speed law.

## One-dimensional d/rho diagnostic

The direct test of a possible `d/rho`-only speed law is generated by
`tests/diagnostics/recal2026/plot_vbm_d_over_rho_relation.py`. It uses
`x=d/rho`, `y=R=t_VBM/t_lcbinint`, and colours each point by `A_point`; all
records satisfy `q>=1e-4`. All measured d/rho values are placed on one row of
three panels, one panel per relative target; the colour denotes `A_point`.

The result is useful precisely because the colour does not collapse into a
single sequence: d/rho has the clearest monotonic tendency, especially at
`1e-3` and `1e-4`, while A_point contributes a secondary correlated trend in
the focused branch. Therefore a single universal law `R=f(d/rho)` would be
too aggressive. The safe interpretation is a d/rho-based regime split with
an `A_point>=10` high-magnification gate, not a standalone A_point power-law.

The figure is
`figures/vbm_d_over_rho_speed_relation_qge1e-4.pdf`, with the numerical rank
correlations in
`figures/vbm_d_over_rho_speed_relation_summary.md`.
