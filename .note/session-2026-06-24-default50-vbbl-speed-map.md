# Default `source_bins=50` and VBBL Point-Speed Map

Date: 2026-06-24

## Changes

Default finite-source inverse-ray settings were aligned to the current tested
balance point:

- `source_bins = 50`
- `finite_source_reltol = 1e-3`
- `max_source_bins = 400`
- `adaptive_source_bins = 1`

Updated defaults:

- `src/lcbinint/lcbinint.cpp`
- `src/lcbinint/magnification/finite_source_magnifier.hpp`
- `src/lcbinint/model/lens_parameters.hpp`
- `python/lcbinint_pybind.cpp`

## Diagnostic Setup

Point-integral benchmark, not light-curve timing:

```text
script      tests/diagnostics/global_point_sweep.py
run dir     .note/diagnostic_runs/20260624-113347-postfix-point32
cases       40 total = 8 fixed + 32 random
points      160
source_bins 50
max_bins    400
reltol      1e-3
VBBL Tol    1e-3
reference   VBBL Tol=1e-5
repeat      1
```

CSV:

```text
.note/diagnostic_runs/20260624-113347-postfix-point32/points.csv
```

Larger `random=48/64/128` runs were attempted, but VBBL can become extremely
slow for a few LD/high-magnification points and made the wall time too long.
The `random=48, points_per_case=3` check timed out after 180 seconds before
writing rows.  The next broad run should add timeout/skip handling around both
reference selection and point timing, not only around the final VBBL point call.

## Accuracy / False Accepts

After the topology-retry fix, this sweep had:

```text
accepted_bad = 0
```

The earlier low-bin false accepts were removed:

```text
source_bins=32 accepted_bad=0
source_bins=40 accepted_bad=0
source_bins=50 accepted_bad=0
```

Remaining unconverged points are conservative failures, not false accepts.

## Speed Map

`ratio = lcbinint_ms / VBBL_ms`; win rate means `ratio < 1`.

### By Limb Darkening

```text
group   n   win_rate  ratio_med  ratio_geo  lc_ms_med  vbb_ms_med  rel_p95   bad  unconv
ld      64    0.047     54.417     75.373     3.5002      0.0619    2.61e-05   0      1
no_ld   96    0.000     47.876     70.671     3.5295      0.0785    6.77e-05   0      2
```

LD alone is not enough.  lcbinint wins only when VBBL falls onto expensive
finite-source/LD points.

### By Source Radius

```text
rho_bin       n   win_rate  ratio_med  ratio_geo
<1e-4        12    0.000    428.498    244.733
1e-4-3e-4    40    0.000    223.957    109.468
3e-4-1e-3    56    0.036     47.425     52.904
1e-3-3e-3    20    0.000     50.713     66.056
3e-3-1e-2    24    0.000     48.682     58.202
>=1e-2        8    0.125     49.537     33.129
```

Very small sources are VBBL-favorable because VBBL often stays in point/fast
finite-source territory.  Large sources can help lcbinint only near expensive
LD/high-magnification finite-source points.

### By Magnification

```text
mag_bin   n   win_rate  ratio_med  ratio_geo  lc_ms_med  vbb_ms_med
A<5      43    0.000    414.726    281.726     3.3351      0.0078
5-20     77    0.000     50.798     80.990     3.5439      0.0673
20-100   21    0.000     23.656     17.766     3.6379      0.1700
100-300  17    0.176     19.283      9.712    17.6651      0.6960
>=300     2    0.000     15.104     15.001    50.7986      3.5820
```

Low magnification is hopeless for lcbinint.  The first realistic win region is
high magnification, especially with LD.

### By Method

```text
method                  n   win_rate  ratio_med  ratio_geo  lc_ms_med  vbb_ms_med
point_source            38    0.000    390.862    264.708     3.2535      0.0081
hexadecapole           101    0.010     47.505     61.521     3.5487      0.0754
inverse_ray_cartesian   21    0.095     27.059     15.357    17.6651      0.9105
```

The current Python light-curve API still pays several ms/point even for
point-source and hex points.  VBBL is dramatically faster in those regimes.
lcbinint only has a chance when both packages are forced into expensive
finite-source work.

## Where lcbinint Wins

The fastest relative wins are LD/high-magnification finite-source points:

```text
ratio  lc_ms    vbb_ms    A       rho       LD  method
0.191  23.0213  120.3845  128.64  1.0e-02  1   inverse_ray_cartesian
0.497   3.4965    7.0337  274.20  4.7e-04  1   hexadecapole
0.737  22.7327   30.8261  130.76  6.1e-04  1   inverse_ray_cartesian
```

Practical conclusion:

- lcbinint can beat VBBL on expensive LD/high-magnification points.
- lcbinint does not beat VBBL on ordinary low-magnification points.
- no-LD finite-source points are usually still VBBL-favorable.

## Follow-Ups

1. Add timeout/skip handling to `global_point_sweep.py` so `random >= 256` can
   complete and still report the expensive VBBL win region.  This needs to wrap
   `selected_points()` as well as the final point timing because VBBL can stall
   during the high-resolution reference scan.
2. Separate C++ core timing from Python API overhead for point/hex modes.  The
   current point benchmark includes Python model construction and full
   light-curve wrapper overhead.
3. Add method-distribution summaries to notebook examples so users can see when
   a curve is point/hex dominated versus IR dominated.
