# Tests After Point / Hex Kernel Optimization

Date: 2026-06-24

## Scope

After optimizing `PointSourceMagnifier::binary_mag0`, the next check was whether
the speedup propagates beyond the pure C++ microbenchmark.

The optimized point-source kernel is:

- warm point-source: `~1.06-1.08 us`
- cold point-source: `~1.60-1.85 us`
- hexadecapole: `~16-18 us`

VBBL public `BinaryMag0` on the same machine is `~1.01-1.03 us`.

## Point Integration Benchmark

Command:

```text
LCBININT_EXTENSION=build/lcbinint...so \
python tests/diagnostics/point_integration_benchmark.py \
  --source-bins 50 --reltols 1e-3 --max-bins 400 \
  --points-per-case 3 --repeat 2 --random 12 --seed 20260624
```

Result summary:

```text
cases=20 points=60
source_bins=50 reltol=1e-3
lc_ms_med=3.1965
lc/vbb_med=49.333
rel_med=2.82e-08
rel_p90=4.23e-05
unconverged=2
accepted_bad=0
```

Interpretation:

- Accuracy remains fine; no accepted bad points.
- This benchmark still constructs `LensModel(...).light_curve([time])` per
  point, so Python/API/dispatcher fixed cost dominates point and hex cases.
- The pure point/hex kernel improvement is mostly hidden by this benchmark's
  per-point wrapper overhead.

Fastest relative wins:

```text
ratio  case                        method                  LD   A       rho
0.18   planetary_large_source_ld    inverse_ray_cartesian   1    128.64  1e-2
0.43   random_004                  hexadecapole            1    274.20  4.7e-4
0.74   random_007                  inverse_ray_cartesian   1    130.76  6.1e-4
```

## Global Point Sweep

Command:

```text
python tests/diagnostics/global_point_sweep.py \
  --extension build/lcbinint...so \
  --random 12 --points-per-case 3 \
  --source-bins 50 --reltol 1e-3 --repeat 1
```

CSV:

```text
.note/diagnostic_runs/20260624-after-point-opt-global12.csv
```

Method summary:

```text
method                  n   win_rate  ratio_med  ratio_geo  lc_ms_med  vbb_ms_med  accepted_bad  unconverged
point_source            18    0.000    301.608    242.388     3.1309      0.0102       0             0
hexadecapole            33    0.030     40.192     43.519     3.2825      0.0855       0             0
inverse_ray_cartesian    9    0.222     16.267      8.811    19.7318      2.0337       0             2
```

Interpretation:

- The point/hex kernel is fast, but single-point Python API timing is still
  dominated by fixed overhead.
- The optimized kernel helps most when the same C++ objects evaluate nearby
  source positions, e.g. hex, finite-source probes, and light curves.

## LD Light-Curve Sweep

The first run accidentally imported the old site-packages `lcbinint`; discard
that result.  The valid run forced the build extension via `sys.path`.

Command:

```text
python -c "remove _lcbinint_editable hook; sys.path.insert(0, 'build'); runpy..."
tests/diagnostics/ld_source_bins_rho_sweep.py --source-bins 50 --times 61 --vbbl-tol 1e-3
```

Valid result:

```text
source_bins=50, LD c=0.5, times=61
cases=42
lc/vbb_med=1.958
lc/vbb_geo=1.853
lc_ms_med=0.5144
vbb_ms_med=0.2367
maxrel_med=3.257e-04
maxrel_worst=2.297e-03
```

Compared with the stale site-packages run:

```text
old/stale lc_ms_med = 0.5805 ms/pt
new/build lc_ms_med = 0.5144 ms/pt
speedup ~= 11%
```

Best light-curve regimes after the point optimization:

```text
geometry              rho      lc/vbb  lc_ms   vbb_ms  maxmag
planetary_close       3e-3     0.27    0.4885  1.7899  178.24
planetary_resonant    1e-2     0.37    0.9236  2.4899  128.64
planetary_close       1e-2     0.38    1.1551  3.0252  124.24
planetary_resonant    3e-2     0.43    1.8127  4.2166   68.01
planetary_wide        3e-2     0.47    0.8702  1.8535   60.55
```

Worst regimes remain wide/equal-mass and low-magnification cases where VBBL
stays on very cheap paths.

## Conclusion

The point/hex optimization is real:

- Pure C++ point/hex is now close to VBBL.
- LD light-curve timing improved by about `10%` on the broad sweep.
- Single-point Python benchmarks still hide the kernel win behind object
  construction and dispatcher overhead.

Next speed work should target C++ light-curve reuse and method dispatch
overhead, not the point-source kernel itself.
