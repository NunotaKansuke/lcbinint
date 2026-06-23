# Local adaptive finite-source defaults

Date: 2026-06-23

## Summary

The finite-source inverse-ray default has been moved to the local adaptive
Cartesian integrator:

- `source_bins = 64`
- `adaptive_source_bins = 1`
- `finite_source_reltol = 1e-3`
- `max_source_bins = 400`

This is now the intended public default.  The old fixed `source_bins=50`
behavior is still available by passing `adaptive_source_bins=0`.

## Current algorithm

The first pass is still a normal Cartesian inverse-ray area calculation on the
base grid.  When the local error estimate is too large, the code does not rerun
the whole source disk at a larger `source_bins`.  Instead it records cells near
image/source boundaries during the coarse pass and refines only the cells with
the largest estimated contribution to the finite-source integral error.

For each candidate cell, the local estimator compares the center value against
corner values and includes a boundary-crossing term.  Refinement splits one
cell into four children, updates the area contribution, and keeps the
limb-darkened flux normalized by each child cell's area fraction.  This avoids
mixing different grid sizes with the wrong LD normalization.

The public convergence flag is set from the final local error estimate:

```text
error_estimate <= 0.999 * (finite_source_tol + reltol * max(|A|, 1))
```

The current safety factor depends on the initial grid and source radius:

```text
source_bins < 32       -> 8.0
rho >= 8e-3           -> 2.5
rho >= 3e-3           -> 1.8
otherwise             -> 1.25
```

`source_bins < 32` remains marked unconverged in adaptive mode.  Those grids can
be useful for diagnostics, but they are not reliable enough as a default.

## Benchmark setup

Diagnostic script:

```text
tests/diagnostics/point_integration_benchmark.py
```

The sweep used the fixed diagnostic cases plus 48 random cases, five selected
time points per case, with three timing repeats.  Reference magnifications were
computed with VBBL at `reference_tol=1e-5`; timing comparisons used VBBL
`tol=1e-3`.

Headline result after the current tuning:

```text
source_bins  points  lc_ms_med  lc_ms_geo  lc/vbb_med  lc/vbb_geo  rel_p90   maxlev  unconv  accepted_bad
20             840     3.4418    3.9394      52.112      57.260   3.39e-05     4      93       0
32             840     3.4241    3.9857      51.460      57.933   3.48e-05     4      40       0
40             840     3.4163    4.0288      52.061      58.559   2.20e-05     4      36       0
50             840     3.4270    4.0872      51.562      59.409   1.95e-05     3      30       0
64             840     3.4235    4.1718      51.644      60.638   1.35e-05     3      21       0
80             840     3.4150    4.2505      51.988      61.782   1.17e-05     3      16       0
```

For the selected default:

```text
source_bins=64, reltol=1e-3, max_source_bins=400
unconverged: 7 / 280 points
accepted_bad: 0 / 280 points
90th percentile relative error: 1.48e-05
max local refinement level: 3
```

`source_bins=80` is slightly more robust, but it did not improve the median
time in this single-point benchmark and costs more grid work in larger light
curve runs.  `source_bins=64` is the current balance point.

## Interpretation

The benchmark is intentionally broad, so VBBL dominates easy low-magnification
points.  lcbinint is still much slower there because inverse ray has a fixed
area-integration overhead.  The useful regime is limb-darkened and/or high
magnification finite-source points, where VBBL's contour calculation becomes
expensive and local adaptive inverse-ray can be competitive or faster.

The convergence policy is deliberately conservative in one direction: if the
local estimator cannot certify the requested tolerance, it returns
`finite_source_converged=false` instead of silently accepting the point.  The
latest tuning removed the known `accepted_bad` cases in the diagnostic sweep
without forcing all difficult points to pass.

## Regression coverage

`tests/regression/test_vbm_consistency.py` now includes two previously bad
single-point cases.  The test accepts an unconverged result, but if the point is
reported converged, the measured VBBL-reference error must be within the
requested tolerance envelope.

The new benchmark script is not a pass/fail unit test by default.  It is meant
for tuning the empirical error estimator and comparing source-bin/reltol
choices over a broad sample.

## Remaining caveats

- Cartesian phase/aliasing is reduced by local cell refinement, not eliminated
  analytically.
- The error estimator is empirical.  More random/corner-case sweeps should be
  used before tightening the default below `reltol=1e-3`.
- Polar mode is still user-selectable but is not part of this default path.
- Very easy no-LD points remain a VBBL-favorable regime.
