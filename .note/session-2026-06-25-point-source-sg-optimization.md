# 2026-06-25 point-source / SG solver optimization branch

Branch: `optimize-newimages-point-source`

Base saved on `master`:

- `6729210 Stabilize Python API and VBM comparison baseline`

## Goal

The slow part shared by point-source, hexadecapole, and inverse-ray calls is the binary point-source solve. The target of this branch is to reduce that core overhead without changing finite-source accuracy policy or adding heavy fallbacks.

## Changes

- Moved the Skowron-Gould `complex` constructors/operators/functions into `SkowronGould.h` as inline functions.
- Removed the corresponding out-of-line definitions from `SkowronGould.cpp`.
- Corrected `expcmplx(z)` to the standard complex exponential `exp(re) * (cos(im), sin(im))`.
  - The old implementation used `atan2(im, re)`. That is wrong for the random-jump direction used when Laguerre/Newton struggles.
  - Regression tests still pass after the correction.
- Added `PointSourceMagnifier::binary_mag0_cached`.
  - `binary_mag0` remains the default non-cache solve.
  - Finite-source internal sampling uses non-cache solves because cached root polishing is slower for local offset grids.
  - Pure point-source light-curve paths use the cached solve because neighboring times are coherent.
- Kept `cmplx_roots_gen` as the production solver.
  - A direct `cmplx_roots_5` swap was tested earlier and broke finite-source method selection / accuracy, so it is not used.

## Latest Measurements

Command path:

- Targeted Python VBM/solver regression tests.
- `example/compare-vbm/quickstart_compare_vbm.py`.
- Low-level point-source loop and pure `LightCurve` point-source loop.
- `ctest --test-dir build --output-on-failure`.

Results:

```text
pytest targeted: 15 passed, 67 deselected
ctest: 1/1 passed

example/compare-vbm ms/point
  lcbinint no LD: 0.5711
  lcbinint LD   : 0.6695
  VBM no LD     : 0.0453
  VBM LD        : 0.9314

relative error vs VBM
  no LD max=4.990e-04 p99=2.528e-04 median=1.850e-06 rms=6.480e-05
  LD    max=3.169e-04 p99=1.696e-04 median=9.867e-06 rms=4.139e-05

method mix
  no LD {'point_source': 104, 'hexadecapole': 233, 'inverse_ray_cartesian': 63}
  LD    {'point_source': 104, 'hexadecapole': 234, 'inverse_ray_cartesian': 62}

point-only binary_mag0 loop
  lcbinint: 0.000902 ms/pt
  VBM:      0.001203 ms/pt
  ratio:    0.75

point-only LightCurve path
  lcbinint: 0.000472 ms/pt
```

## Interpretation

- The main win is removing function-call overhead inside the SG complex arithmetic.
- The finite-source example now beats VBM for LD in this representative case.
- No-LD still loses to VBM because VBM often avoids inverse-ray integration entirely with contour/point/hex paths; this is expected unless the finite-source integration method changes.
- Root cache is only useful for pure point-source light curves. In finite-source / hex sampling it worsens runtime because neighboring samples are not ordered like a physical trajectory and polishing old roots can cost more than starting fresh.

## Remaining Work

- If more point-source speed is needed, the next real target is a dedicated 5th-order binary-lens root pipeline equivalent to VBM's `NewImages`, not a blind `cmplx_roots_5` call.
- Any such port should preserve current method-selection accuracy tests and should be measured through finite-source light curves, not only point-source loops.
