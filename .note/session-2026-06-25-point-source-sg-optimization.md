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

## Follow-up: VBM 5.5 Root Ordering

After the first optimization commit, the SG root solver was compared directly against the VBM 5.5 source shipped in the local Python package.

Adopted:

- Select the smallest current starting root before each deflation step.
- Polish only `degree - 1` roots after the deflated quadratic solve, matching VBM's current `cmplx_roots_gen`.

Rejected after measurement:

- Caching binary geometry (`a`, `m1`, `m2`, powers) inside `PointSourceMagnifier`.
  - This adds branch/state overhead and made finite-source/LD timings worse in the current lcbinint design.
- Rewriting residual/Jacobian divisions with reciprocal temporaries.
  - It gave only a tiny speed change and moved one convergence-boundary case in the example, so it was not worth keeping.

Validation after adopting the root ordering:

```text
pytest targeted: 15 passed, 67 deselected
ctest: 1/1 passed

example/compare-vbm representative run
  lcbinint no LD: 0.5671 ms/pt
  lcbinint LD   : 0.7694 ms/pt
  VBM no LD     : 0.0442 ms/pt
  VBM LD        : 0.9359 ms/pt

relative error vs VBM
  no LD max=4.990e-04 p99=2.528e-04 median=1.850e-06 rms=6.480e-05
  LD    max=3.169e-04 p99=1.696e-04 median=9.867e-06 rms=4.139e-05
```

The example timing is noisy, especially for VBM LD, so the important signal is that the method mix and regression accuracy are unchanged while the pure point-source `LightCurve` path remains around `4.4e-4 ms/pt` in the local benchmark.

## Follow-up: Hex Batch and LD Lookup

Implemented after the root-ordering optimization:

- Added `PointSourceMagnifier::binary_mag0_batch`.
  - This is an internal C++ batch path for repeated point-source solves with fixed `(s, q)`.
  - `hexadecapole_binary` now evaluates the 12 off-center hex sample points through this batch path.
  - This is not the full VBM `NewImages` derivative pipeline yet; it is a safe first step that keeps the existing finite-difference hex formula unchanged.
- Optimized limb-darkening table lookup.
  - Table size increased from 5,000 to 20,000.
  - Per-cell lookup changed from linear interpolation to nearest-table lookup.
  - This removes the interpolation arithmetic in the inverse-ray inner loop while keeping the discretization error below the current integration noise.

Validation:

```text
pytest targeted: 15 passed, 67 deselected
ctest: 1/1 passed

example/compare-vbm representative run with LD lookup optimization
  lcbinint no LD: 0.8361 ms/pt  (timing-noisy run)
  lcbinint LD   : 0.6400 ms/pt
  VBM no LD     : 0.0456 ms/pt
  VBM LD        : 0.9277 ms/pt

relative error vs VBM
  no LD max=4.990e-04 p99=2.528e-04 median=1.850e-06 rms=6.480e-05
  LD    max=3.163e-04 p99=1.703e-04 median=9.867e-06 rms=4.138e-05

method mix
  no LD {'point_source': 104, 'hexadecapole': 233, 'inverse_ray_cartesian': 63} converged=355/400
  LD    {'point_source': 104, 'hexadecapole': 234, 'inverse_ray_cartesian': 62} converged=355/400
```

Rejected / deferred:

- 5,000-point nearest LD lookup was faster but moved one LD convergence-boundary point from converged to unconverged in the example. The 20,000-point table kept convergence stable.
- Full derivative-based `NewImages` hex remains future work. The batch path does not remove the 12 extra point-source solves, so it is expected to be mostly neutral in `benchmark_point_hex`.

## Follow-up: Derivative-Based Point-Source Shortcut

Implemented a first `NewImages`-style derivative path:

- Added `PointSourceMagnifier::binary_mag0_with_derivatives`.
  - It solves the center-source images once.
  - It reuses the physical-root selection logic.
  - It computes a VBM-like derivative smoothness indicator from `J1`, `J2`, and `J3`.
- Exposed the derivative indicator through `diagnostic_hexadecapole_binary` for benchmarking.
- Added a finite-source shortcut:
  - If the source is not near a caustic and the derivative smoothness indicator is below the requested tolerance after the same caustic-distance safety factor, return point-source magnification directly.
  - This is disabled when `hex_threshold == 0`, because tests and expert settings use that as a way to force the hex path.

Important interpretation:

- The derivative indicator is much more conservative than the finite-difference hex self-consistency error.
- It is not yet a replacement for the 12-point finite-difference hex magnification.
- It is useful for skipping hex when the point-source approximation is already safely within tolerance.

Validation:

```text
ctest: 1/1 passed
pytest targeted: 15 passed, 67 deselected

example/compare-vbm median timing
  lcbinint no LD: 0.5617 ms/pt
  lcbinint LD   : 0.6799 ms/pt
  VBM no LD     : 0.0666 ms/pt
  VBM LD        : 1.3613 ms/pt

relative error vs VBM
  no LD max=4.990e-04 p99=3.295e-04 median=6.670e-16 rms=8.024e-05
  LD    max=3.635e-04 p99=2.723e-04 median=6.670e-16 rms=6.284e-05

method mix
  no LD {'point_source': 297, 'hexadecapole': 40, 'inverse_ray_cartesian': 63}
  LD    {'point_source': 297, 'hexadecapole': 41, 'inverse_ray_cartesian': 62}
```

The shortcut greatly reduces hex usage in this example (`~233 -> ~40`), but runtime improves only modestly because caustic-distance checks and the inverse-ray points dominate the remaining cost.
