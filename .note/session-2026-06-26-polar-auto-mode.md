# 2026-06-26 polar/cartesian auto mode

## Summary

Polar inverse-ray mode is now reliable enough to use directly, but the current
benchmark sweep shows it should not replace Cartesian globally.  It is useful
mainly at high magnification, where the image geometry is elongated and the
polar scanline grid follows the image structure better.

The new public finite-source mode is:

- `mode=1`: Cartesian inverse-ray, default
- `mode=2`: polar inverse-ray
- `mode=4`: automatic Cartesian/polar inverse-ray

`mode=4` currently chooses polar only when the point-source magnification at the
source center is at least 100.  Otherwise it uses Cartesian.  This keeps the
known low-magnification polar outliers out of the automatic path.

## Diagnostic Setup

Added:

```text
tests/diagnostics/polar_cartesian_mode_sweep.py
```

The script forces inverse-ray evaluation by disabling point-source and
hexadecapole exits, then compares:

- `mode=1` Cartesian
- `mode=2` polar
- `mode=4` auto

against `VBMicrolensing.BinaryLightCurve` point references.  It reports timing
and relative error binned by `rho`, magnification, `q`, `s`, and limb darkening.

Important detail: repeated evaluation of the exact same point hits the
`FiniteSourceMagnifier` cache, so the default repeat count is 1.  Use more
random cases instead of repeated identical timing loops when evaluating
mode-level performance.

## Current Sweep Result

Light and medium sweeps show the same trend:

- polar wins most often for `A >= 100`
- polar is usually not worth using for `A < 100`
- low-magnification polar can have accuracy outliers at the `1e-3` to `1e-2`
  level for some wide/large-source cases
- `mode=4` avoids those low-magnification polar outliers while keeping most of
  the high-magnification speed/stability gain

Representative medium sweep:

```text
points=140, source_bins=50

mag 100..300:   auto/cart median ~0.67, auto win ~86%, auto bad 0
mag 300..1000:  auto/cart median ~0.71, auto win ~78%, auto bad 0
mag >=1000:     auto/cart median ~0.66, auto win ~90%, auto bad 0
mag 30..100:    auto/cart median ~1.00, auto win ~32%, auto bad 0
mag 10..30:     auto/cart median ~1.00, auto win ~55%, auto bad 0
```

The high-magnification threshold of 100 is deliberately conservative.  It is not
yet evidence that `mode=4` should become the default, but it is a useful public
option for high-magnification curves.

Commit-time light sweep:

```text
PYTHONPATH=build python tests/diagnostics/polar_cartesian_mode_sweep.py \
  --random 10 --points-per-case 4 --source-bins 50 --repeat 1 --top 6

points=72, source_bins=50

mag 100..300:   auto/cart median ~0.50, auto win 75%,  auto bad 0
mag 300..1000:  auto/cart median ~0.58, auto win 67%,  auto bad 0
mag >=1000:     auto/cart median ~0.67, auto win 100%, auto bad 0
mag 30..100:    auto/cart median ~1.00, auto win 36%,  auto bad 0
mag 10..30:     auto/cart median ~0.99, auto win 77%,  auto bad 0
rho <3e-4:      Cartesian p90 rel ~1.7e-1, auto p90 rel ~1.8e-4
```

The last line is important: for tiny-source, high-magnification points, the
Cartesian grid can still show large aliasing error, while polar/auto remains
stable in this sweep.

## Tests

Added a regression test that checks the automatic method choice:

- high-magnification forced inverse-ray point uses `inverse_ray_polar`
- lower-magnification forced inverse-ray point uses `inverse_ray_cartesian`

Regression commands used:

```text
cmake --build build -j
PYTHONPATH=build python -m pytest \
  tests/regression/test_vbm_consistency.py::test_lcbinint_auto_inverse_ray_uses_polar_only_for_high_magnification \
  tests/regression/test_vbm_consistency.py::test_lcbinint_polar_high_magnification_curve_matches_vbm_without_cartesian_fallback -q
ctest --test-dir build --output-on-failure
PYTHONPATH=build python -m pytest \
  tests/regression/test_vbm_consistency.py \
  tests/regression/test_adaptive_precision_redesign.py \
  tests/regression/test_component_union_validation.py -q
```

Observed result:

```text
94 passed
```

## Next Checks

Before considering `mode=4` as a default, run a larger random sweep and inspect:

- low-magnification auto outliers
- cases near the current threshold `A ~ 100`
- limb-darkened high-magnification cases
- whether a threshold depending on `rho` or caustic distance improves the
  speed/accuracy tradeoff without adding heavy diagnostics
