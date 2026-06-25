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

## Follow-up: outlier investigation

Two different outlier classes were checked after the first auto-mode commit.

### Low-magnification polar outliers

Cause: the polar grid used a constant angular spacing based only on `dr`:

```text
dphi ~ dr * grid_ratio
```

The actual tangential cell size is `r * dphi`, so low-magnification images far
from the origin were undersampled in the angular direction.  The polar kernel now
chooses `phi_bins` from the largest relevant seed radius so that the outermost
image has tangential spacing no larger than `grid_ratio * dr`.

Regression case:

```text
s=1.251936920212136
q=0.010229080749960234
u0=-0.045915477051696046
alpha=1.008883116714675
rho=0.03791189085132994
t=0.2526222052684984
linear LD c=0.5
```

With `mode=2`, `source_bins=50`, the relative error is now below `1e-3`.

Tradeoff: polar became slower in low/moderate magnification bins because the
angular grid is no longer under-resolved.  This is expected; mode 4 keeps using
Cartesian below the high-magnification threshold.

For auto mode only, the polar branch now relaxes `grid_ratio` to at least `12`.
This keeps explicit `mode=2` controlled by the user-provided grid while making
the high-magnification automatic path cheaper.  Spot checks:

```text
A~3900 tiny source: mode=2 grid_ratio=4  rel~2.5e-5, 230 ms
A~3900 tiny source: mode=4 grid_ratio=12 rel~5.1e-4,  90 ms

A~256 resonant:     mode=2 grid_ratio=4  rel~2.7e-5, 17.6 ms
A~256 resonant:     mode=4 grid_ratio=12 rel~7.5e-5,  8.0 ms

A~131 LD:           mode=2 grid_ratio=4  rel~1.5e-4, 10.4 ms
A~131 LD:           mode=4 grid_ratio=12 rel~1.4e-4,  5.2 ms
```

Light sweep after this change:

```text
mag 100..300:   auto/cart median ~0.39, auto win 100%, auto bad 0
mag 300..1000:  auto/cart median ~0.40, auto win 100%, auto bad 0
mag >=1000:     auto/cart median ~0.41, auto win 100%, auto bad 0
rho <3e-4:      auto/cart median ~0.92, Cartesian p90 rel ~1.7e-1, auto p90 rel ~6e-4
```

### Tiny-source high-magnification Cartesian outlier

Representative case:

```text
s=1.0
q=1e-3
u0=-1e-4
alpha=0.5
rho=1e-4
t=0.0006
```

At `source_bins=50`, forced Cartesian gives a relative error around `0.19`, while
polar gives `~2.5e-5`.  The diagnostics show `O(100)` seeds and enormous
gap-repair/jump counts; the failure is not a simple missing seed.  The Cartesian
legacy scanline kernel adds areas seed-by-seed and then tries to remove overlaps,
which is fragile for annular/near-critical high-magnification images around the
lens.  A quick global cell-coverage patch was tested and rejected because it
interacted badly with the seed-local scanline traversal and became too slow.

Current robust behavior: `mode=4` routes this case to polar and matches VBM below
`1e-3`.  A true forced-Cartesian fix should be a larger redesign: either a common
Cartesian lattice with global flood-fill/interval union, or replacing this
high-magnification topology with the polar kernel.  Do not patch this with
source-radius-specific seed suppression; that only hides the seed explosion and
does not fix the area-union problem.

Additional regression tests:

```text
test_lcbinint_auto_inverse_ray_avoids_tiny_source_cartesian_aliasing
test_lcbinint_polar_uses_radius_aware_angular_resolution_for_low_magnification
```

## Follow-up: separate polar resolution knobs

`source_bins=50` does not have exactly the same meaning in Cartesian and polar
inverse-ray modes.

Current interpretation:

```text
Cartesian: source_bins controls the Cartesian grid scale.
Polar:     source_bins controls radial bins, while angular bins are derived from
           dr and grid_ratio.
```

This made tuning ambiguous, so the public/internal option set now has:

```text
polar_source_bins = 0   # 0 means use source_bins
polar_grid_ratio  = 0   # <=0 means use grid_ratio
```

The default therefore preserves the previous behavior.  Explicit polar tuning can
now be done without changing the Cartesian grid resolution.

Small spot sweep, forced polar, `source_bins=50`, VBM reference tolerance `1e-5`:

```text
combo            max_all   p90_med   ms_med
default          1.8e-4    1.3e-4    0.0043
polar_bins=40    2.2e-4    1.5e-4    0.0051
polar_bins=50    1.8e-4    1.3e-4    0.0053
polar_bins=50,
  grid_ratio=5   2.6e-4    9.8e-5    0.0043
polar_bins=64    2.6e-4    1.2e-4    0.0043
```

Medium diagnostic sweep:

```text
PYTHONPATH=build python tests/diagnostics/polar_cartesian_mode_sweep.py \
  --random 20 --points-per-case 4 --repeat 1 --source-bins 50

points=112, polar_source_bins=0, polar_grid_ratio=0
mag >=1000:    auto/cart median 0.342, auto win 100%, auto_bad 0
mag 100..300:  auto/cart median 0.375, auto win 100%, auto_bad 0
all bins:      polar_bad 0, auto_bad 0
```

The same sweep with `polar_source_bins=40, polar_grid_ratio=4`:

```text
mag >=1000:    auto/cart median 0.275, auto win 100%, auto_bad 0
mag 100..300:  auto/cart median 0.330, auto win 100%, auto_bad 0
all bins:      polar_bad 0, auto_bad 1
```

Conclusion: `polar_source_bins=40` can be faster in the intended
high-magnification auto branch, but it introduced one `>3e-3` auto outlier in
this medium sweep.  Do not make it the default yet.  The current recommended
default remains `source_bins=50` with `polar_source_bins=0`.

Verification:

```text
ctest --test-dir build --output-on-failure
PYTHONPATH=build python -m pytest \
  tests/regression/test_vbm_consistency.py \
  tests/regression/test_adaptive_precision_redesign.py \
  tests/regression/test_component_union_validation.py -q

96 passed
```

## Public API direction

The Python public API should not expose numeric finite-source modes.  This
option does not choose between point-source/hexadecapole/inverse-ray/contour
algorithms; it only chooses the grid used after the calculation falls through to
inverse-ray integration.  Users choose it by name:

```python
lcbinint.Options(inverse_ray_grid="auto")       # default, recommended
lcbinint.Options(inverse_ray_grid="cartesian")  # expert/debug
lcbinint.Options(inverse_ray_grid="polar")      # expert/debug
```

`auto` is now the default for Python `Options()`, `LightCurve()` without explicit
options, and the C default options.  The numeric mode remains only as an internal
C/C++ representation:

```text
1 = cartesian
2 = polar
4 = auto
```

Python exposes `_mode` as a read-only diagnostic field but does not accept
`Options(mode=...)` or the old `Options(finite_source_method=...)`.
Experimental spine mode is not part of the public API.

The string grid type is resolved once when `Options` is constructed, so it does
not add any per-point overhead or touch the Cartesian overlap bookkeeping.
