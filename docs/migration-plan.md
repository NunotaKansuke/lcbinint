# Migration Plan

## Direction

Do not wrap the legacy `lcbinint` executable for the public package. Extract the
magnification path into a C++ library with a small C-compatible ABI, then bind
that library from Python.

The first useful target is point-source magnification for both binary and triple
lenses. That path already covers the most important API shape and exercises the
hardest immediate replacement: polynomial root solving.

## Public Boundary

The intended C-compatible boundary is declared in `include/lcbinint/lcbinint.h`.
The implementation can use C++ internally.

The corresponding Python API should look roughly like:

```python
from lcbinint import LensModel, LensParams, Options

params = LensParams(t0=0, tE=20, umin=0.1, q=0.2, sep=1.1, theta=0.0)
options = Options(center_of_mass=1)
model = LensModel(params, options)
amp = model.magnifications([2450000.1, 2450001.1])
curve = model.light_curve([2450000.1, 2450001.1])
```

The parameter objects also keep mutable fields for interactive use and fitting
code that wants to update values in place.

`light_curve()` returns a small result object with the sampled times,
magnifications, point-source/finite-source components, source-plane trajectory
coordinates, and image counts.

Annual parallax should follow the `../jacscanomaly` projector model rather than
the old Kepler approximation: use a packaged Horizons Earth ephemeris table,
interpolate Earth position/velocity, project to sky north/east, and subtract the
reference-frame position and velocity at `tfix`.

Triple-lens support should not be a separate model class. It should be enabled
by `q2 > 0` with `sep2` and `ang` set.

## C++ Design Direction

The goal is not to mechanically rename legacy C files to `.cpp`. The extracted
core should use C++ types to make the numerical code easier to test, extend, and
publish.

Keep a small C-compatible ABI for package stability, but route it into C++
objects internally:

```text
lcbi_magnification()
  -> lcbinint::LensModel
  -> lcbinint::Trajectory
  -> lcbinint::LensSystem
  -> lcbinint::PolynomialRootSolver
```

Proposed internal types:

| Type | Responsibility |
| --- | --- |
| `LensParameters` | Validated physical parameters converted from `lcbi_params`. |
| `ComputationOptions` | Numerical and mode options converted from `lcbi_options`. |
| `SourcePosition` | Source coordinate at one time, including trajectory effects. |
| `LensSystem` | Binary/triple lens geometry and mass fractions. |
| `PolynomialRootSolver` | Skowron-Gould complex polynomial roots, isolated from lens logic. |
| `PointSourceMagnifier` | Binary/triple point-source image finding and magnification. |
| `LensModel` | User-level orchestration for one model evaluation. |
| `FiniteSourceMagnifier` | Finite-source strategy selection and inverse-ray implementation. |

This should keep global mutable state out of the point-source path. When
finite-source cache state becomes necessary, add an explicit context object
rather than restoring legacy globals.

Finite-source model selection should not expose the legacy `FINITE=1..6`
numbers as the public API. The user-facing controls should be:

```python
options = Options(tolerance=1e-3, relative_tolerance=0.0)
```

The C++ core should then choose an internal strategy based on local accuracy
tests. Cost estimates are diagnostics for refinement planning, not a reason to
silently downgrade to a less accurate method:

```text
point_source
inverse_ray_cartesian
inverse_ray_polar
```

Polar inverse-ray should remain available internally because it can be better
for high-magnification cases. It should be selected by the strategy layer, not
by forcing users to know which finite-source mode is best. An expert override
is still available through `Options(inverse_ray_method=...)` for users who want
to force cartesian or polar inverse-ray.

For expert compatibility, `FiniteSourceMode.LEGACY` is available with
old-style finite-source knobs:

```python
options = Options(
    finite_source_mode=FiniteSourceMode.LEGACY,
    legacy_finite_mode=4,
    legacy_kinji=9.0,
    legacy_hex=2.0,
)
```

This preserves the KINJI/HEX decision structure using a sampled binary-caustic
distance. It is a compatibility path, not the recommended default interface.

The detailed finite-source plan and current limitations are tracked in
`.note/finite-source-strategy.md`. In particular, microJAX should be treated as
a useful reference for accuracy-control and method-selection logic, not as an
implementation to port directly.
`../microlux` is the most useful local reference for the Bozza-style
quadrupole/ghost-image/planetary-caustic fast-path acceptance test.

## Initial Extraction Target

Start from this legacy call chain:

```text
finiteAt()
  -> amp_point2()       binary point-source
  -> amp_point3()       triple point-source
  -> amp_point3q()      triple point-source fallback
  -> imageposition()
  -> imageposition3()
  -> imageposition3l()
  -> imageposition3q()
  -> zroots2 / zroots2l / zroots2q
```

For the first library milestone, exclude fitting, data reading, drawing, MCMC,
and artificial light-curve generation.

## Numerical Recipes Replacement Inventory

Replace before public release:

| Legacy routine | Current role | Replacement direction |
| --- | --- | --- |
| `zroots2`, `zroots2l`, `zroots2q` + `laguer*` | complex polynomial roots for image positions and caustics | Replace with the Skowron & Gould complex polynomial solver used in microlensing codes such as VBMicrolensing. The published SG code is Apache-2.0/LGPL; keep attribution and isolate it behind `PolynomialRootSolver`. |
| `zbrent4`, `zbrent5` | scalar root finding for orbital anomaly and ray boundary | GSL Brent root solver or small clean-room Brent implementation. |
| `qromb`, `qromb1`, `qromb2`, `trapzd*`, `polint` | legacy finite-source integration | Do not port as a public path. Prefer inverse-ray plus quadrupole/point-source safety tests. If scalar integration is later needed, use GSL or clean-room code. |
| `piksrt`, `indexx` | sorting | C standard `qsort` or local small sort. |
| `dvector`, `dmatrix`, `nrutil` | offset-indexed allocation | plain `calloc`/`free` with zero-based arrays. |
| `ran1`, `gasdev` | artificial light curves and ray jitter | remove from first magnification core; later use a permissive RNG implementation. |
| `powell3`, `linmin2`, `brent2`, `mnbrak2`, `djacobi` | fitting/MCMC support | out of initial scope; later use GSL, scipy-side fitting, or clean-room code. |

GSL is used as a general dependency following `../genulens`, but the polynomial
root solver should not use GSL's companion-matrix solver. Use the
Skowron-Gould algorithm for the microlensing root path. On this machine, CMake
also checks `/rogue1_8/nunota/local/gsl` as a fallback prefix so the build does
not depend on manually exporting `GSL_ROOT`.

## Milestones

1. Create a standalone C++ library skeleton around `lcbi_magnification`.
2. Add internal C++ value types and route the C ABI through `LensModel`.
3. Add a Skowron-Gould-backed `PolynomialRootSolver`.
4. Port point-source binary and triple calculations into the C++ structure.
5. Add VBM/VBMicrolensing consistency tests for the point-source path.
6. Add regression tests comparing the extracted library against selected legacy
   executable outputs.
7. Add Python bindings.
8. Add finite-source inverse-ray modes after the point-source path is stable.

Finite-source milestone detail:

1. Keep the public finite-source controls to `tolerance` and
   `relative_tolerance`. Done.
2. Implement the quadrupole safety test as the cheap first decision. Initial
   C++ port of the `../microlux` `Quadrupole_test()` logic is implemented:
   quadrupole correction, cusp correction, ghost-image test, and
   planetary-caustic test.
3. Implement cartesian inverse-ray area calculation. Initial binary image-plane
   grid estimator is implemented.
4. Implement polar inverse-ray area calculation and select it for high
   magnification or source-centered cases. Initial binary estimator and strategy
   hook are implemented.
5. Add VBM comparisons against `BinaryMag2`/`BinaryMagDark`. Initial
   `BinaryMag2` comparisons and linear `BinaryMagDark` comparisons are present.
6. Add explicit non-convergence handling. Done for the C ABI/Python scalar path:
   failed finite-source refinement returns `LCBI_NUMERICAL_ERROR` and Python
   raises `RuntimeError("numerical error")`.
7. Add quadratic/square-root limb-darkening coverage for the local
   Gamma/Lambda implementation. The C++ path supports both `limb_darkening_c`
   and `limb_darkening_d`; VBMicrolensing's Python `BinaryMagDark` binding
   currently provides a simple linear-coefficient comparison point.

## Design Notes

- Keep global mutable state out of the new C++ core. The legacy code uses globals
  for caustic caches and ray grids; the public library should move those into an
  explicit context only when finite-source modes need them.
- Keep the Python model stable even while C internals change.
- Preserve binary and triple behavior in one parameter struct. Separate classes
  would make later fitting code more awkward.
- Use C++ classes where they clarify ownership, validation, or numerical
  boundaries. Avoid creating class wrappers that only mirror legacy functions.

## VBM Consistency Tests

Use VBMicrolensing/VBMicrolensing as an independent reference for the new
Skowron-Gould-based path. The initial binary point-source tests live in
`tests/regression/test_vbm_consistency.py`.

The low-level validation API should expose direct point-source evaluation:

```python
lcbinint.binary_mag0(separation, mass_ratio, y1, y2)
```

This is intentionally separate from the higher-level trajectory API:

```python
model.magnification(times)
```

Direct `binary_mag0()` tests make it easier to validate the root solver and
image finder before finite-source and trajectory effects are involved.

The initial binary point-source implementation is now available as:

```python
lcbinint.binary_mag0(separation, mass_ratio, y1, y2)
```

This low-level API is matched against `VBMicrolensing().BinaryMag0(...)`.
Keep it as a direct numerical validation hook even after the higher-level
`LensModel` API exists.

The higher-level `LensModel` path preserves the legacy wide-binary coordinate
choice for `sep > 1` when `Options(center_of_mass=0)`:

```text
xc0 = m2 * sep - m2 / sep
source_cm.x = source.x - xc0
```

This keeps the useful center-of-caustic coordinate behavior from `finiteAt()`
without changing the low-level `binary_mag0()` API, which remains VBM/CM-like.

The root solver itself is compared directly against VBMicrolensing in
`tests/regression/test_solver_vbm_consistency.py` through:

```python
lcbinint.polynomial_roots(coefficients)
VBMicrolensing().cmplx_roots_gen([[re, im], ...])
```

## Polynomial Root Solver Boundary

`src/lcbinint/math/polynomial_roots.hpp` is the dedicated boundary for complex
polynomial roots. Coefficients are constant-first:

```text
c[0] + c[1] z + ... + c[n] z^n
```

The class handles degree one and two analytically. Degree three and higher are
delegated to the official Skowron-Gould C++ solver vendored in
`third_party/skowron_gould`.

The upstream SG distribution is dual-licensed under Apache-2.0 or LGPL. This
repo uses the Apache-2.0 option and keeps `LICENSE`, `NOTICE`, and a local
README alongside the vendored files. Keep future local modifications isolated
there or in a separate adapter so attribution remains clear.
