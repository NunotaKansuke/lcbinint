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

params = LensParams()
params.t0 = 0
params.tE = 20
params.umin = 0.1
params.q = 0.2
params.sep = 1.1
params.theta = 0.0
options = Options()
options.center_of_mass = 1
model = LensModel(params, options)
amp = model.magnifications([2450000.1, 2450001.1])
```

Current binding note: `LensParams()` and `Options()` are default-constructed
objects with mutable fields rather than keyword-style constructors. Keyword
construction can be added after the stable parameter set is clear.

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
| `FiniteSourceMagnifier` | Later finite-source/inverse-ray implementation. |

This should keep global mutable state out of the point-source path. When
finite-source cache state becomes necessary, add an explicit context object
rather than restoring legacy globals.

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
| `zroots2`, `zroots2l`, `zroots2q` + `laguer*` | complex polynomial roots for image positions and caustics | Replace with the Skowron & Gould complex polynomial solver used in microlensing codes such as VBBinaryLensing. The published SG code is Apache-2.0/LGPL; keep attribution and isolate it behind `PolynomialRootSolver`. |
| `zbrent4`, `zbrent5` | scalar root finding for orbital anomaly and ray boundary | GSL Brent root solver or small clean-room Brent implementation. |
| `qromb`, `qromb1`, `qromb2`, `trapzd*`, `polint` | finite-source integration | GSL integration (`gsl_integration_qag` / `qags`) or clean-room adaptive Gauss-Kronrod. |
| `piksrt`, `indexx` | sorting | C standard `qsort` or local small sort. |
| `dvector`, `dmatrix`, `nrutil` | offset-indexed allocation | plain `calloc`/`free` with zero-based arrays. |
| `ran1`, `gasdev` | artificial light curves and ray jitter | remove from first magnification core; later use a permissive RNG implementation. |
| `powell3`, `linmin2`, `brent2`, `mnbrak2`, `djacobi` | fitting/MCMC support | out of initial scope; later use GSL, scipy-side fitting, or clean-room code. |

GSL is used as a general dependency following `../genulens`, but the polynomial
root solver should not use GSL's companion-matrix solver. Use the
Skowron-Gould algorithm for the microlensing root path.

## Milestones

1. Create a standalone C++ library skeleton around `lcbi_magnification`.
2. Add internal C++ value types and route the C ABI through `LensModel`.
3. Add a Skowron-Gould-backed `PolynomialRootSolver`.
4. Port point-source binary and triple calculations into the C++ structure.
5. Add VBM/VBBinaryLensing consistency tests for the point-source path.
6. Add regression tests comparing the extracted library against selected legacy
   executable outputs.
7. Add Python bindings.
8. Add finite-source modes after the point-source path is stable.

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

Use VBBinaryLensing/VBMicrolensing as an independent reference for the new
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

This low-level API is matched against `VBBinaryLensing().BinaryMag0(...)`.
Keep it as a direct numerical validation hook even after the higher-level
`LensModel` API exists.

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
