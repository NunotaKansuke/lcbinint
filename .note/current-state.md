# Current State and Direction

## What We Are Building

The goal is to make the existing `lcbinint` microlensing calculation usable as a
publishable Python package backed by a C++ core.

Important direction:

- Do not build a Python wrapper that shells out to the legacy `lcbinint`
  executable.
- Extract the magnification calculation into a proper C++ library.
- Keep a small C-compatible ABI in front of the C++ implementation.
- Expose the library through a clean Python API.
- Replace Numerical Recipes-derived routines before public release.

## Existing Legacy Code

The reference implementation currently lives outside this repo:

```text
/moao38_7/nunota/binfit/integral/lcbinint.c
```

The key legacy path for magnification is:

```text
finiteAt()
  -> amp_point2()       binary point-source magnification
  -> amp_point3()       triple point-source magnification
  -> amp_point3q()      triple fallback with quad precision roots
  -> amp_finite()       binary finite-source magnification
  -> amp_finite3()      triple finite-source magnification
```

For the first milestone, focus on point-source magnification:

```text
finiteAt()
  -> amp_point2()
  -> amp_point3()
  -> imageposition()
  -> imageposition3()
  -> imageposition3l()
  -> imageposition3q()
  -> zroots2 / zroots2l / zroots2q
```

Finite-source modes should come later, after the point-source binary/triple path
is clean and tested.

## Files Added So Far

Current repo files:

```text
CMakeLists.txt
Makefile
README.md
docs/migration-plan.md
include/lcbinint/lcbinint.h
pyproject.toml
python/lcbinint_pybind.cpp
src/lcbinint/lcbinint.cpp
src/lcbinint/magnification/point_source_magnifier.cpp
src/lcbinint/magnification/point_source_magnifier.hpp
src/lcbinint/math/polynomial_roots.cpp
src/lcbinint/math/polynomial_roots.hpp
src/lcbinint/types.hpp
src/lcbinint/model/lens_model.cpp
src/lcbinint/model/lens_model.hpp
src/lcbinint/model/lens_parameters.cpp
src/lcbinint/model/lens_parameters.hpp
src/lcbinint/model/lens_system.cpp
src/lcbinint/model/lens_system.hpp
src/lcbinint/model/trajectory.cpp
src/lcbinint/model/trajectory.hpp
tests/regression/test_vbm_consistency.py
tests/regression/test_solver_vbm_consistency.py
tests/unit/test_core.cpp
third_party/skowron_gould/SkowronGould.cpp
third_party/skowron_gould/SkowronGould.h
```

`include/lcbinint/lcbinint.h` declares the intended public C-compatible boundary:

```c
lcbi_status lcbi_magnification(
    double time,
    const lcbi_params *params,
    const lcbi_options *options,
    lcbi_result *result
);
```

It intentionally supports both binary and triple-lens parameters in the same
parameter struct. Triple mode is enabled by `q2 > 0`, using `sep2` and `ang`.
The header is C-compatible, but the implementation is C++.

## Numerical Recipes Replacement Plan

Before public release, replace or remove these legacy dependencies:

| Legacy routine | Role | Replacement direction |
| --- | --- | --- |
| `zroots2`, `zroots2l`, `zroots2q`, `laguer*` | complex polynomial roots | Skowron & Gould complex polynomial solver, as used in microlensing tools such as VBBinaryLensing |
| `zbrent4`, `zbrent5` | scalar roots | GSL Brent or clean-room Brent |
| `qromb*`, `trapzd*`, `polint` | finite-source integration | GSL integration or clean-room adaptive integration |
| `piksrt`, `indexx` | sorting | `qsort` or small local sort |
| `dvector`, `dmatrix`, `nrutil` | offset array allocation | normal zero-based `calloc`/`free` |
| `ran1`, `gasdev` | artificial LC/ray jitter | out of first scope, later permissive RNG |
| `powell3`, `linmin2`, `brent2`, `mnbrak2`, `djacobi` | fitting/MCMC support | out of first scope |

GSL was not found in this environment through `pkg-config` at the time of this
note. We still follow `../genulens` and use GSL as a project dependency, but
the polynomial root solver should use Skowron & Gould rather than GSL's generic
polynomial solver.

Update: the repo now follows the `../genulens` build direction:

- CMake C++ project with `lcbinint_core`
- `scikit-build-core` + `pybind11` in `pyproject.toml`
- `Makefile` targets for `core`, `python`, and `test`
- GSL lookup through `find_package(GSL QUIET)`, then a `GSL_ROOT`/system-path
  fallback

Current environment check:

```text
pkg-config --modversion gsl     # not available
gsl-config --version            # not available
/usr/lib64/libgsl.so.25         # exists
/usr/lib64/libgslcblas.so.0     # exists
/usr/include/gsl                # not found
```

Without a GSL hint, `cmake -S . -B build -DLCBININT_BUILD_PYTHON=OFF`
currently fails with:

```text
GSL was not found. Set GSL_ROOT or install GSL development headers/libraries.
```

This is expected until GSL headers are available.

`../genulens` uses:

```text
GSL_INCLUDE_DIR=/rogue1_8/nunota/local/gsl/include
GSL_LIBRARY=/rogue1_8/nunota/local/gsl/lib/libgsl.so
```

Using the same prefix works here:

```text
GSL_ROOT=/rogue1_8/nunota/local/gsl cmake -S . -B build -DLCBININT_BUILD_PYTHON=OFF
cmake --build build --target test_core
ctest --test-dir build --output-on-failure
```

Result: `unit_core` passed.

Update: the core skeleton is now C++, not C. The public header remains
C-compatible.

Important C++ design decision:

- Do not just rename legacy C files to `.cpp`.
- Keep the public ABI small and C-compatible.
- Implement the numerical core with C++ value types/classes.
- Route `lcbi_magnification()` into internal C++ objects.

Planned internal structure:

```text
lcbi_magnification()
  -> lcbinint::LensModel
  -> lcbinint::Trajectory
  -> lcbinint::LensSystem
  -> lcbinint::PolynomialRootSolver
```

Likely classes/value types:

| Type | Responsibility |
| --- | --- |
| `LensParameters` | Validated physical parameters converted from `lcbi_params`. |
| `ComputationOptions` | Numerical and mode options converted from `lcbi_options`. |
| `SourcePosition` | Source coordinate at one time. |
| `LensSystem` | Binary/triple lens geometry and mass fractions. |
| `PolynomialRootSolver` | Skowron-Gould root solving isolated from lens logic. |
| `PointSourceMagnifier` | Binary/triple point-source image finding and magnification. |
| `LensModel` | Orchestrates one model evaluation. |
| `FiniteSourceMagnifier` | Later finite-source/inverse-ray implementation. |

Implemented skeleton:

- `lcbi_magnification()` now converts C structs into C++ `LensParameters` and
  `ComputationOptions`.
- It constructs `lcbinint::model::LensModel`.
- `LensModel` currently delegates source-position calculation to `Trajectory`.
- `LensSystem` already builds binary or triple lens geometry from parameters.
- `PolynomialRootSolver` now exists as the dedicated root-solver boundary. It
  supports linear and quadratic equations analytically, and delegates degree
  three and higher to the vendored Skowron-Gould solver.
- The Skowron-Gould source is vendored under `third_party/skowron_gould` from
  the official upstream distribution and used under Apache-2.0.
- The Python extension exposes `polynomial_roots(coefficients)` for direct
  solver validation.
- `tests/regression/test_solver_vbm_consistency.py` compares this solver API
  against `VBMicrolensing().cmplx_roots_gen(...)`.
- Binary point-source magnification now exists as
  `PointSourceMagnifier::binary_mag0(separation, q, source)` and is exposed in
  Python as `lcbinint.binary_mag0(separation, mass_ratio, y1, y2)`.
- Python also exposes keyword-friendly `LensParams`, `Options`, and `LensModel`
  bindings with scalar and vector magnification methods for supported binary
  point-source cases.
- `tests/regression/test_vbm_consistency.py` compares `binary_mag0()` against
  `VBBinaryLensing().BinaryMag0(...)`, and also checks the Python `LensModel`
  path for the same cases.
- The high-level C ABI `lcbi_magnification()` now returns `LCBI_OK` for a
  restricted binary point-source subset: no triple lens, no finite source, no
  parallax/xallarap/orbital motion. It preserves the legacy wide-binary
  `sep > 1` center-of-caustic offset when `center_of_mass == 0`. Unsupported
  cases still return `LCBI_UNSUPPORTED`.
- Unit test verifies the C ABI routes through the C++ trajectory path and that
  the root solver handles degree 1, 2, 3, and 5 polynomials.

## Near-Term Plan

1. Keep the public C ABI small and stable.
2. Add internal C++ value types and route the C ABI through `LensModel`.
3. Add a Skowron-Gould polynomial root solver wrapper. Done for degree 3+.
4. Port binary point-source magnification into that structure. Initial
   low-level API done and VBM-tested.
5. Port triple point-source magnification.
6. Connect binary point-source magnification into `LensModel` and the C ABI
   after coordinate/trajectory behavior is verified.
7. Add regression tests against selected outputs from the legacy executable.
8. Add user-facing Python `LensModel` bindings after the C++ library path is
   stable.
9. Add finite-source modes and limb darkening later.

## Design Decisions Already Made

- No subprocess backend for the public API.
- Binary and triple lens should use one model and one parameter struct.
- `q2 > 0` means triple lens.
- Fitting and MCMC are not first milestone.
- The first useful API is magnification-only.

## VBM Consistency Testing

`tests/regression/test_vbm_consistency.py` now pins a small set of
VBBinaryLensing `BinaryMag0(separation, q, y1, y2)` reference values.

The VBM reference-value test is active. The `lcbinint` comparison test is marked
`xfail` until the binary point-source Python API exists. The expected direct
validation API is:

```python
lcbinint.binary_mag0(separation, mass_ratio, y1, y2)
```

Keep this low-level API separate from the future user-facing `LensModel`
trajectory API so the root solver and image finder can be tested directly.

## Open Questions

- Should the root solver depend on GSL, or should we implement a clean-room
  solver locally?
- Should the first Python binding use Cython, pybind11, cffi, or CPython C API?
- How much legacy behavior should be preserved exactly for the first regression
  suite?
- Which legacy command/output cases should become canonical regression fixtures?
