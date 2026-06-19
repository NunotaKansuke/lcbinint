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
is clean and tested. New finite-source work should focus on inverse-ray
strategies, not on preserving the old `FINITE=1` NR-style integration path.

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
src/lcbinint/magnification/finite_source_magnifier.cpp
src/lcbinint/magnification/finite_source_magnifier.hpp
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
| `qromb*`, `trapzd*`, `polint` | legacy finite-source integration | Do not port as the public finite-source path; prefer inverse-ray plus quadrupole/point-source safety tests |
| `piksrt`, `indexx` | sorting | `qsort` or small local sort |
| `dvector`, `dmatrix`, `nrutil` | offset array allocation | normal zero-based `calloc`/`free` |
| `ran1`, `gasdev` | artificial LC/ray jitter | out of first scope, later permissive RNG |
| `powell3`, `linmin2`, `brent2`, `mnbrak2`, `djacobi` | fitting/MCMC support | out of first scope |

GSL is available on this machine under `/rogue1_8/nunota/local/gsl`. We still
follow `../genulens` and use GSL as a project dependency, but the polynomial
root solver uses Skowron & Gould rather than GSL's generic polynomial solver.

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

The CMake fallback now checks this local prefix automatically:

```text
/rogue1_8/nunota/local/gsl
```

This also still works explicitly:

```text
GSL_ROOT=/rogue1_8/nunota/local/gsl cmake -S . -B build
cmake --build build --target test_core
ctest --test-dir build --output-on-failure
```

Current verification:

```text
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
PYTHONPATH=build pytest -q tests/regression
```

Result: `unit_core` passed and regression tests passed.

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
| `FiniteSourceMagnifier` | Finite-source strategy selection and inverse-ray implementation. |

Implemented skeleton:

- `lcbi_magnification()` now converts C structs into C++ `LensParameters` and
  `ComputationOptions`.
- It constructs `lcbinint::model::LensModel`.
- `LensModel` currently delegates source-position calculation to `Trajectory`.
- `Trajectory` supports annual parallax through a C++ port of the
  `../jacscanomaly` Earth-orbital parallax projector style: load the Horizons
  Earth ephemeris table, linearly interpolate position/velocity, project onto
  sky north/east, subtract the reference position and velocity at `tfix`, and
  apply `piEN/piEE`. The projector object owns the sky basis and reference
  state internally.
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
  bindings with scalar/vector magnification methods, source trajectory helpers,
  and `light_curve(times)` for supported binary cases. `light_curve()` returns
  sampled times, magnifications, point-source/finite-source components,
  source-plane coordinates, and image counts.
- `tests/regression/test_vbm_consistency.py` compares `binary_mag0()` against
  `VBBinaryLensing().BinaryMag0(...)`, and also checks the Python `LensModel`
  path for the same cases.
- The high-level C ABI `lcbi_magnification()` now returns `LCBI_OK` for
  supported binary point-source and finite-source cases, including annual
  parallax and the new lens orbital-motion modes. Triple lens and old
  `omega/v_sep` style orbital-motion fields remain unsupported in the main
  model path. It preserves the legacy wide-binary `sep > 1`
  center-of-caustic offset when `center_of_mass == 0`.
- Public finite-source mode selection has been simplified. The legacy
  `FINITE=1..6` mode numbers are not part of the new user-facing API. Users
  should provide `tolerance` and `relative_tolerance`; the C++ core chooses
  between a point-source fast path, cartesian inverse-ray, and polar inverse-ray
  internally. Cost estimates are kept as diagnostics for refinement planning,
  not as a reason to silently downgrade to a cheaper method.

## Lens Orbital Motion

The C++ core now has a first LOM implementation in
`src/lcbinint/model/orbital_motion.cpp`.

Implemented modes:

- `LCBI_ORBIT_STATIC`: default static binary.
- `LCBI_ORBIT_CIRCULAR`: VBBinaryLensing-style circular 3D orbital motion. The
  public parameters are `g1=(1/s) ds/dt`, `g2=dalpha/dt`, and
  `g3=(1/s) dsz/dt`.
- `LCBI_ORBIT_KEPLER`: VBBinaryLensing-style eccentric/Keplerian 3D orbital
  motion with `g1,g2,g3,szs,ar`.

Python exposure:

- `LensParams(..., orbital_motion_mode=..., g1=..., g2=..., g3=..., lom_szs=..., lom_ar=...)`
- `OrbitalMotionMode.STATIC`, `CIRCULAR`, `KEPLER`
- low-level checks:
  - `lcbinint.circular_orbital_motion(...)`
  - `lcbinint.kepler_orbital_motion(...)`

LOM uses the same reference epoch as annual parallax: `tfix` when it is set,
otherwise `t0`. There is no separate public `lom_tref` in the model
parameters.

`LensModel` uses the instantaneous projected separation `s(t)` and rotates the
source coordinate into the instantaneous lens frame. The static case is
preserved exactly because the applied rotation is only `alpha(t)-theta`.

Verification added:

- circular LOM state compared against the vendored microjax/VBBinaryLensing
  compatible formula from `../microlux`.
- Kepler/eccentric LOM state compared against the same microjax implementation.
- `LensModel` circular LOM path checked against a manual `binary_mag0`
  evaluation using the instantaneous state.

## Current Public-Package Direction

The current usable Python surface is:

```python
params = lcbinint.LensParams(
    t0=...,
    tE=...,
    umin=...,
    q=...,
    sep=...,
    theta=...,
    rho=...,
    piEN=...,
    piEE=...,
    ra=...,
    dec=...,
    tfix=...,
    orbital_motion_mode=lcbinint.OrbitalMotionMode.CIRCULAR,
    g1=...,
    g2=...,
    g3=...,
)
curve = lcbinint.LensModel(params, lcbinint.Options()).light_curve(times)
```

Near-term plan:

- Keep the C-compatible boundary small, but make the Python API more ergonomic
  for fitting/MCMC by adding a higher-level function or class that accepts
  frequently changing model parameters at call time.
- Continue using VBBinaryLensing/microjax/microlux as numerical references for
  binary point-source, finite-source, parallax, and LOM behavior.
- Add direct VBBinaryLensing light-curve comparisons only when the binding can
  be called in a lightweight point-source-like setup; current direct
  `BinaryLightCurveOrbital/Kepler` calls are too heavy for routine regression.
- Triple-lens point-source and finite-source support are still pending.
- Old `omega/v_sep` fields remain legacy/unsupported. New LOM work should use
  `OrbitalMotionMode` and `g1,g2,g3`.
- Expert users can force the inverse-ray variant with
  `Options(inverse_ray_method=InverseRayMethod.CARTESIAN/POLAR)`. The default
  is still automatic selection.
- Expert users can select `FiniteSourceMode.LEGACY` to preserve old-style
  KINJI/HEX finite-source decisions. This path uses sampled binary-caustic
  distance and exposes `legacy_finite_mode`, `legacy_kinji`, `legacy_hex`,
  `caustic_bins`, `source_bins`, and `grid_ratio`.
- `FiniteSourceMagnifier` now exists as the finite-source strategy boundary. It
  implements method selection, cost estimates, a Bozza/microlux-style
  quadrupole safety test, and initial binary inverse-ray area estimators for
  cartesian and polar image-plane grids. Polar inverse-ray remains an internal
  strategy candidate for high-magnification cases.
- The finite-source fast path now uses a C++ port of `../microlux`
  `Quadrupole_test()`: quadrupole correction, cusp correction, ghost-image
  test, and planetary-caustic test. If it passes, the point-source
  magnification is used.
- The inverse-ray path has an internal coarse/fine refinement diagnostic:
  evaluate at `source_bins`, then at doubled bins. Once two consecutive
  differences are available, it estimates a conservative remaining tail from
  the observed convergence ratio and compares that diagnostic against
  `tolerance` or `relative_tolerance * abs(fine)`. This is still not a rigorous
  error bound.
- Binary finite-source limb darkening is wired through `LensParams` using
  `limb_darkening_c` and `limb_darkening_d`. Inverse-ray weights samples by
  `1-c(1-mu)-d(1-sqrt(mu))`; hexadecapole uses the legacy Gamma/Lambda
  correction.
- Non-converged finite-source evaluations now propagate as
  `LCBI_NUMERICAL_ERROR` through the C ABI. The Python `LensModel` wrapper
  raises `RuntimeError("numerical error")`.
- See `.note/finite-source-strategy.md` for the finite-source plan and the
  corrected interpretation of microJAX as guidance for accuracy-control logic,
  not as an implementation to port wholesale.
- `tests/regression/test_vbm_consistency.py` includes initial finite-source
  comparisons against `VBBinaryLensing().BinaryMag2(...)`. These are small
  source, non-pathological binary cases and are not yet a full caustic-crossing
  validation suite.
- Unit test verifies the C ABI routes through the C++ trajectory path and that
  the root solver handles degree 1, 2, 3, and 5 polynomials.

## 2026-06-20 Polar / High-Magnification Update

- Legacy `smode=6` no longer builds the cached polar image-plane map before it
  knows whether it will fall back to the cartesian legacy area path. This avoids
  doing polar setup work that is immediately discarded near caustics.
- The augmented caustic seed list is reused when `smode=6` falls back to the
  legacy cartesian area calculation, instead of recomputing the same caustic
  seed search inside `imagearea4`.
- The pure polar `smode=5` path is still not accurate enough for difficult
  caustic/high-magnification curves. It remains exposed as a legacy expert
  option, but precision regression coverage now focuses on `smode=4` and
  `smode=6`.
- Added a high-magnification light-curve regression against
  `VBBinaryLensing.BinaryLightCurve(...)` using forced finite-source legacy
  settings. Both `smode=4` and `smode=6` are checked.
- Current quick benchmark with `source_bins=80`:
  - caustic light curve, `s=1.4, q=0.4, u0=-0.15, rho=0.025`:
    VBBL 0.044 ms/pt, `smode=4` 0.205 ms/pt, `smode=6` 0.179 ms/pt,
    max relative error about `9.6e-4`.
  - forced finite-source high-magnification curve,
    `s=1.0, q=0.1, u0=0.01, rho=0.003`:
    VBBL 0.126 ms/pt, `smode=4` 2.85 ms/pt, `smode=6` 2.74 ms/pt,
    max relative error about `4.1e-4`.

## 2026-06-20 Source-Bin Accuracy Diagnostic

- Python `LensModel` now exposes
  `estimate_source_bins(times, candidate_bins=[20,30,40,50,60,80], max_sample_points=64)`.
- The diagnostic evaluates representative light-curve times at the largest
  candidate bin count, treats that as the internal reference, and reports
  max/rms relative differences for smaller candidates. It recommends the first
  candidate satisfying `tolerance + relative_tolerance * abs(reference)` at all
  sampled points.
- This is a self-convergence diagnostic, not a rigorous integration error
  bound. It is meant to catch obviously over-conservative `source_bins` choices
  before running full light curves.
- With `relative_tolerance=1e-3`, the current caustic benchmark recommends
  `source_bins=50` against an 80-bin internal reference. The forced
  high-magnification benchmark recommends `source_bins=40`. Direct VBBL sweeps
  still suggest `50-60` is the safer default for the caustic example, while
  `80` is usually overkill for the current mode-4/mode-6 implementation.

## Near-Term Plan

1. Keep the public C ABI small and stable.
2. Add internal C++ value types and route the C ABI through `LensModel`.
3. Add a Skowron-Gould polynomial root solver wrapper. Done for degree 3+.
4. Port binary point-source magnification into that structure. Initial
   low-level API done and VBM-tested.
5. Finish the finite-source strategy layer enough to make binary finite-source
   results diagnosable: compare the quadrupole safety classification against
   VBM/microlux grids, add non-convergence status propagation, and improve
   inverse-ray variant selection.
6. Port triple point-source magnification.
7. Add regression tests against selected outputs from the legacy executable.
8. Improve finite-source inverse-ray region construction and add limb darkening.

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
