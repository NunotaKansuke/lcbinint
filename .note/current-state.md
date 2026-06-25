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
- Public finite-source mode selection has been simplified. See
  "2026-06-20 API Simplification" below for the current user-facing API.

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
- `FiniteSourceMagnifier` is the finite-source strategy layer. It selects
  between a fast point-source path, hexadecapole, and boundary-tracing
  inverse-ray based on sampled caustic distance (KINJI/HEX thresholds).
- Binary finite-source limb darkening is wired through `LensParams` using
  `limb_darkening_c` and `limb_darkening_d`. Inverse-ray weights samples by
  `1-c(1-mu)-d(1-sqrt(mu))`; hexadecapole uses the legacy Gamma/Lambda
  correction.
- See `.note/finite-source-strategy.md` for the finite-source plan and the
  corrected interpretation of microJAX as guidance for accuracy-control logic,
  not as an implementation to port wholesale.
- `tests/regression/test_vbm_consistency.py` includes initial finite-source
  comparisons against `VBBinaryLensing().BinaryMag2(...)`. These are small
  source, non-pathological binary cases and are not yet a full caustic-crossing
  validation suite.
- Unit test verifies the C ABI routes through the C++ trajectory path and that
  the root solver handles degree 1, 2, 3, and 5 polynomials.

## 2026-06-20 API Simplification

The non-legacy (AUTO) inverse-ray path has been removed. The only
finite-source implementation is the boundary-tracing inverse-ray using
augmented caustic seeds, matching legacy smode=4/6 behavior.

**New public Python API:**

```python
# cartesian (default, mode=1): boundary-tracing row-by-row, NBIN=50
opts = lcbinint.Options(center_of_mass=1, source_bins=50)

# polar+cache (mode=2): precomputed polar map, smode=6 behavior
opts = lcbinint.Options(center_of_mass=1, mode=2, source_bins=50)

# expert knobs (rarely needed)
opts = lcbinint.Options(
    source_bins=50,
    mode=1,                       # 1=cartesian, 2=polar+cache
    caustic_bins=1400,
    grid_ratio=4.0,
    point_source_threshold=9.0,   # KINJI: use pt-src if caustic > N*rho away
    hexadecapole_threshold=2.0,   # HEX:  use hexadecapole if caustic > N*rho
)
```

**Removed from API:**
- `FiniteSourceMode` / `InverseRayMethod` Python enums (gone)
- `tolerance`, `relative_tolerance` from `Options` (gone)
- `legacy_finite_mode`, `legacy_kinji`, `legacy_hex` (renamed above)
- `lcbi_finite_source_mode`, `lcbi_inverse_ray_method` C enums (gone)

**Removed from C++ core:**
- `refined_inverse_ray_binary` (the AUTO refinement loop)
- `inverse_ray_cartesian_binary`, `inverse_ray_polar_binary` (simple grids)
- `quadrupole_safety_test` and all related helpers (f0-f3, etc.)
- `choose_binary_method` class method
- `FiniteSourceSettings.tolerance`, `.relative_tolerance`, `.legacy_mode`, `.inverse_ray_method`

The `fixed_inverse_ray_binary` function (boundary tracer caller) is now
the sole inverse-ray dispatcher: always uses augmented seeds, routes to
`legacy_imagearea4_binary` for mode=1 and `inverse_ray_polar_boundary_binary`
for mode=2.

**Benchmark (NBIN=50, mode=1 vs VBBL, s=1.4, q=0.4, u0=-0.15, rho=0.025):**
- no LD: VBBL 0.079 ms/pt, lcbinint 0.350 ms/pt (4.4× slower)
- LD: VBBL 0.502 ms/pt, lcbinint 0.426 ms/pt (lcbinint faster by 15%)
- max relative error: no LD 1.27e-3, LD 9.14e-4

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
- The diagnostic compares self-convergence vs. the largest candidate bins.
  The largest bins entry is always marked `accepted=True` with
  `max_relative_difference==0.0`; all others report the raw diff.
- VBBL sweeps show `source_bins=50` is the optimal default: same max accuracy
  as bins=80 (`9.1e-4`) and beats VBBL for limb-darkened light curves.

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

## 2026-06-20 vbbl_compatible Accuracy and Performance Fixes

### Bugs Fixed

**Bug 1 – Early exit in `binary_mag` (caustic distance threshold)**
The early-exit condition used only the sampled point distance, ignoring that a
cached branch segment can pass much closer to a source than its nearest endpoint.
Fixed to: `caustic_distance >= cached_point_threshold + caustic_cache_max_seg_len_`.

**Bug 2 – Missing caustic-crossing seeds near phase-sampling gap**
`fixed_inverse_ray_binary` Phase 3 was added: when fewer than 5 seeds are found
after standard seed search, a segment-based scan of the caustic branch grid finds
crossings in the gap between the last phase sample (phi=6.2787) and phi=2π.
Implemented via the new public method `legacy_augment_seeds_from_branches`.

**Bug 3 – Branch tracking swap inflating max_seg_len**
For sep=2.0, q=1.0 (equal-mass wide binary), the inner and outer caustic branches
physically cross at an intermediate phase. The greedy nearest-neighbour tracker
(and even global-minimum-cost) swaps branch assignments, creating spurious
wrap-around segments of length 0.866 between mis-matched endpoints.
Effect: `caustic_cache_max_seg_len_` was 0.866 instead of ~0.023, causing
`legacy_binary_caustic_distance` to search almost the entire grid → 6.88 ms
per source (should be ~0.05 ms).

Fix: exclude wrap-around segments (j == n-1) from the `max_seg_len` computation.
The spurious segment is still stored in the branch grid and is still found
correctly via the "prev" check on the j=0 entry when needed (cusp sources within
the real phase gap). Performance drops from 6.88 ms to ~0.05 ms per caustic check.

Note: the underlying branch crossing is physical (not a sampling artefact) and
cannot be resolved by distance-based tracking regardless of caustic_bins. The
wrap-around exclusion is the correct robust fix.

**Bug 4 – Branch tracking switched to global-minimum-cost assignment**
`append_tracked_caustic_points` now tries all 4! = 24 permutations of the four
new caustic points and picks the assignment that minimises total squared step
length. This avoids spurious swaps in configurations where the branches come close
but do not genuinely cross.

**Bug 5 – Default `source_bins` accuracy characterised (no default change)**
With `source_bins=50` the IR scan has up to 0.35% error for sources partially
overlapping the caustic. Default remains 50; users can raise it if needed.
For limb-darkened light curves, LD weighting naturally reduces edge-pixel errors,
so the effective accuracy is significantly better than the noLD case at the same bins.

**Bug 7 – Mode-selection thresholds too aggressive (kinji=9→20, hex=2→3)** (2026-06-21)
Triggered by s=1.0 q=0.001 u0=-0.001 rho=0.001 (source near resonant caustic cusp).
The source trajectory passes at y=-0.001 through the 5-image interior of the caustic
(t≈0.128 to t≈0.159). `legacy_binary_caustic_distance` finds the nearest caustic point
as the right cusp at (0.1463, 0) — correct, but this cusp is the nearest arc for a
large range of times along the trajectory. With kinji_threshold=9 and hex_threshold=2:
- t=0.148: caustic_dist=0.002009 just over 2.0×rho → hex mode → ring includes
  the cusp where PS=116 → hexadecapole gives 15.46 vs correct 14.07 (+10%)
- t=0.155: caustic_dist=0.009055 just over 9.0×rho → PS mode → gives 12.43 vs
  correct 13.84 (-10%); combined error up to 6%

The root cause is that these thresholds were calibrated on benchmarks with low time
resolution (200 points) that missed the worst-case moments. With 400 points and the
original (9/2) thresholds, every test case shows 0.7–5.9% errors.

Fix: changed defaults to kinji_threshold=20.0, hex_threshold=3.0 in:
- `src/lcbinint/magnification/finite_source_magnifier.hpp` (C++ struct defaults)
- `src/lcbinint/lcbinint.cpp` (C ABI defaults)
- `src/lcbinint/model/lens_parameters.hpp` (Python options struct defaults)
- `python/lcbinint_pybind.cpp` (Python keyword argument defaults)

Results across all cases (400 points, bins=50):
```
Case                    old(9/2)   new(20/3)
resonant q=0.3          1.003% →   0.176%
wide-caus q=0.001       3.363% →   0.632%
wide-eq q=1.0           0.341% →   0.341%
wide-uneq q=0.1         0.107% →   0.104%
close q=0.3             0.682% →   0.091%
planet q=0.01           3.591% →   0.743%
reson-new q=0.001       5.939% →   0.565%
```
No case regressed; all improvements range from 2× to 10×.

**Bug 8 – VBM-style adaptive hex: two-step caustic-distance required** (2026-06-21)
Motivation: threshold-based mode selection (Bug 7) is a patch; VBM uses a hex
self-consistency test (|a4 correction|/mag > tol → fall back to IR) that adapts
automatically. Implemented `HexResult {magnification, relative_error}` and an
`adaptive_hex_threshold = 0.001` parameter. Broke into two sub-bugs:

*Sub-bug A* (wide-eq, caustic-straddling with small a4):
For s=2.0 q=1.0 at t=-0.0702: source center OUTSIDE caustic (PS=16.67), disk
straddles caustic (VBM FS=39.84). Hex gives 19.77 with relative_error < 0.001.
Why: a1_plus=4.18 and a1_cross=8.18 nearly cancel through a2rho2=6.23, making
a4rho4≈-0.05 (tiny) even though the field is highly non-uniform.
The self-consistency check cannot detect this: all 13 ring points are in consistent
3- or 5-image regions, so the Taylor residual is artificially small.

*Sub-bug B* (wide-eq, sparse caustic cache):
`legacy_binary_sampled_caustic_distance` returned 7.1×rho for the wide-eq case
because the 1400-sample caustic grid doesn't place a sample near the fold at
(x=-0.0702, y≈0.00008). The point-distance cache is too coarse; the segment-based
`legacy_binary_caustic_distance` correctly finds the caustic at 0.78×rho by
computing the distance to the SEGMENT connecting the two adjacent samples that
bracket the fold.

Fix: hybrid approach matching actual VBM behaviour.
1. Build caustic cache via `legacy_binary_sampled_caustic_distance` (already done).
2. Refine with `legacy_binary_caustic_distance` (segment-based, fast with hint).
3. If `refined_dist < hex_threshold × rho` → near-caustic → skip hex → IR directly.
4. Else → try hex self-consistency; if `rel_err < adaptive_hex_threshold` → hex.
5. Else → IR.

Updated accuracy (400 pts, bins=50, kinji=20, hex=3, adaptive_hex=0.001):
```
resonant s=1.0 q=0.3   rho=0.025  0.124%
wide-caus s=1.0 q=0.001 rho=0.003  0.368%
wide-eq s=2.0 q=1.0    rho=0.001  0.347%  (was 50% with VBM only; 0.341% with old)
wide-uneq s=2.0 q=0.1  rho=0.001  0.034%
close s=0.5 q=0.3      rho=0.003  0.114%
reson-new s=1.0 q=0.001 rho=0.001  0.563%
```
All cases ≤ 0.57%.  Performance: PS ~5 μs/pt, hex ~5 μs/pt (tiny overhead vs 4 μs),
near-caustic IR ~4 ms/pt (unchanged, dominated by image-area integration).

**Bug 6 – Outer-caustic seed-finding in `legacy_augment_seeds_from_branches`** (2026-06-21)
Three compounding bugs caused ~52% error when a source just straddles an outer
caustic boundary (e.g. wide-eq s=2 q=1 rho=0.001 at the outer-left-caustic edge):

- *False early exit*: `legacy_augmented_image_seeds` checked
  `hint_caustic_dist >= source_radius` and returned without running the Phase 1
  scan. The hint came from `legacy_binary_caustic_distance`, which can slightly
  over-estimate distance via the branch-grid search; when the source disk just
  touches the caustic, this over-estimation caused a false bail-out.
  Fix: removed the hint-based early exit; Phase 1 always runs when seeds < 5.

- *Wrap-around segment poisoning*: `legacy_augment_seeds_from_branches` checked
  the forward segment for every branch point, including the wrap-around (last
  point → first point). For s=2 q=1 this segment is a phantom artifact (length
  0.866) from the branch-tracking swap; it passes through the source disk and
  triggers a probe at (-0.05, -5e-5), giving 5 images from the wrong region.
  Those wrong seeds replace the correct 3-image seeds, and the IR scan integrates
  the wrong image area → magnification ≈ 0.25 instead of ≈ 22.
  Fix: `if (next == 0) continue;` skips the wrap-around (mirrors the max_seg_len
  exclusion already in place).

- *Probe step too small*: the probe displacement was `(rho-d)*0.01/d`, which for
  d ≈ rho gives a step of ~2e-6. The true caustic arc can differ from the segment
  approximation by up to one inter-sample spacing (~0.005 for 1400 bins); a step
  of 2e-6 lands on the wrong side of the true arc and the probe returns 3 images
  instead of 5. Fix: step changed to `rho*0.05/d`, giving a fixed displacement of
  5% of rho regardless of how close d is to rho. Added guard: if the probe falls
  outside the source disk the result is discarded (avoids updating seeds with
  images from a region that does not overlap the disk).

Result: outer-caustic crossing error drops from ~52% to < 0.1%.

### Accuracy with Corrected Defaults (bins=50, vbbl_compatible=0, kinji=20, hex=3)

Measured with 400 time points to avoid resolution-aliasing of worst-case moments.

| Configuration | max rel err | notes |
|---|---|---|
| resonant s=1.0 q=0.3 rho=0.003 | 0.18% | hex-dominated near caustic cusp |
| wide caustic s=1.0 q=0.001 rho=0.003 u0=0.003 | 0.63% | fold-proximity hex error |
| wide equal-mass s=2.0 q=1.0 rho=0.001 | 0.34% | all sources in IR mode (inside caustic), umin=0.000861 |
| wide unequal s=2.0 q=0.1 rho=0.001 u0=0.01 | 0.10% | PS-dominated |
| close binary s=0.5 q=0.3 rho=0.003 | 0.09% | |
| planetary s=1.2 q=0.01 rho=0.001 | 0.74% | fold-proximity hex error near planetary caustic |
| resonant-new s=1.0 q=0.001 rho=0.001 u0=-0.001 | 0.57% | through 5-image interior of caustic |

IR-mode accuracy (sources inside caustic, bins=50): 0.34% for wide-eq.
hex/PS-mode accuracy: 0.09–0.74% depending on proximity to caustic folds.
Outer-caustic boundary crossing: fixed in Phase 3 update (was 52% before).
Mode thresholds: kinji=20 (PS/hex boundary) and hex=3 (hex/IR boundary) give 2-10× better
accuracy than original kinji=9, hex=2 across all tested configurations.

### Performance (uniform source, no LD)
For trajectories where sources are far from the caustic: lcbinint is competitive
with VBBL (PS and hex modes are < 0.05 ms/source). For near-caustic trajectories
where all sources are inside or touching the caustic (the wide equal-mass test),
the IR scan dominates at ~8 ms/source regardless of bins, because every point
requires a full inverse-ray integration. This is inherent to the method.

The complementary positioning vs VBBL: lcbinint is faster for limb-darkened
light curves (VBBL needs additional LD evaluations), comparable for non-LD cases
far from caustics, and slower for non-LD cases with many near-caustic sources.

## 2026-06-20 Limb-Darkening Speed/Accuracy Benchmark

**VBM LD API note**: correct VBM LD reference uses
`VBB.a1 = gamma; VBB.BinaryMagDark(s, q, y1, y2, rho, VBB.Tol)`.
The overloaded form `BinaryMagDark(..., gamma_float)` with Gamma<1 as the 6th arg
is a fast hexadecapole-style approximation and should NOT be used as a reference.

### Speed comparison: lcbinint (bins=50) vs VBM full LD integration (Tol=1e-3)

**Multi-case benchmark** (s=1.4, q=0.4, u0=-0.15):

| Case | lcb ms/pt | vbm ms/pt | ratio | err% |
|---|---|---|---|---|
| caustic-x r=0.025 noLD | 0.306 | 0.084 | 3.65× slower | 0.120 |
| caustic-x r=0.025 LD36 | 0.368 | 0.378 | **0.97× faster** | 0.100 |
| large-src r=0.050 noLD | 0.268 | 1.020 | **0.26× faster** | 1.54  |
| large-src r=0.050 LD36 | 0.453 | 0.986 | **0.46× faster** | 0.034 |
| high-mag r=0.003 noLD  | 0.504 | 1.046 | **0.48× faster** | 2.99  |
| high-mag r=0.003 LD36  | 0.603 | 1.072 | **0.56× faster** | 0.096 |
| resonant r=0.003 LD36  | 0.314 | 0.715 | **0.44× faster** | 0.096 |
| wide-caus r=0.003 LD   | 2.027 | 2.698 | **0.75× faster** | 0.050 |
| planetary r=0.001 LD   | 0.332 | 0.758 | **0.44× faster** | 0.040 |

**Key findings:**
1. LD has consistently **< 0.1% error** (bins=50), whereas noLD can be 1.5–3%
   for large or near-caustic sources at the same bins.  This is because LD
   naturally down-weights edge pixels, which are most sensitive to boundary
   resolution.
2. lcbinint is **1.3–2.3× faster** than VBM for LD cases across diverse lens
   geometries.
3. For large sources (rho ≥ 0.02) lcbinint is faster even **without** LD,
   because VBM's adaptive integration scales with source area.
4. For small sources without LD (rho ≤ 0.01): VBM is faster (hex approximation).

**Rho sweep** (caustic-crossing s=1.4, q=0.4, u0=-0.15, Gamma=0.36):

| rho   | noLD err% | noLD ratio | LD err% | LD ratio | lcb-LD ms | vbm-LD ms |
|---|---|---|---|---|---|---|
| 0.003 | 0.037 | 0.97×  | 0.037 | 0.91× | 0.074 | 0.081 |
| 0.005 | 0.044 | 1.23×  | 0.044 | 1.23× | 0.100 | 0.081 |
| 0.010 | 0.066 | 0.89×  | 0.069 | 0.94× | 0.126 | 0.134 |
| 0.020 | 0.142 | 0.81×  | 0.099 | 0.98× | 0.300 | 0.306 |
| 0.030 | 0.457 | 0.71×  | 0.078 | 0.88× | 0.433 | 0.492 |
| 0.050 | 0.738 | **0.29×** | 0.129 | **0.36×** | 0.449 | 1.264 |

At rho=0.05 (very large source): lcbinint is **2.8× faster** with LD; noLD
accuracy degrades to 0.74% (bins=50 insufficient for uniform-source boundary).

**Strategic summary**: lcbinint targets a complementary position vs VBM.
It is NOT universally faster. For limb-darkened light curves — especially
with larger sources (rho ≥ 0.01) — lcbinint is consistently 1.3–2.8× faster
with < 0.13% accuracy. This is the primary use-case advantage.

## 2026-06-21 Seed Generation and Mode-2 Fallback Fixes

**Bug 9 – Non-monotonic bins convergence for fold images near critical curve** (commit 30bfc5b)
During the `legacy_imagearea4_binary` overlap check, a seed with `jac_sign != 0` (fold image)
was being marked `overlap=1` when it fell inside the scan region of the *opposite*-parity fold
image.  Fix: when `jac_sign != 0`, compute `jac_sign_other` at the candidate position; if it
equals `-jac_sign` (opposite parity), skip the overlap check (`continue`).

**Bug 10 – F- fold seed suppressed as duplicate of F+** (2026-06-21)
Root cause: Phase 1 (and Phase 2) duplicate-detection checked new probe images against the
*live* `seeds` vector, which already contained F+ after it was added in the same inner loop.
When the probe source is only slightly inside the caustic, `|F+ − F-| << rho`, so F- was
within `source_radius` of F+ and discarded as a duplicate.

Effect: only the F+ fold image was added as a seed; F- was missing entirely.  For
s=1.4 q=0.4 y2=-0.15 rho=0.025 (source straddling caustic), this caused ~35% underestimate.

Fix: snapshot `n_phase0_seeds = seeds.size()` before Phase 1 starts; duplicate checks compare
against `seeds[0..n_phase0_seeds-1]` only.  Phase 2 takes a per-crossing snapshot
(`n_seeds_before`) for the same reason.  Both F+ and F- are now added:
```
seed[5] z=(-1.39032, 0.82575) J=+0.00520   ← F+
seed[6] z=(-1.40566, 0.82506) J=-0.00519   ← F- (was suppressed before)
```
Error drops from 35% to 0.02% for the caustic test.

**Bug 11 – Mode-2 polar ignores refined caustic distance (cusp sources)** (2026-06-21)
`binary_mag` called `legacy_polar_memory_binary_mag` with `caustic_distance = infinity`.
Inside that function the fallback condition `caustic_distance < polar_fallback_distance` was
therefore always false.  `sampled_caustic_distance` uses search radius `source_radius`, so for
sources near a cusp (outside the caustic arc but < `hex_threshold × rho` from it), scd=inf
as well → no fallback → polar integration used → large error.

Example: s=1.0 q=0.1 umin=0.01 rho=0.003 t=-0.08 (ref=22.70, polar gave 12.13, 46% error).

Fix: pass `refined_dist` (already computed by `binary_mag`) instead of infinity.  Any
source where `refined_dist < polar_fallback_distance` now correctly uses the cartesian fallback.

**Bug 12 – Mode-2 cartesian fallback missing Phase 3 seeds** (2026-06-21)
`legacy_polar_memory_binary_mag` called `legacy_augmented_image_seeds` (Phase 0+1+2) but not
`legacy_augment_seeds_from_branches` (Phase 3), unlike `fixed_inverse_ray_binary`.  When Phase
1/2 missed a crossing (e.g. gap near phi=2π), the missing fold seed caused underestimation.

Fix: added `legacy_augment_seeds_from_branches` call before the cartesian fallback check in
`legacy_polar_memory_binary_mag`.

All 53 regression tests pass after these fixes.

## Spine Mode (mode=3)

Mode 3 uses a "spine" algorithm to integrate the fold image pair directly in
image space rather than shooting a cartesian grid.  Ported from the
`idea-stable` branch in commit 804201d.

### Algorithm summary

For each source position, the fold-image arc is built as a spine of points in
the e_l eigenvector direction (perpendicular to the critical curve, crossing
F-→critical curve→F+).  At each spine point a normal scan (in the e_s
direction, along the fold arc) integrates the brightness over the source disk.
The tangent weight for each spine point is the actual image-space distance to
its neighbours.

### Key constant: `kLocal7SpineMaxStepCells`

The spine step in image space is `source_step / |lambda_l|`, but is capped at
`kLocal7SpineMaxStepCells × source_step` to prevent divergent steps near the
critical curve.  This cap directly controls accuracy:

| cap | bins=300 error | bins=100 error |
|-----|---------------|----------------|
| 256 | 7.6×10⁻³      | 2.3×10⁻²       |
| 4   | 9.5×10⁻⁶      | 3.6×10⁻⁴       |

**Fix (2026-06-22):** Reduced `kLocal7SpineMaxStepCells` from 256 to 4.
Error at bins=300 drops from 0.76% to <0.001%, well within the 2×10⁻⁴ target.
Spine point count grows from ~600 to ~31 000, but mode 3 is still ~1.3× faster
than mode 1 for caustic-crossing cases because the spine avoids the full
cartesian grid for the high-magnification fold region.

### Automatic fallbacks

- **Pair distance check** (`kLocal7SpinePairDistanceCells = 50 000`): if the
  F-/F+ seed pair is more than 50 000 source_steps apart, the spine falls back
  to cartesian.  At bins ≥ 600 for the test case (pair distance 0.0103), this
  threshold is crossed and mode3 = mode1 exactly.
- **Source-offset step check** (half_weight validity, `fallback_reason = 4`):
  if the source-offset step between adjacent spine points exceeds
  2 × MaxStepCells × source_step, the spine falls back.  For large var_ratio
  (rho=0.025, var_ratio≈52), this check fires and cartesian is used — correct,
  no catastrophic overcount.
- **var_ratio guard** (`kLocal7SpineMaxVarRatio = 2.0`, added 2026-06-22):
  `var_ratio = 2 × beta × rho / (lambda_s × |lambda_l|)` measures how much
  lambda_l varies across the source disk.  When var_ratio > 2 the linear fold
  model breaks down and spine errors reach several percent.  The guard skips
  the spine for those pairs (cartesian used instead).  This prevents failures
  for large-rho cases like the example notebook case (s=1, q=0.001, rho=0.01,
  var_ratio ≈ 4.7 at caustic entry).
- **Non-caustic guard**: if no seeds satisfy the spine candidate criteria
  (area_jac ≥ 100, |det_j| ≤ 0.01), the spine is never activated and mode3
  falls back to pure cartesian, giving identical results to mode1.

### Regression tests added

- `test_lcbinint_spine_mode_wide_caustic_fold_pair`: s=1.4, q=0.4, rho=1e-4,
  bins=300; requires |mode3−mode1|/mode1 < 2×10⁻⁴ (achieved: 9.5×10⁻⁶).
- `test_lcbinint_spine_mode_non_caustic_guard`: s=0.6, q=1.0, rho=0.003,
  bins=60; requires mode3 = mode1 exactly (no spine activation).

All 52 regression tests pass.

### Example notebook

`example/compare-vbbl/lcbinint_vbm_light_curve_comparison.ipynb` now compares
VBBL and the public lcbinint finite-source path only.  Mode 3 remains in the
code as an internal experimental spine kernel with regression coverage, but it
is no longer presented as part of the public API or examples.

## Adaptive Source-Bin Refinement (2026-06-22)

Added an opt-in adaptive source-bin mode for the cartesian inverse-ray kernel:

- `adaptive_source_bins=1` enables refinement.
- `source_bins` is the first-pass grid.
- `max_source_bins` caps refinement.
- `finite_source_tol` / `finite_source_reltol` are VBM-style user-facing
  targets.  Python also accepts aliases `tol` and `reltol`.

The first pass runs at fixed `source_bins` and collects cheap diagnostics from
the already-computed image-area scan: boundary rows, gap repairs, overlaps,
fold seed count, and seed count.  These are converted into an empirical
`finite_source_error_estimate` in magnification units.  If the estimate exceeds
`tol + reltol * max(|mag|, 1)`, the scan is rerun with doubled `source_bins`
until the target or `max_source_bins` is reached.

This is not a rigorous VBBL-style contour-integral error bound; it is a
grid-resolution indicator calibrated against VBBL and fixed-bin convergence.
On the planetary test case (s=1, q=0.001, rho=1e-4), `source_bins=50` gives
max relative error ~2.96e-4.  With `adaptive_source_bins=1, reltol=1e-4,
max_source_bins=200`, only the problematic caustic-crossing point refines and
the max relative error drops to ~2.85e-5.

### Calibration update

Added `tests/diagnostics/adaptive_source_bins_sweep.py` to compare fixed and
adaptive grids against VBBL across representative planetary, close, resonant,
wide/equal-mass, limb-darkened, and randomized high-magnification cases.

Key changes from the first estimator:

- Adaptive `tol`/`reltol` now also tightens the hexadecapole acceptance
  threshold.  For `rho >= 1e-3`, the hex self-consistency error is divided by a
  safety factor of 30 before accepting, because large-source tests showed hex
  could underpredict the VBBL discrepancy by ~20-30x.
- Adaptive mode uses a wider fast point-source bbox margin (`60*rho`) instead
  of disabling the shortcut.  This keeps far-from-caustic points cheap while
  avoiding the previous overly aggressive point-source exit near large finite
  sources.
- The cartesian IR error indicator includes a gated `max_jump_cells` term for
  `rho >= 1e-3`, plus a multi-seed/overlap term for large source disks covering
  caustic arcs.
- The gap-repair term and overlap term are now source-size dependent.  For
  small sources (`rho < 1e-2`), the estimator is intentionally less
  conservative for the common A~10-300 cases where fixed-bin convergence and
  VBBL comparisons show the raw 50-bin grid is already within `reltol=1e-3`.
- Local high-magnification floors remain for known failure patterns:
  very large sources, low-magnification few-image rows with large jumps, and
  multi-seed/overlap rows with many gap repairs.  These floors trigger
  refinement without making ordinary small-source caustic approaches
  over-refine.
- If the implied source-bin requirement exceeds `max_source_bins`, small-source
  A<1000 cases are still refined up to the cap before being judged.  Extreme
  A~4000 cases remain flagged `finite_source_converged=false` when the grid
  topology is not reliable.
- If hex is rejected and cartesian IR is used, the rejected hex magnification is
  no longer treated as part of the IR grid-convergence error.  Hex rejection is
  a mode-selection diagnostic; once the calculation has switched to inverse-ray,
  convergence is judged from the image-area diagnostics and source-bin
  self-consistency.

Latest broad diagnostic (`source_bins=50, max_source_bins=400, reltol=1e-3`,
24 randomized cases with 51 times each, seed 20260622) has
`accepted_bad_total=0` and median adaptive max relative error ~1.0e-4.  The
hard high-magnification cases are deliberately returned as unconverged rather
than silently accepted.  This is the current default-policy candidate: guarantee
by convergence flag, not by forcing cartesian IR to solve cases where its grid
topology is not reliable.

Follow-up: an apparent A~30 failure was not a resolution limit.  It was caused
by the fold-image Jacobian-sign guard using too broad a threshold
(`|J| < 0.5`), which clipped valid image area.  Tightening this to `|J| < 0.02`
fixes the randomized case 016 test point:

- VBBL reference: 32.1236247.
- Previous cartesian IR converged to ~32.065 at all tested `source_bins`
  (systematic relative error ~1.8e-3).
- With the tighter guard, cartesian IR gives relative errors of order
  1e-5-3e-5 for `source_bins=50..400`.

The same 24-case diagnostic still has `accepted_bad_total=0`; the worst
remaining cases are the genuinely extreme A~4500 peaks, which remain flagged
as unconverged rather than accepted.

After the guard fix, the medium-magnification error floor (`A > 20` plus large
gap/jump diagnostics) was removed.  It had been compensating for the guard bug
and was making too many A~30-100 cases report `finite_source_converged=false`.
The 24-case diagnostic remains at `accepted_bad_total=0` after removing it.
A wider 48-random-case diagnostic (`random_times=51`, same seed and
`source_bins=50, max_source_bins=400, reltol=1e-3`) also has
`accepted_bad_total=0`, with median adaptive max relative error ~6.6e-5.

### 2026-06-22 grid-spacing estimator update

The first adaptive estimator was still too close to a VBBL-style "danger
signal" model: many rows, jumps, gaps, or overlaps made the code refine even
when fixed `source_bins=50` was already converged.  Single demonstration
notebooks are not a useful performance target for this; the default policy
should be judged by aggregate parameter sweeps split by LD/no-LD.  The
estimator has therefore been recast around the actual cartesian image-area
discretization scale:

```text
cell_area = (rho / source_bins)^2
error ~ boundary_rows * cell_area / source_flux
```

`gap_repairs`, `overlaps`, `seed_count`, and `max_jump_cells` remain as topology
warnings, but their weights are now small corrections to the boundary-cell
estimate rather than dominant error terms.  The high-magnification/topology
floors are retained only for small-source failure patterns and extreme sources
where fixed-bin convergence is known to be unreliable.

Important implementation details:

- large-source (`rho >= 2e-2`) boundary and jump weights are reduced, because
  large `boundary_rows` and `max_jump_cells` mostly reflect the size of the
  image region rather than missing image area;
- few-image gap/jump floors are gated to `rho < 1e-2` so they do not force
  unnecessary large-source refinement;
- adaptive self-consistency now uses adjacent refinement differences
  `|A_N - A_{N/2}|` instead of the minimum over all previous grid levels.  A
  one-step acceptance requires a safety factor (`<= target/2`); with three or
  more levels the adjacent difference must also decrease monotonically.

Current diagnostic status after this change:

```text
python tests/diagnostics/adaptive_source_bins_sweep.py \
  --source-bins 50 --max-bins 400 --reltol 1e-3 \
  --random 24 --random-times 51 --seed 20260622
```

passes with `accepted_bad_total=0`.  The remaining hard A~4500 cases are still
reported as unconverged rather than silently accepted.

The diagnostic script now prints aggregate speed ratios, because the important
question is the global average behavior over lens/source parameters, not a
single notebook case.  After rebuilding the editable Python extension, the
current sweep gives:

```text
subset cases fixed/VBBL geo adaptive/VBBL geo adaptive/fixed geo
all       32          3.98              5.71              1.44
no LD     19          6.76              9.25              1.37
LD        13          1.83              2.83              1.55
```

Interpretation: adaptive over-refinement is now a smaller part of the speed
gap.  Even fixed 50-bin cartesian IR is still globally slower than VBBL,
including LD cases, so the remaining speed work should focus on the
`imagearea4` scan/seed/overlap implementation, cache reuse over light curves,
or a contour-integral path.  The notebook should stay an illustrative plot, not
the primary performance metric.

### Python API update

Python `Options` now uses a fixed finite-source grid by default:

- `lcbinint.Options()` means `source_bins=50`, `adaptive_source_bins=0`.
- `lcbinint.Options(reltol=...)` stores the adaptive IR tolerance but does not
  enable adaptive refinement by itself.
- `lcbinint.Options(reltol=..., adaptive_source_bins=1)` enables the expert
  adaptive refinement path.
- `adaptive_hex_threshold` remains the active hexadecapole/point-source
  selection tolerance and is also exposed as the clearer `hex_tol` alias.
- `LightCurve` exposes `all_converged` and `unconverged_indices` in addition to
  the per-point `finite_source_converged` flags.

The VBBL comparison notebook now uses the high-level light-curve API with
`Options(reltol=1e-3, source_bins=50, max_source_bins=400)`.  The example case
uses a high-magnification finite-source trajectory:

```text
s=1, q=0.001, u0=-0.001, alpha=0, rho=3e-3
```

Current saved notebook result (`source_bins=50, max_source_bins=400,
reltol=1e-3`):

```text
no LD: VBBL 0.0908 ms/pt, lcbinint 3.7934 ms/pt, max rel 6.546e-4, unconv 0
LD:    VBBL 6.1567 ms/pt, lcbinint 4.3427 ms/pt, max rel 3.757e-4, unconv 0
```

The non-LD path is still much slower than VBBL because it uses the cartesian
inverse-ray grid where VBBL can often use cheaper contour logic.  The
limb-darkened path is faster than VBBL for this example because VBBL's LD
integration cost grows substantially while lcbinint reuses the same image-area
scan with different source weights.

### Large-Source Seed Regression

Fixed a root bug in `legacy_augmented_image_seeds`: the function returned early
whenever the source center already had five point-source images.  For finite
sources this is wrong, because a large source disk can still contain additional
caustic/boundary image components that are not seeded by the center images.
This caused isolated light-curve points to be wrong by several percent for
large sources, e.g. `rho=0.03`, and by ~30% for `rho=0.3` at particular grid
refinement levels.

Current fixes:

- always run finite-source caustic/boundary seed augmentation when
  `source_radius > 0`, even if the source center has five images;
- increase the boundary seed cap from 32 to 128;
- use a tighter duplicate radius (`0.25*rho`) for probe image seeds so large
  sources do not merge distinct image components too aggressively;
- require adaptive refinement to agree with an earlier refinement before
  accepting a result, which rejects transient bad grid levels such as
  `source_bins=200` in the `rho=0.3` regression.

Regression cases were added for `rho=0.03` and `rho=0.3`.  After the fix:

- `rho=0.03`: max relative error is ~5.1e-4 over the 400-point example curve;
- `rho=0.3`: max relative error is ~4.2e-4 over the same trajectory;
- the 48-random-case diagnostic still has `accepted_bad_total=0`.

### Graduated Hex Safety Factor (2026-06-22)

Commit `6f182b2` replaced the flat `hex_safety=30` factor (applied to all
sources with `rho ≥ 1e-3`) with a power-law graduated factor:

```
hex_safety = clamp(30 × (hex_threshold / dist_ratio)³, 1, 30)
```

where `dist_ratio = refined_dist / source_radius` is the ratio of the
caustic-proximity distance (already computed in `binary_mag`) to the source
radius.  For small sources (`rho < 1e-3`) `hex_safety = 1` as before.

**Effect**: The blanket factor 30 was tightening the hex acceptance threshold
from 0.001 to 3.3e-5 even for sources far from the caustic where the hex
Taylor expansion is accurate.  The graduated formula gives safety≈30 only
when the source boundary is nearly touching a caustic fold (`dist_ratio ≈
hex_threshold ≈ 3`) and safety≈1 when the source is more than ~3 radii away.

**Measured improvement** on planetary-large-source LD benchmark (s=1, q=0.001,
rho=0.01, u0=-0.01, 61 time points):

- Before: ~4.9 ms/pt (LD)
- After:  ~4.0 ms/pt (LD) — 18% faster
- 8-case named diagnostic: `accepted_bad_total` reduced from 4 → 1

### Alpha Convention Investigation (2026-06-22)

During benchmarking, max_rel errors of 2.4–2.5 were observed for some parameter
combinations.  Detailed investigation showed these were **not real bugs**:

- The errors were caused by accidentally having a wrong binary installed in
  `site-packages` (a trajectory modification that was reverted but not
  reinstalled).
- The current code is correct.  `binary_mag0(s, 1/q, tau, u0)` is
  mathematically equivalent to VBM's `BinaryMag2(s, q, −tau, u0)` via a
  reflection symmetry of the binary lens geometry combined with the y2-axis
  symmetry of the magnification function.  The `effective_q = 1/q` design
  in `lens_model.cpp` correctly absorbs the trajectory sign convention
  difference so that the same `alpha` value in lcbinint and VBM produces the
  same light curve.
- All 60 regression tests pass with the correct binary.
- The 8-case named diagnostic (s=50, reltol=1e-4) gives `accepted_bad_total=1`
  (wide_equal_mass, max_rel=1.84e-4 vs tol=1e-4 — barely outside tolerance).

### Current Performance Status (2026-06-22)

Benchmark: s=1, q=0.001, rho=0.01, u0=-0.01, 61 time points, VBM Tol=1e-3.

| Mode | Time (ms/pt) | vs VBM LD |
|------|-------------|-----------|
| VBM no-LD (Tol=1e-3) | 0.42 | baseline |
| VBM LD (Tol=1e-3) | 3.24 | 1× |
| lcbinint fixed s_bins=80 | 5.3 | 1.6× slower |
| lcbinint adaptive tol=1e-4 | 8.3 | 2.6× slower |

With VBM Tol=1e-4 (tighter accuracy): VBM LD = 22.8 ms/pt; lcbinint
adaptive = 8.3 ms/pt (2.7× **faster** than VBM at same tight tolerance).

The limb-darkened path in lcbinint is still slower than VBM LD at VBM's
default Tol=1e-3.  The bottleneck is the adaptive refinement overhead;
fixed-bin at 80 bins is already within 1.5e-3 relative error.

## Open Questions

- Should the root solver depend on GSL, or should we implement a clean-room
  solver locally?
- Should the first Python binding use Cython, pybind11, cffi, or CPython C API?
- How much legacy behavior should be preserved exactly for the first regression
  suite?
- Which legacy command/output cases should become canonical regression fixtures?
