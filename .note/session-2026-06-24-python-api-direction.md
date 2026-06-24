# 2026-06-24 Python API direction

## Goal

The public Python API should be simple and VBM-like at the call site, but less
annoying for repeated modeling work.  The main public path is a lightweight
callable built once with slow-changing configuration, then called many times
with named fitted parameters.  Helper objects are acceptable only for
configuration that is not naturally part of the model parameter vector.

`LensModel` is no longer part of the Python public API.  The C++ internal model
object still exists for non-static-binary fallback paths, but Python users
should reach the backend through `LightCurve(...)`
for normal use.
Low-level functions remain available for tests, diagnostics, and one-off
experiments, but should not be the primary documented workflow.

## Recommended shape

Use `LightCurve(...)` as the main user-facing entry point.  It constructs a
reusable callable/evaluator with slow-changing configuration fixed at
construction time.  Older builder-style factory names were removed before
release; `LightCurve(...)` is the single public construction path.  `buildLC`
was considered, but it is too cryptic for a public API.

```python
lightcurve = lcbinint.LightCurve(
    lens="binary_lens",
    event=lcbinint.EventCoordinates(ra=267.6, dec=-29.1, tfix=2459000.0),
    limb_darkening=lcbinint.LimbDarkening.linear(0.5),
    options=lcbinint.Options(reltol=1.0e-3),
    orbital_motion_mode=lcbinint.OrbitalMotionMode.STATIC,
)

curve = lightcurve(
    times,
    t0=0.0,
    tE=1.0,
    u0=-0.01,
    alpha=0.5,
    s=1.0,
    q=1.0e-3,
    rho=1.0e-3,
)
```

Use `LightCurve(parallax=True)` for annual-parallax calls:

```python
lightcurve = lcbinint.LightCurve(
    lens="binary_lens",
    event=lcbinint.EventCoordinates(ra=267.6, dec=-29.1, tfix=2459000.0),
    options=lcbinint.Options(reltol=1.0e-3),
    limb_darkening=lcbinint.LimbDarkening.linear(0.5),
    orbital_motion_mode=lcbinint.OrbitalMotionMode.CIRCULAR,
    parallax=True,
)

curve = lightcurve(
    times,
    t0=2459001.0,
    tE=80.0,
    u0=0.12,
    alpha=0.4,
    s=1.1,
    q=1.0e-3,
    rho=1.0e-3,
    piEN=0.02,
    piEE=0.03,
    g1=0.001,
    g2=0.002,
    g3=0.0,
)
```

This is intentionally not a model object that owns fitted physical parameters.
It only closes over slow-changing configuration: lens family, event
coordinates, numerical options, limb darkening, orbital-motion model, and
whether annual parallax is enabled.  Fitted parameters remain call arguments.
Limb-darkening coefficients are fixed in the callable by default, because they
are usually observational setup/model-choice parameters rather than coordinates
that should be repeated on every evaluation.

`LightCurve(parallax=False)` uses the static source-trajectory overload.
`LightCurve(parallax=True)` enables the annual-parallax source-trajectory
overload, which accepts `piEN` and `piEE`.  The selected trajectory model is
fixed at object construction time; ordinary light-curve calls only dispatch to
the corresponding C++ kernel.

The result object returned by `info(...)` is named `LightCurveInfo`, so the
public names separate the evaluator from diagnostic results.

The parameter names are intentionally `s`, `q`, `u0`, and `alpha`, rather than
the older `sep`, `umin`, and `theta`, because this is closer to ordinary
microlensing notation and avoids VBM's positional-argument confusion.

## Helper objects

`Options` remains the place for numerical settings:

- `source_bins`
- `max_source_bins`
- `tol` / `reltol`
- finite-source mode and expert controls

`LimbDarkening` currently stores the existing two coefficients:

- `LimbDarkening.none()`
- `LimbDarkening.linear(c)`
- `LimbDarkening.quadratic(c, d)`

Longer term, this should become a C++ kernel family rather than a Python
callback in the inner loop.  A Python callback would be convenient but would
destroy the main inverse-ray performance advantage.

`EventCoordinates` stores `ra`, `dec`, and `tfix`.  These are not hot fitted
parameters and should not clutter the main function signature more than needed.

## Current implementation state

- `LightCurve` is the recommended public entry point for normal modeling.
- `light_curve` and `magnification` remain low-level convenience functions for
  tests, diagnostics, and simple one-off checks.  They should not be the main
  public workflow because they make callers repeat configuration such as
  numerical options and limb darkening.
- `binary_light_curve` and `binary_magnification` remain compatibility wrappers
  for now.
- The builder creates a lightweight callable with fixed slow-changing
  configuration.  `lens="binary_lens"` is the
  only implemented lens family for now; the builder shape is meant to allow a
  future `lens="triple_lens"` implementation without changing the call site.
- `LightCurve` stores fixed limb-darkening coefficients and does not accept
  `piEN` / `piEE`.
- `LightCurve(parallax=True)` stores the same fixed limb-darkening settings and
  accepts `piEN` / `piEE` as fitted parameters.
- The callable precomputes fixed C++ configuration at build time.  Static
  binary calls use a direct binary kernel.  Parallax and orbital-motion calls
  also avoid constructing a Python-visible or fallback `LensModel` per
  evaluation; they evaluate the source trajectory/orbital state and call the
  point/finite-source magnifiers directly.
- Annual-parallax projection now uses a thread-local projector cache keyed by
  event coordinates and reference time, avoiding per-time projector rebuilds
  along a light curve.
- `light_curve` returns a NumPy array and should be preferred when the caller
  already has NumPy times.
- `light_curve_info` returns the same magnification plus diagnostic metadata
  such as finite-source method, source position, image count, error estimate,
  refinement level, and convergence flag.
- Static binary, VBM-compatible calls take a direct C++ path instead of forcing
  construction of a Python-visible model object.
- The direct finite-source Python path keeps a thread-local
  `FiniteSourceMagnifier` cache keyed by numerical settings and limb darkening.
  This avoids rebuilding the caustic/magnifier cache for repeated single-point
  calls with the same options.

## Geometry helpers

The reusable callable now also exposes plotting/diagnostic helpers.  These are
not used by the hot light-curve evaluation path, so they do not add overhead to
ordinary light-curve calls.

```python
lightcurve = lcbinint.LightCurve(options=lcbinint.Options(coordinates="vbm"))

trajectory = lightcurve.source_trajectory(
    times,
    t0=0.0,
    tE=1.0,
    u0=-0.01,
    alpha=0.5,
    s=1.0,
    q=1.0e-3,
)
caustics = lightcurve.caustics(s=1.0, q=1.0e-3, n_points=900)
critical = lightcurve.critical_curves(s=1.0, q=1.0e-3, n_points=900)
```

`source_trajectory(...)` returns a `GeometryCurve` with `times`, `x`, and `y`.
The coordinates match the source coordinates reported by `info(...)`, including
the VBM-compatible internal lens frame.  `caustics(...)` and
`critical_curves(...)` return `GeometryBranches` with branch-wise `x` and `y`
lists.  For orbital motion, pass `time=...` plus the same LOM parameters to get
the instantaneous caustic/critical curve at that epoch.

Parallax-aware callables expose the same helpers; their
`source_trajectory(...)` signature also accepts `piEN` and `piEE`.  Parallax
does not change the lens caustic itself, while orbital motion can change the
instantaneous separation.

## Next cleanup

1. Stop using `LensParams` in user-facing examples.  It is still bound for
   compatibility and diagnostics, but new documentation should use
   `LightCurve(...)`, `Options`, `LimbDarkening`, and `EventCoordinates`.
2. Keep the internal C++ `LensParameters` / `LensModel` types for now.  They
   still support the C ABI and non-direct fallback paths.  Removing them safely
   requires routing those paths through the direct callable kernels first.
3. Add C++ limb-darkening kernel variants before advertising arbitrary
   limb-darkening profiles.

## Example state

`example/compare-vbm/` has been rewritten around the latest public API:

- `quickstart_compare_vbm.py` is the runnable CLI/script version.
- `lcbinint_vbm_light_curve_comparison.ipynb` is the compact notebook version.
- both examples use `LightCurve(...)`, `lightcurve(...)`,
  `source_trajectory(...)`, and `caustics(...)`;
- neither example teaches `LensParams` or model-object construction.
