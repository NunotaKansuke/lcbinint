# User guide

`lcbinint` evaluates microlensing magnification and light curves with a C++17
core and a small NumPy-oriented Python API. Construct a `LightCurve` once,
then reuse it while an inference sampler changes the lens parameters.

This guide describes the current development API. It is deliberately explicit
about physical-model choices and numerical choices: they are independent.

## Install

The Python package needs Python 3.9 or newer, a C++17 compiler, CMake 3.16 or
newer, and the GSL development library. From a source checkout:

```sh
python -m pip install -U pip
python -m pip install -e ".[test]"
```

If GSL is outside the system search path, point the build at its prefix:

```sh
GSL_ROOT=/opt/gsl python -m pip install -e ".[test]"
```

`.[test]` installs `pytest` and VBMicrolensing. The latter is only needed by
the optional comparison tests and diagnostics, not by the `lcbinint` runtime.
For a native-only build, see the [developer guide](development.md).

## First light curve

Times are one-dimensional NumPy-compatible arrays. The result is a
one-dimensional `float64` NumPy array with one magnification per time.

```python
import numpy as np
import lcbinint

times = np.linspace(-0.5, 0.5, 200)
params = {
    "t0": 0.0, "tE": 1.0, "u0": -0.01, "alpha": 0.5,
    "s": 0.95, "q": 1.0e-2, "rho": 5.0e-3,
}

curve = lcbinint.LightCurve(
    options=lcbinint.Options(nbin="auto"),
    limb_darkening=lcbinint.LimbDarkening.linear(0.5),
)
magnification = curve(times, params)
```

The default coordinate convention is `"vbm"`, matching the usual
VBMicrolensing-style input convention. `alpha` is in radians; all times use
the same unit as `tE` (normally days); `rho` is in Einstein-radius units.

Pass a mapping, a `lcbinint.Parameters` instance, or keyword arguments to a
single-source curve. The friendly names `u0`, `alpha`, and `s` have the
equivalent native aliases `umin`, `theta`, and `sep`.

## Separate physics from numerics

Use `Model` for *what physical effects are present* and `Options` for *how
the magnification is evaluated*. This makes sharing a physical event model
between observatories safe and clear.

```python
model = lcbinint.Model(
    parallax=True,
    terrestrial=True,
    orbital_motion="kepler",
    sky=lcbinint.obs.SkyCoord(270.0, -30.0),
    t_ref=2459000.0,
)
options = lcbinint.Options(
    coordinates="vbm",
    nbin="auto",
    inverse_ray_grid="auto",
)
ground = lcbinint.LightCurve(
    model=model,
    options=options,
    site=lcbinint.obs.Site("ground", -29.0, 70.7),
)
```

`Model` supports binary or triple lenses, single or binary sources, annual and
terrestrial parallax, circular or Keplerian lens orbital motion, and several
xallarap parameterizations. Its full accepted values and corresponding
per-call parameters are in the [API reference](python-api.md).

## Finite-source accuracy and diagnostics

For binary finite sources, `nbin="auto"` is the recommended default. It uses
calibrated geometry rules to choose Cartesian or polar inverse-ray integration
and its initial resolution. A Cartesian calculation may retry at a larger
resolution when its independent area-error estimate exceeds the requested
budget. A fixed positive `nbin` is exactly that: one resolution with no retry.

Request a budget with an absolute tolerance, a relative tolerance, or both:

```python
accurate = lcbinint.LightCurve(
    options=lcbinint.Options(nbin="auto", tol=1e-5, reltol=1e-4)
)
info = accurate.info(times, params)

assert info.all_converged
print(info.finite_source_method_names)
print(info.finite_source_error_estimates)
```

Treat `all_converged=False` or a false entry in `finite_source_converged` as a
failed requested accuracy budget. The returned magnification is still useful
for diagnosis, but it must not be represented as meeting that tolerance.
`LightCurve.info()` currently supports single-source models only.

`info` also exposes source coordinates, image counts, time-evolved separation
and mass ratio, point-source safety indicators, root-solver diagnostics, and
caustic distances. See [Numerical methods](numerical-methods.md) for the
meaning of automatic resolution and the error budget.

## Parallax and observatories

Annual parallax requires all of the following:

1. `Model(parallax=True, sky=..., t_ref=...)`;
2. non-zero `piEN` and/or `piEE` in the per-evaluation parameters; and
3. a consistent Julian-date time system.

Terrestrial parallax additionally requires `terrestrial=True` and a ground
site. Latitude is north-positive and longitude is east-positive, in degrees.
A site alone does not enable a correction.

```python
sky = lcbinint.obs.SkyCoord("18:00:00", "-30:00:00", unit="hours")
model = lcbinint.Model(parallax=True, terrestrial=True, sky=sky, t_ref=2459000.0)
site = lcbinint.obs.Site("ground", -29.0, 70.7)
curve = lcbinint.LightCurve(model=model, site=site)
```

For a satellite, supply an `N × 4` strictly time-increasing array with columns
`JD, RA_deg, Dec_deg, distance_AU`:

```python
space_site = lcbinint.obs.Site("space", satellite_table)
space_curve = lcbinint.LightCurve(model=model, site=space_site)
```

The same `Model` can be shared by ground and space curves. With
`terrestrial=True`, the terrestrial term is applied to ground curves only.

## Batch evaluation and binary sources

For independent parameter rows (for example walkers), use one native batch
call instead of a Python loop:

```python
rows = [params_a, params_b, params_c]
matrix = curve.magnification_batch(times, rows)
# matrix.shape == (len(rows), len(times))
```

The native evaluation releases the Python GIL. Builds with OpenMP can also
parallelize independent rows.

For `Model(source="binary")`, call with a dictionary/keywords and include
`q_source` (or `fluxratio`) plus `t0_2` and `u0_2`. Alternatively, set a
positive `q_mass` to use the coupled-xallarap secondary-source construction.
`Parameters` alone cannot carry these binary-source-only fields.

## Inspect trajectories and geometry

The reusable curve exposes the transformed source trajectory without solving
lens roots:

```python
trajectory = curve.source_trajectory(times, params)
geometry = curve.finite_source_geometry(times, params)
caustics = curve.caustics(params)
critical = curve.critical_curves(params)
```

`trajectory.x` and `.y` are lens-frame positions. `caustics` and `critical`
are branch collections with parallel `x` and `y` lists. When lens orbital
motion is enabled, `separation`, `caustics`, and `critical_curves` require the
epoch as their first argument, for example `curve.caustics(time, params)`.

For binary-lens plotting or individual image locations, use `ImagePlane`:

```python
plane = lcbinint.ImagePlane(q=0.01, s=0.95, x=0.02, y=-0.01, rho=0.005)
image_positions = plane.images()
ax = plane.plot(legend=True)
```

`ImagePlane.plot()` imports Matplotlib only when called. Its finite-image
points are a visualization-oriented inverse-ray sample, not a replacement for
the production magnification calculation.

## Next steps

The runnable examples are a useful complement to this guide:

- [`example/light-curve/`](../example/light-curve/) for model and diagnostics;
- [`example/image-plane/`](../example/image-plane/) for geometry plots;
- [`example/compare-vbm/`](../example/compare-vbm/) for validation comparisons;
- [`example/kepler-lom/`](../example/kepler-lom/) for Kepler reference epochs.
