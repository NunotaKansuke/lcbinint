# lcbinint

`lcbinint` is a Python package with a C++ core for binary-lens microlensing
magnification and light-curve calculations.

The original `lcbinint` code was developed by Takahiro Sumi.  This package is a
modernized Python/C++ implementation. Its API design, validation strategy, and
performance-oriented development are strongly informed by
[VBMicrolensing](https://github.com/valboz/VBMicrolensing/tree/main).

This is an early developer release.  The Python API is intended to provide
lightweight, reusable callables for repeated model evaluation.

Features:

- point-source and finite-source binary-lens magnification
- inverse-ray finite-source integration with Cartesian and polar grids
- linear limb darkening
- reusable light-curve callables
- annual and terrestrial parallax
- circular and Keplerian lens orbital motion
- calibrated finite-source resolution with error-guided Cartesian correction

## Developer Install

Requirements:

- C++17 compiler
- CMake >= 3.16
- Python >= 3.9
- GSL development headers/libraries
- `pybind11`, `numpy`, `scikit-build-core`

Set `GSL_ROOT` if GSL is not installed system-wide.

```sh
git clone https://github.com/NunotaKansuke/lcbinint.git
cd lcbinint

python -m pip install -U pip
python -m pip install -e ".[test]"
```

If GSL is installed in a custom prefix:

```sh
GSL_ROOT=/path/to/gsl python -m pip install -e ".[test]"
```

You can also build directly with CMake:

```sh
GSL_ROOT=/path/to/gsl cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Run the Python regression tests against the in-tree build:

```sh
PYTHONPATH=build python -m pytest -q
```

`VBMicrolensing` is used by several comparison tests and diagnostics.

## Quick Use

```python
import numpy as np
import lcbinint

times = np.linspace(-0.5, 0.5, 200)
params = {
    "t0": 0.0,
    "tE": 1.0,
    "u0": -0.01,
    "alpha": 0.5,
    "s": 0.95,
    "q": 1.0e-2,
    "rho": 5.0e-3,
}

lc = lcbinint.LightCurve(
    options=lcbinint.Options(
        coordinates="vbm",
        nbin="auto",  # default; pass a positive integer for a fixed grid
    ),
    limb_darkening=lcbinint.LimbDarkening.linear(0.5),
)

mag = lc(times, params)
```

Physical model choices are separate from numerical options:

```python
model = lcbinint.Model(
    parallax=True,
    terrestrial=True,
    orbital_motion="kepler",
    sky=lcbinint.obs.SkyCoord(270.0, -30.0),
    t_ref=2459000.0,
)
lc = lcbinint.LightCurve(
    model=model,
    site=lcbinint.obs.Site("ground", -29.0, 70.7),
    options=lcbinint.Options(nbin="auto"),
)
```

For joint ground/space fits, pass that same `model` to each curve and give
each curve its own site. `terrestrial=True` affects only the ground curve:

```python
ground = lcbinint.LightCurve(model=model, site=lcbinint.obs.Site("ground", -29.0, 70.7))
space = lcbinint.LightCurve(model=model, site=lcbinint.obs.Site("space", satellite_table))
```

`Model` selects physical terms; `Options` controls numerical evaluation.

For a direct finite-source binary-lens evaluation in lens-plane coordinates,
use the low-level API:

```python
amplification = lcbinint.binary_ray_shooting(
    x, y, s=1.2, q=0.05, rho=0.01,
    options=lcbinint.Options(inverse_ray_grid="polar"),
)
```

It corresponds to VBMicrolensing's `BinaryMag2`: `x` and `y` are
center-of-mass source coordinates and `rho` must be positive.

For binary finite-source inverse-ray calculations, `nbin="auto"` selects the
Cartesian/polar grid and an initial resolution per source position from
calibrated geometry diagnostics. If the Cartesian area-error estimate misses
the requested budget, only that position advances to the smallest suitable
grid bucket, up to `max_source_bins`. Fixed integer `nbin` never retries. The
calibration experiment and frozen artifacts are documented in
[`docs/finite-source-auto-calibration.md`](docs/finite-source-auto-calibration.md).

`LightCurve.info(...)` reports the selected method, convergence state, and
numerical error diagnostics. lcbinint never imports or dispatches to an external
solver. Numerical conventions and limits are summarized in
[`docs/numerical-methods.md`](docs/numerical-methods.md).

Inference engines can evaluate independent parameter rows without changing
the scalar API:

```python
rows = [params_a, params_b, params_c]  # dict or lcbinint.Parameters
magnifications = lc.magnification_batch(times, rows)
# shape: (len(rows), len(times))
```

For single- and binary-source models this is one GIL-free native call and
writes directly to a row-major output matrix.

The moasarc adapter also uses lcbinint's internal fused likelihood entry point.
It streams one reusable epoch row through magnification, source/blend flux
solving, and Gaussian or Student-t likelihood evaluation, avoiding the full
walker-by-epoch matrix while leaving the scalar LightCurve API unchanged.
The fused path supports binary sources as well as Gaussian/Student-t and
fit/sample/marginalized flux modes.

Annual parallax requires `Model(parallax=True, ...)`, `t_ref`, and a sky
position supplied on the model or in the per-call parameters.
Terrestrial parallax additionally requires `terrestrial=True` and an explicit
ground `lcbinint.obs.Site("ground", lat, lon)`; merely passing non-zero `piEN`/`piEE`
does not activate parallax. For a space observatory, pass `Site("space", table)`, with table columns
`(JD, RA_deg, Dec_deg, distance_AU)`.

## Diagnostics

Optional diagnostic checks:

```sh
PYTHONPATH=build python tests/diagnostics/polar_cartesian_mode_sweep.py --random 10 --points-per-case 4
```

An executed VBM comparison notebook is included at
`example/compare-vbm/lcbinint_vbm_light_curve_comparison.ipynb`.

Runnable examples are grouped by purpose:

- `example/light-curve/`: reusable callables, diagnostics, and parallax
- `example/compare-vbm/`: binary/triple accuracy and timing comparisons
- `example/image-plane/`: caustics, critical curves, and image positions
- `example/kepler-lom/`: Kepler orbital-motion reference epochs

API details may change before the first stable package release.
