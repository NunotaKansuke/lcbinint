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
- calibrated one-shot finite-source resolution and Cartesian/polar selection

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

For binary finite-source inverse-ray calculations, `nbin="auto"` selects the
Cartesian/polar grid and resolution once per source position from calibrated
geometry diagnostics.  It does not run a trial integration or retry at a higher
resolution.  The calibration experiment and frozen artifacts are documented in
[`docs/finite-source-auto-calibration.md`](docs/finite-source-auto-calibration.md).

`LightCurve.info(...)` reports the selected method, convergence state, and
numerical error diagnostics. lcbinint never imports or dispatches to an external
solver. Numerical conventions and limits are summarized in
[`docs/numerical-methods.md`](docs/numerical-methods.md).

Annual parallax requires `parallax=True`, `t_ref`, and a sky position supplied
on the callable or in the per-call parameters.
Terrestrial parallax additionally requires `terrestrial=True` and an explicit
`lcbinint.obs.Site`; merely passing non-zero `piEN`/`piEE` does not activate
parallax.

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
