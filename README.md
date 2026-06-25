# lcbinint

`lcbinint` is a Python package and C++ core for microlensing magnification
calculations derived from the original `lcbinint` code.

This is an early developer release.  The goal is a publishable C++/Python
implementation, not a wrapper around the legacy executable.

Current focus:

- binary-lens point-source magnification
- binary-lens finite-source magnification with inverse-ray integration
- linear limb darkening
- light-curve callables for repeated evaluation
- annual parallax and lens orbital motion support
- GSL-based replacements for Numerical Recipes routines
- Skowron-Gould style polynomial root solving

The default finite-source path uses `inverse_ray_grid="auto"`, which keeps the
Cartesian grid for ordinary inverse-ray evaluations and switches to a polar grid
for high-magnification cases where Cartesian aliasing is risky.

## Developer Install

Requirements:

- C++17 compiler
- CMake >= 3.16
- Python >= 3.9
- GSL development headers/libraries
- `pybind11`, `numpy`, `scikit-build-core`

GSL is discovered with the same local-prefix convention used by `genulens`.
Set `GSL_ROOT` if GSL is not installed system-wide.

```sh
git clone https://github.com/NunotaKansuke/lcbinint.git
cd lcbinint

python -m pip install -U pip
python -m pip install -e ".[test]"
```

For this machine, the usual explicit GSL prefix is:

```sh
GSL_ROOT=/rogue1_8/nunota/local/gsl python -m pip install -e ".[test]"
```

You can also build directly with CMake:

```sh
GSL_ROOT=/rogue1_8/nunota/local/gsl cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
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

lc = lcbinint.LightCurve(
    options=lcbinint.Options(
        coordinates="vbm",
        source_bins=50,
        inverse_ray_grid="auto",  # default
    ),
    limb_darkening=lcbinint.LimbDarkening.linear(0.5),
)

mag = lc(
    times,
    t0=0.0,
    tE=1.0,
    u0=-0.01,
    alpha=0.5,
    s=0.95,
    q=1.0e-2,
    rho=5.0e-3,
)
```

For debugging the inverse-ray grid choice:

```python
lcbinint.Options(inverse_ray_grid="auto")       # recommended
lcbinint.Options(inverse_ray_grid="cartesian")  # expert/debug
lcbinint.Options(inverse_ray_grid="polar")      # expert/debug
```

## Diagnostics

Useful local checks:

```sh
PYTHONPATH=build python tests/diagnostics/polar_cartesian_mode_sweep.py --random 10 --points-per-case 4
PYTHONPATH=build python example/compare-vbm/quickstart_compare_vbm.py
```

The repository is still moving quickly, so API details may change before a
proper package release.
