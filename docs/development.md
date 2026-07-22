# Developer guide

## Repository map

| Path | Responsibility |
| --- | --- |
| `include/lcbinint/lcbinint.h` | Public C API: parameter/options/result structures and scalar/array calls. |
| `src/lcbinint/magnification/` | Point- and finite-source kernels plus polynomial-root machinery. |
| `src/lcbinint/model/` | Lens parameters, lens systems, trajectories, orbital motion, and triple-lens geometry. |
| `src/lcbinint/lc/` | Reusable light-curve orchestration and inference-oriented batch paths. |
| `src/lcbinint/obs/` | Sky/site coordinates and parallax support. |
| `python/` | pybind11 bindings and the small Python convenience layer. |
| `tests/unit/` | Native C++ unit test. |
| `tests/regression/` | Python behavior and compatibility regression tests. |
| `tests/diagnostics/` | Reproducible sweeps, timing probes, calibration scripts, and frozen results. |
| `example/` | Runnable user-facing examples. |

The dependency direction is intentional: `magnification/` is a leaf layer;
`model/`, `obs/`, and `lc/` build on it. Avoid making a numerical kernel depend
on higher-level trajectory or Python code.

## Build

Requirements are CMake 3.16+, a C++17 compiler, GSL headers/libraries, Python
3.9+, NumPy, and pybind11 for the Python extension. Scikit-build-core installs
the Python package:

```sh
python -m pip install -e ".[test]"
```

For an explicit CMake build:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
PYTHONPATH=build python -m pytest -q
```

Set `GSL_ROOT=/path/to/prefix` before configuring if CMake cannot find GSL.
The project enables OpenMP for independent batch rows when the compiler and
runtime provide it. Disable it reproducibly with
`-DLCBININT_ENABLE_OPENMP=OFF`.

Useful CMake switches:

| Switch | Default | Effect |
| --- | --- | --- |
| `LCBININT_BUILD_PYTHON` | `ON` | Build the pybind11 extension. |
| `LCBININT_ENABLE_OPENMP` | `ON` | Use OpenMP for independent parameter rows when available. |
| `LCBININT_BUILD_TESTS` | follows `BUILD_TESTING` | Build the native unit executable. |

## Test strategy

Run the native unit test and Python regression suite before handing off a
change. Tests automatically exclude the diagnostic directory, because those
programs are experiments rather than pass/fail regression cases.

```sh
ctest --test-dir build --output-on-failure
PYTHONPATH=build python -m pytest -q
```

When changing finite-source switching, inverse-ray resolution, a root solver,
or coordinate conventions, also run focused diagnostics and preserve the
command/output needed to reproduce conclusions. Existing entry points include:

```sh
PYTHONPATH=build python tests/diagnostics/polar_cartesian_mode_sweep.py --random 10 --points-per-case 4
PYTHONPATH=build python tests/diagnostics/finite_source_safety_sweep.py
```

VBMicrolensing is an optional comparison dependency used by a subset of those
tests and notebooks. `lcbinint` does not import, link, or dispatch to it at
runtime.

## API-change checklist

The Python API is defined in both `python/bind_lc.cpp` and
`python/lcbinint/__init__.py`; keep the native binding, convenience wrapper,
and [Python API reference](python-api.md) consistent. Changes to the C API
also require reviewing `include/lcbinint/lcbinint.h` for ABI and documented
default-value implications.

For a user-visible change:

1. Add or update a regression test.
2. Update a runnable example if the normal workflow changes.
3. Update the user guide/API reference and any relevant numerical-method note.
4. State coordinate conventions, units, error semantics, and unsupported
   combinations rather than leaving them implicit.

## Numerical changes

Do not describe a finite-source result as accurate merely because it was
computed. Respect `finite_source_converged` and document the requested error
budget. The calibrated automatic-resolution artifacts live under
`tests/diagnostics/results/finite-source-auto-20260716/`; their scope and
limits are described in [finite-source-auto-calibration.md](finite-source-auto-calibration.md).

The diagnostic scripts are part of the evidence trail. If a calibration rule
or a frozen artifact changes, record the data set, command, tolerances, and
comparison engine/version in the accompanying result README.

## C API notes

The C header exposes `lcbi_default_params()` and `lcbi_default_options()`;
always initialize structures through those functions rather than assuming zero
is a valid default. Use `lcbi_magnification_array()` for many epochs and
`lcbi_finite_source_geometry_array()` when an external finite-source engine
needs lcbinint's transformed geometry without root solving. Check every return
value against `LCBI_OK` and obtain a printable status with
`lcbi_status_string()`.

The C API has no Python dependency. The installed CMake targets are
`lcbinint_magnification` and `lcbinint_lightcurve`; the latter depends on the
former. Treat `third_party/skowron_gould/` as third-party code and retain its
license/notice files when distributing it.
