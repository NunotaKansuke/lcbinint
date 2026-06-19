# lcbinint

Public Python package and C++ library for microlensing magnification
calculations derived from the existing `lcbinint` code.

The goal is not to wrap the legacy executable. The goal is to extract the
magnification engine into a publishable C++ core, replace Numerical
Recipes-derived support routines, and expose a small Python API.

Initial scope:

- point-source binary lens magnification
- point-source triple lens magnification
- a stable parameter model that can later cover finite-source modes
- a C-compatible ABI layer over the C++ implementation

Later scope:

- finite-source integration
- inverse ray-shooting modes
- limb darkening
- fitting utilities

See [docs/migration-plan.md](docs/migration-plan.md).

## Build

This project follows the same GSL discovery approach as `../genulens`.

```sh
GSL_ROOT=/rogue1_8/nunota/local/gsl make test
```

If GSL is installed elsewhere, set `GSL_ROOT` to that prefix.
