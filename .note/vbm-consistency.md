# VBM Consistency Tests

## Purpose

Use VBBinaryLensing/VBMicrolensing as an external numerical reference while the
legacy `lcbinint` magnification path is moved into the new C++ core.

This is separate from legacy-executable regression tests:

- VBM comparison checks whether the new public calculation agrees with an
  established microlensing implementation.
- Legacy regression checks whether behavior has changed relative to the old
  `lcbinint.c` implementation.

Both are useful. VBM comparison is especially important for the new
Skowron-Gould polynomial root solver path.

## Local Reference API

The current environment has both Python packages installed:

```python
from VBBinaryLensing import VBBinaryLensing

vb = VBBinaryLensing()
mag = vb.BinaryMag0(separation, mass_ratio, y1, y2)
```

`../MulensModel/source/VBBL/README.md` and
`../MulensModel/source/VBBL/VBBinaryLensingLibrary.h` state that VBBinaryLensing
uses the Skowron & Gould complex-root algorithm.

## Test Shape Added

`tests/regression/test_vbm_consistency.py` defines binary point-source cases
away from exact caustic singularities and hard-codes the current VBM reference
values.

For binary point-source magnification there are two tests:

- `test_vbm_binary_reference_values_are_stable`: normal pytest test; confirms
  VBM is available and returns the pinned reference values.
- `test_lcbinint_binary_point_source_matches_vbm`: compares
  `lcbinint.binary_mag0(...)` directly against VBM.

The expected future convenience function is:

```python
lcbinint.binary_mag0(separation, mass_ratio, y1, y2)
```

That low-level point-source API is useful for direct solver validation. The
higher-level `LensModel` API should be tested separately after trajectory and
finite-source behavior are implemented.

## Current Reference Cases

| separation | q | y1 | y2 | VBM BinaryMag0 |
| --- | --- | --- | --- | --- |
| 1.0 | 0.1 | 0.2 | 0.1 | 5.871444912771214 |
| 0.7 | 0.3 | -0.4 | 0.2 | 2.116643550532278 |
| 1.5 | 1.0 | 0.05 | -0.2 | 1.5493462433112466 |
| 1.0 | 0.001 | 0.3 | 0.4 | 2.1789388609029046 |

Initial comparison tolerance:

```text
rtol = 1e-10
atol = 1e-11
```

This may need loosening around caustics or when comparing finite-source
integration, but it should be reachable for point-source cases away from
singularities.

## Near-Term Steps

1. Keep `tests/regression/test_solver_vbm_consistency.py` passing against
   `VBMicrolensing().cmplx_roots_gen(...)`.
2. Keep `tests/regression/test_vbm_consistency.py` passing for
   `lcbinint.binary_mag0(...)`.
3. Connect binary point-source magnification into the higher-level
   `LensModel`/C ABI after trajectory-level coordinate handling is verified.
4. Add triple-lens VBM/VBMicrolensing comparison cases after the binary path is
   passing.
