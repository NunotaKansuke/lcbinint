# 2026-06-25 polar mode evaluation

## Summary

Mode 2 is now a real polar inverse-ray path for finite-source integration.  The
old behavior could route near-caustic/high-magnification cases back to Cartesian
inverse ray, which made mode 2 uninformative for the regime where polar should
matter.

The original polar implementation was not reliable enough after removing that
fallback:

- high-mag curve, `s=0.95, q=1e-2, u0=-1e-3, alpha=0.5, rho=5e-3`
- strict mode 2 gave negative magnifications at some wing points
- after preventing repeated subtraction, it still undercounted by about 15%
- root cause was not the SG/NewImages solver; it was polar image-area
  bookkeeping

## Fix

The polar area kernel was changed from boundary-row tracing with post-hoc image
overlap subtraction to a polar-grid flood fill:

- cells are indexed by `(ir, iphi)`
- each cell is counted at most once by per-phi visited radial intervals
- same-phi radial runs are filled scanline-style, similar to Cartesian scanline
- area contribution is `brightness * r * dr * dphi`
- limb darkening uses the same mapped-source radius lookup as Cartesian

This makes polar a coordinate change of the Cartesian inverse-ray idea rather
than a separate fragile overlap algorithm.

## Benchmark snapshot

All runs below used `source_bins=50`, forced inverse-ray
(`hexadecapole_threshold=1e9`), local build, and VBM `Tol=1e-3`.

| case | LD | Cartesian ms/pt | polar ms/pt | polar/cart speedup | polar max rel vs VBM |
| --- | ---: | ---: | ---: | ---: | ---: |
| `s=0.95 q=1e-2 rho=5e-3` high-mag | no | 6.07 | 4.09 | 1.49x | 9.5e-4 |
| `s=0.95 q=1e-2 rho=5e-3` high-mag | yes | 7.26 | 4.59 | 1.58x | 5.8e-4 |
| `s=1.0 q=1e-3 rho=3e-3` high-mag | no | 7.70 | 8.02 | 0.96x | 1.4e-3 |
| `s=1.0 q=1e-3 rho=3e-3` high-mag | yes | 9.16 | 9.33 | 0.98x | 8.8e-4 |
| `s=0.8 q=1e-2 rho=5e-3` close/high-mag | no | 5.73 | 4.34 | 1.32x | 1.3e-3 |
| `s=0.8 q=1e-2 rho=5e-3` close/high-mag | yes | 6.83 | 4.94 | 1.38x | 8.1e-4 |
| `s=1.0 q=1e-3 rho=1e-4` tiny/high-mag | no | 42.27 | 22.74 | 1.86x | 3.4e-4 |
| `s=1.0 q=1e-3 rho=1e-4` tiny/high-mag | yes | 52.45 | 27.32 | 1.92x | 2.2e-4 |
| `s=1.0 q=1e-3 rho=3e-2` large source | no | 3.34 | 4.31 | 0.78x | 6.7e-3 |
| `s=1.0 q=1e-3 rho=3e-2` large source | yes | 3.78 | 4.91 | 0.77x | 4.5e-3 |

Interpretation:

- polar is now useful in high-magnification/tiny-source regimes
- polar is not universally faster; for larger sources Cartesian is usually
  better
- polar can be more accurate than Cartesian in tiny-rho high-mag cases where
  Cartesian aliasing is visible
- automatic switching should eventually be based on geometry/rho/magnification,
  not enabled globally

## Regression

Added a high-magnification polar regression:

- `test_lcbinint_polar_high_magnification_curve_matches_vbm_without_cartesian_fallback`
- checks that mode 2 actually reports `inverse_ray_polar`
- checks max relative error against VBM is below `1.5e-3`

During testing, an unrelated pybind ndarray bug was found and fixed:

- `py::array_t<double>(count)` produced stride `(0,)` in this environment
- ndarray light curves repeated the first value
- fixed by constructing arrays with explicit shape and stride
