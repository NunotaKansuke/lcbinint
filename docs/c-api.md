# C/C++ API reference

The public native interface is [`include/lcbinint/lcbinint.h`](../include/lcbinint/lcbinint.h).
It is a C ABI and can therefore be called from C++ and other FFI-capable
languages. The Python package is a consumer of the same core but is not needed
to use these functions.

## Include and initialization

```c
#include <lcbinint/lcbinint.h>

lcbi_params params = lcbi_default_params();
lcbi_options options = lcbi_default_options();
lcbi_result result;

params.t0 = 2459000.0;
params.tE = 20.0;
params.umin = 0.01;
params.theta = 0.5;
params.sep = 1.1;
params.q = 0.01;

lcbi_status status = lcbi_magnification(2459001.0, &params, &options, &result);
if (status != LCBI_OK) {
    fprintf(stderr, "lcbinint: %s\n", lcbi_status_string(status));
}
```

Always initialize `lcbi_params` and `lcbi_options` with their default factory
functions. The structures contain fields for optional physical effects, and a
zero-filled structure is not a supported substitute for their defaults.

## Status values

Every compute call returns an `lcbi_status`.

| Value | Meaning |
| --- | --- |
| `LCBI_OK` | Calculation completed. |
| `LCBI_INVALID_ARGUMENT` | A pointer, array length, parameter, or option was invalid. |
| `LCBI_NUMERICAL_ERROR` | The requested calculation could not complete numerically. |
| `LCBI_UNSUPPORTED` | The requested combination is not implemented. |

Use `lcbi_status_string(status)` for a stable printable description. Do not
read result fields after a non-OK status unless a future API contract explicitly
states that a particular field is valid.

## Entry points

| Function | Use |
| --- | --- |
| `lcbi_default_params()` | Return initialized physical parameters. |
| `lcbi_default_options()` | Return initialized numerical options. |
| `lcbi_magnification(time, params, options, result)` | Evaluate one epoch. |
| `lcbi_magnification_array(times, count, params, options, results)` | Evaluate `count` epochs into caller-owned storage. |
| `lcbi_finite_source_geometry(time, params, options, geometry)` | Resolve trajectory/orbital geometry without lens-root solving. |
| `lcbi_finite_source_geometry_array(...)` | Array version of the preceding function. |
| `lcbi_status_string(status)` | Convert a status enum to text. |

Array inputs and outputs are contiguous caller-owned arrays of `count` items.
`count` must be valid and non-negative; the `times`, `results`, or `geometries`
pointer must be suitable for that count. The array API is preferable when the
physical parameters and numerical options are shared across epochs.

## Physical parameters: `lcbi_params`

The canonical native names are shown below. Time quantities use one consistent
caller-selected unit; angles are radians unless a field explicitly says
degrees.

| Fields | Meaning |
| --- | --- |
| `t0`, `tE`, `umin`, `theta` | Rectilinear source trajectory: closest epoch, Einstein timescale, signed impact parameter, and trajectory angle. |
| `sep`, `q`, `rho` | Binary separation, mass ratio, and source radius in Einstein-radius units. |
| `q2`, `sep2`, `ang` | Triple-lens companion; a positive `q2` enables triple mode. |
| `piEN`, `piEE` | North/East annual-parallax components. |
| `ra`, `dec`, `earth_axis`, `tfix` | Sky/reference data used by the trajectory and parallax implementation. |
| `obs_lat`, `obs_lon` | Ground observatory latitude/longitude in degrees. `NaN` represents no site. Longitude is east-positive. |
| `limb_darkening_c`, `limb_darkening_d` | Coefficients of the two-term source profile. |
| `orbital_motion_mode` | `LCBI_ORBIT_STATIC`, `LCBI_ORBIT_CIRCULAR`, or `LCBI_ORBIT_KEPLER`. |
| `g1`, `g2`, `g3`, `lom_szs`, `lom_ar`, `v_sep` | Lens-orbital-motion parameters. |
| `xi_1`, `xi_2`, `omega_xa`, `inc_xa`, `phi_xa` | Xallarap amplitudes/velocity-form orbital values. |
| `piEN_xa`, `piEE_xa`, `ra_xa`, `dec_xa`, `period_xa`, `ecc_xa`, `peri_xa` | Element-based xallarap values and related source-orbit data. |

The selected `lcbi_xallarap_param_type` determines which xallarap fields have
meaning. `LCBI_XALLARAP_ANGULAR_VELOCITY` is deprecated; new integrations
should select one of the element or velocity modes documented in the header.

## Numerical options: `lcbi_options`

| Fields | Meaning |
| --- | --- |
| `vbm_compatible`, `center_of_mass` | Coordinate/parameter convention controls. The Python default corresponds to VBM-compatible input. |
| `source_bins`, `automatic_source_bins`, `max_source_bins` | Fixed or calibrated automatic finite-source resolution and its cap. |
| `mode` | Finite-source inverse-ray mode: `1` Cartesian, `2` polar, `4` automatic. |
| `grid_ratio`, `polar_source_bins`, `polar_grid_ratio` | Cartesian/polar grid sizing. A non-positive polar ratio inherits the regular ratio. |
| `finite_source_tol`, `finite_source_reltol` | Absolute and relative terms of the finite-source error budget. |
| `adaptive_hex_threshold`, `hexadecapole_threshold` | Finite-source fast-path acceptance controls. |
| `point_source_threshold` | Fast point-source exit margin, in units of `rho`. |
| `caustic_bins` | Samples for caustic tracing. |
| `xallarap_param_type`, `parallax_mode`, `orbit_pair` | Advanced model/compatibility controls. |

The error budget is `finite_source_tol + finite_source_reltol * max(|A|, 1)`.
When both terms are zero, the calibrated default is selected. Read
[Numerical methods](numerical-methods.md) before changing finite-source
switching or interpreting an unconverged result.

## Results

`lcbi_result` holds the primary result and diagnostics for one epoch.

| Fields | Meaning |
| --- | --- |
| `magnification` | Selected final magnification. |
| `point_source_magnification`, `finite_source_magnification` | Contributions/diagnostic values from the fast and finite-source paths. |
| `source_x`, `source_y`, `image_count` | Resolved source position and point-image count. |
| `finite_source_method`, `finite_source_error_estimate`, `finite_source_refinement_level`, `finite_source_converged` | Numerical method and requested-budget outcome. Check `finite_source_converged` for finite-source accuracy claims. |
| `root_*` | Root-candidate, deduplication, retry, precision, and residual diagnostics. |
| `point_source_*` | Quadrupole/cusp/ghost/planetary safety indicators and flags used in point-source decisions. |
| `separation`, `mass_ratio`, `caustic_distance` | Time-evolved lens state and source-to-caustic diagnostic. |

`lcbi_geometry` is the lightweight output of the finite-source-geometry calls.
It contains effective `separation`, `mass_ratio`, source position/radius,
limb-darkening coefficients, absolute/relative tolerances, and `valid`.
It is designed for callers that own a separate finite-source engine but need
the exact trajectory, parallax, and orbital transformations used by lcbinint.

## Linking from CMake

The project installs the headers and two library targets:
`lcbinint_magnification` and `lcbinint_lightcurve`. Link consumers against
`lcbinint_lightcurve`, which brings in the magnification layer. The library
requires GSL and the math library on platforms where it is separate.

The C header uses `extern "C"` under C++, so no C++ name mangling leaks through
this interface. C++ applications can use the public C API directly; the
internal headers under `src/` are implementation details and are not the
compatibility boundary.
