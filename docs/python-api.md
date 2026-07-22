# Python API reference

This is the public Python surface exported by `import lcbinint`. It documents
the development version in this repository; names prefixed with `_` are
private diagnostic hooks and are intentionally omitted.

## `LightCurve`

`LightCurve` is the main reusable evaluator.

```python
lcbinint.LightCurve(options=Options(), model=None,
                    limb_darkening=LimbDarkening.none(), site=None)
```

It may also be constructed with convenience keyword arguments accepted by
`Options`, `Model`, `limb_darkening`, `sky`, `site`, and `t_ref`. Prefer the
explicit objects for nontrivial workflows.

| Method / property | Meaning |
| --- | --- |
| `curve(times, params)` / `magnification(...)` | Magnification array. `times` must be scalar or 1-D; the return is a 1-D NumPy array. |
| `magnification_batch(times, parameter_rows)` | One native call for independent parameter rows; returns shape `(n_rows, n_times)`. |
| `info(times, params)` | `LightCurveInfo` diagnostics for a single source. |
| `source_trajectory(times, params)` | Lens-frame `SourceTrajectory` after active trajectory effects. |
| `finite_source_geometry(times, params)` | Root-solve-free geometry and effective tolerance per epoch. |
| `separation([time], params)` | Effective binary separation; `time` is necessary for lens orbital motion. |
| `caustics([time], params)` | Source-plane `GeometryBranches`. |
| `critical_curves([time], params)` | Image-plane `GeometryBranches`. |
| `.options`, `.model`, `.site`, `.sky`, `.t_ref` | Construction-time configuration. |
| `.lens`, `.source`, `.orbital_motion`, `.terrestrial` | Read-only convenience views of the model. |

`light_curve_log_likelihood_batch(...)` is an internal inference integration
API. It is public for compatible host packages, but applications should use
their normal likelihood layer unless they specifically need its fused native
implementation.

## Evaluation parameters

Pass a dictionary or `Parameters`; dicts also accept the aliases listed below.
Unknown keys raise `KeyError`.

| Key | Alias | Meaning |
| --- | --- | --- |
| `t0` | `t_0` | Closest-approach epoch. |
| `tE` | `t_E` | Einstein timescale; uses the units of `times`. |
| `u0` | `umin` | Signed impact parameter. |
| `alpha` | `theta` | Source-trajectory angle in radians. |
| `s` | `sep` | Binary projected separation in Einstein-radius units. |
| `q` | — | Binary mass-ratio input in the selected coordinate convention. |
| `rho` | — | Source angular radius in Einstein-radius units; use positive values for finite-source integration. |
| `piEN`, `piEE` | — | North/East microlens-parallax components. |
| `q2`, `sep2`, `ang` | — | Enable/configure a triple lens when `q2 > 0`; `ang` is radians. |
| `ra`, `dec`, `tfix` | — | Per-call sky/reference overrides for parallax-capable trajectories. |
| `g1`, `g2`, `g3`, `lom_szs`, `lom_ar`, `v_sep` | — | Lens orbital-motion parameters. |
| `xi_1`, `xi_2`, `period_xa`, `ecc_xa`, `peri_xa`, `inc_xa` | — | Element-based xallarap inputs. |
| `w1`, `w2`, `w3` | `omega_xa`, `inc_xa`, `phi_xa` | Velocity-form xallarap inputs. |
| `xa_szs`, `xa_ar` | — | Kepler-velocity xallarap inputs. |
| `limb_darkening_c`, `limb_darkening_d` | — | Per-call limb-darkening coefficients. |

The limb-darkening values attached to `LightCurve` are normally sufficient.
Per-call coefficients are available when a host needs them in its parameter
container.

Binary-source dictionary fields are `q_source` (alias `fluxratio`), `t0_2`,
and `u0_2`; `q_mass` selects the coupled-xallarap form. They are deliberately
not properties of `Parameters`.

## `Options`

```python
lcbinint.Options(
    coordinates="vbm", nbin="auto", inverse_ray_grid="auto",
    tol=0.0, reltol=0.0,
)
```

| Option | Values / semantics |
| --- | --- |
| `coordinates` / `param_type` | `"vbm"` (default), `"lcbinint"`, `"center_of_mass"`, or `"vbm_center_of_mass"`. This controls input convention, not physics. |
| `nbin` / `source_bins` | Positive integer for a fixed inverse-ray grid, or `"auto"` (default) for calibrated automatic resolution. |
| `inverse_ray_grid` | `"cartesian"`, `"polar"`, or `"auto"` (default). |
| `tol` / `finite_source_tol` | Absolute finite-source error-budget term. |
| `reltol` / `finite_source_reltol` | Relative finite-source error-budget term. |
| `max_source_bins` | Upper resolution bound for automatic finite-source work. |
| `grid_ratio`, `polar_grid_ratio` | Image-plane grid extent controls; a non-positive polar value inherits `grid_ratio`. |
| `polar_nbin` / `polar_source_bins` | Optional polar-grid resolution override. |
| `caustic_bins` | Samples used to trace caustic/critical-curve branches. |
| `hex_tol` / `adaptive_hex_threshold` | Relative acceptance threshold for the adaptive hexadecapole path. |
| `point_source_threshold`, `hexadecapole_threshold` | Point-source and finite-source switching controls. |
| `xallarap_param_type`, `parallax_mode`, `mode` | Low-level compatibility controls; prefer `Model.xallarap` and named options. |

`tol` and `reltol` define one budget: `tol + reltol * max(|A|, 1)`. Leaving
both at zero selects the calibrated default. Details, including convergence
behavior, are in [Numerical methods](numerical-methods.md).

## `Model`, `obs.SkyCoord`, and `obs.Site`

```python
lcbinint.Model(
    lens="binary", source="single", orbital_motion="static",
    xallarap="none", parallax=False, terrestrial=False,
    sky=None, t_ref=None,
)
```

| Field | Accepted values |
| --- | --- |
| `lens` | `"binary"` or `"triple"`. A positive `q2` also selects triple-lens calculation at evaluation. |
| `source` | `"single"`, `"binary"`, or `"binary_source"`. |
| `orbital_motion` | `"static"`, `"circular"`, or `"kepler"`. |
| `xallarap` | `"none"`, `"orbital_elements"`/`"kepler"`, `"circular_elements"`/`"circular"`, `"circular_velocity"`, or `"kepler_velocity"`. |
| `parallax` | Enables annual-parallax trajectory treatment when parallax inputs are supplied. |
| `terrestrial` | Enables the ground-site term only when `parallax=True` and a ground `Site` is present. |
| `sky`, `t_ref` | Event sky coordinate and reference epoch required for annual parallax. |

`obs.SkyCoord(ra, dec, unit="deg")` accepts numerical degrees or sexagesimal
strings. Set `unit="hours"` when numeric RA is in hours; strings are parsed as
`hh:mm:ss`.

`obs.Site("ground", lat, lon)` takes north/east degrees. `obs.Site("space",
table)` takes an `N × 4` satellite table: `(JD, RA_deg, Dec_deg, distance_AU)`.
The table must have at least two rows and strictly increasing finite times.

## Limb darkening, outputs, and image plane

`LimbDarkening.none()` gives a uniform source. `LimbDarkening.linear(u)` uses
`I(mu) = 1 - u(1 - mu)`. `LimbDarkening.square_root(c, d)` uses
`I(mu) = 1 - c(1 - mu) - d(1 - sqrt(mu))`; `LimbDarkening.quadratic(c, d)`
is provided as a compatibility alias for the same two-coefficient container.

`LightCurveInfo` stores lists aligned with the supplied epochs. The key fields
are `magnifications`, `point_source_magnifications`,
`finite_source_magnifications`, `finite_source_method_names`,
`finite_source_error_estimates`, `finite_source_converged`, `all_converged`,
and `unconverged_indices`. It additionally exposes source, caustic, root, and
point-source-safety diagnostic vectors.

`SourceTrajectory` has `times`, `x`, and `y`. `FiniteSourceGeometry` has
`separation`, `mass_ratio`, `source_x`, `source_y`, `source_radius`, limb
darkening, and absolute/relative tolerance vectors. `GeometryBranches` has
parallel nested `x` and `y` lists.

`ImagePlane(q, s, x, y, rho=0, n_points=512, coordinates="vbm")` supplies
`images()`, `image_table()`, `seeds()`, `ray_shooting_images()`, `caustics()`,
`critical_curves()`, and `plot()`. It is a binary-lens geometry helper. Its
point-image data include position, magnification, Jacobian determinant, and
parity.

## Low-level finite-source function

```python
lcbinint.binary_ray_shooting(x, y, s, q, rho,
                              limb_darkening=LimbDarkening.none(),
                              options=Options())
```

This directly evaluates a finite binary source in lens-plane coordinates via
inverse-ray integration. `rho` must be positive. It corresponds to
VBMicrolensing's `BinaryMag2`: `x` and `y` are centre-of-mass source
coordinates. Use `LightCurve` for time-domain models and diagnostics.
