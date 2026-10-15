# Binary light-curve automatic integration

`LightCurve(lens="binary")` uses `Options(inverse_ray_grid="auto", nbin="auto")`
by default. Point-source and hexadecapole safety checks run first. When an
integral is required, binary sources with linear limb darkening use the
empirical Cartesian/Polar selector and run one retained adaptive integrator.

The selector uses centre-image Jacobians, local second derivatives, image
radii, lens/source geometry, limb darkening and requested relative tolerance.
It consists of forty depth-three regression trees. Its fitted domain is
relative tolerance 1e-3 or 1e-4, limb coefficient 0 or 0.5, and the geometry
bounds in `adaptive_route_selector.hpp`. Other linear-LD configurations use
adaptive Polar without claiming a calibrated speed prediction.

Cartesian starts at 16, Polar at 2. The default `max_source_bins` is now 512,
matching Cartesian's dyadic refinement ladder. Explicitly supplied limits are
respected. Explicit fixed grids and non-linear LD keep their existing paths.
The centre images used for seed construction also supply selector features;
this light-curve path does not call the separate Python convenience router.

The selected integrator supplies the error indicator and acceptance state.
The light-curve layer additionally checks the dimensional budget at the
returned magnification and retains the existing grazing-caustic guard.
Unmet tolerances are reported through the normal numerical-error status;
selecting a method does not certify its error model.

The Python diagnostic convenience function `binary_adaptive_auto_flux` is
also available. That wrapper separately solves the centre images before
calling an existing diagnostic entry, so its timing boundary differs from
the light-curve path.

Calibration, timing limits, known numerical failures and data provenance:
`tests/diagnostics/results/recal2026/predictive-route-20261015/REPORT.md`.
Default light-curve checks:
`tests/diagnostics/recal2026/check_lightcurve_predictive_auto.py`.
