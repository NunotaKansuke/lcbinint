# Numerical methods and runtime decisions

## Finite-source fast paths

Binary and triple finite-source evaluations first test whether the point-source
solution is safe. Away from caustics, the next approximation is the
hexadecapole expansion. It evaluates the source centre plus 12 samples on the
source disk: four axial samples at radius `rho`, four at `rho/2`, and four
diagonal samples at `rho`.

The resulting second- and fourth-order terms are combined as

```text
A = A0 + 1/2 A2 rho^2 (1 - Gamma/5 - Lambda/9)
       + 1/3 A4 rho^4 (1 - 11 Gamma/35 - 7 Lambda/39).
```

The absolute fourth-order contribution divided by `|A|` is the local
self-consistency indicator. A caustic-distance guard and the independent
quadrupole/cusp/ghost topology checks prevent a small fourth-order term from
being trusted when cancellation or a topology change makes the expansion
unsafe. Rejected positions continue to source-plane quadrature or inverse-ray
integration.

## Automatic `nbin`

`Options(nbin="auto")` is the default. For each source position it makes one
calibrated prediction from quantities already available on the hot path:
point-source magnification, source radius, the smaller mass ratio, normalized
caustic distance, companion-resolution risk, and linear limb darkening.

The prediction is rounded upward to a supported bucket and capped by
`max_source_bins`. High-magnification positions may select the polar grid;
ordinary positions use the Cartesian model. This is deliberately a one-shot
choice: no trial integration or silent retry is performed. A result that does
not meet its requested budget is returned with `finite_source_converged=False`.

The exact fitted coefficients and validation metrics are frozen under
`tests/diagnostics/results/finite-source-auto-20260716/`. Independent validation
had zero underpredictions across 3,655 evaluable rows at the calibrated target
of relative `1e-3` plus absolute `1e-4`. Tighter requested tolerances use a
conservative square-root resolution scaling but do not inherit that exact
zero-violation claim.

## External contour recommendation

lcbinint never imports or dispatches to VBMicrolensing. Its finite-source core
computes an internal per-position hint that an external contour engine is
likely to be both accurate and faster:

- uniform source: `A_point < 1000`, excluding the tangent band
  `0.9 < d_caustic/rho < 1.1`;
- limb-darkened source: `A_point < 5` and `d_caustic/rho > 1.05`.

This is an internal backend-routing hint, not a public API or accuracy oracle. The
independent calibration selected 2,261 rows with no inaccurate or failed VBM
results; among rows where both engines were accurate, VBM was faster in
2,158/2,249 cases. A future generic sampler/backend adapter can consume the
internal result without making VBMicrolensing a dependency of lcbinint.

## Annual and terrestrial parallax

Parallax is active only when the `LightCurve` physical model sets
`parallax=True`. Non-zero `piEN`/`piEE` parameters alone do not activate it.
Annual parallax uses the bundled Earth ephemeris, sky coordinates, and the
fixed reference epoch `t_ref`.

Terrestrial parallax additionally requires `terrestrial=True` and an explicit
`obs.Site`. Observatory longitude is east-positive. The geocentric telescope
position is rotated using mean sidereal time and projected onto the same north
and east sky basis as annual parallax. A real site at latitude/longitude
`(0, 0)` is valid; absence of a site is represented separately and produces no
terrestrial offset.

The current terrestrial calculation uses a spherical Earth at the equatorial
radius and GMST without polar motion, elevation, or sub-arcsecond nutation.
Those approximations are much smaller than the present microlensing use case,
but should be revisited before claiming precision astrometry at that scale.
