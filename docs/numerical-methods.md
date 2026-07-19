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

Finite-source accuracy uses one combined absolute budget,

```text
finite_source_tol + finite_source_reltol * max(|A|, 1).
```

If either term is explicitly positive, the other term is exactly zero unless
the caller also sets it. In particular, `tol=1e-5, reltol=0` is absolute-only;
it never inherits the survey default relative tolerance. When both terms are
unset, the calibrated default is `1e-4 + 1e-3 * max(|A|, 1)`. This convention
is shared by point-source and hexadecapole acceptance, Cartesian and polar
inverse rays, source-plane quadrature, and `finite_source_converged`.

`Options(nbin="auto")` is the default. For each source position it makes one
calibrated prediction from quantities already available on the hot path:
point-source magnification, source radius, the smaller mass ratio, normalized
caustic distance, companion-resolution risk, and linear limb darkening.

The prediction is rounded upward to a supported bucket and capped by
`max_source_bins`. High-magnification positions may select the polar grid;
ordinary positions use the Cartesian model. Cartesian integration checks its
independent area-error estimate after the predicted grid is evaluated. If that
estimate exceeds the same tolerance budget used by `finite_source_converged`,
automatic mode increases to the smallest supported bucket implied by the
measured shortfall and retries, up to `max_source_bins`. Fixed integer `nbin`
never retries. A result that still cannot meet its budget at the cap is returned
with `finite_source_converged=False`.

An explicit tolerance is converted to its equivalent relative budget at the
current point-source magnification before the initial `nbin` prediction. Thus
tight absolute-only requests increase the first grid up front instead of
starting at the loose-default prediction. Smooth corrected Cartesian
boundaries and polar grids use second-order (square-root) tolerance scaling;
caustic-contact and unresolved-small-companion Cartesian rows use their
observed first-order scaling. Tight triple-lens requests start at the configured
ceiling because no independently validated tighter-tolerance triple regression
exists. These choices avoid knowingly under-resolved preliminary passes.

The predictor was calibrated at the loose default, so its tighter-tolerance
scaling is conservative rather than independently fitted. The post-check
remains authoritative, and a caller that needs VBMicrolensing-style fallback
should switch engines whenever `finite_source_converged` is false rather than
treating the requested value as proof of achieved accuracy.

For polar inverse rays, an explicit tolerance activates an independent coarse/
fine grid comparison. Automatic mode jumps directly to the bucket implied by
the measured shortfall, up to `max_source_bins`; fixed `nbin` reports the
coarse/fine error without retrying.

The Cartesian area estimator follows the corrected scan's actual order. A
no-fold boundary with only small row-to-row jumps receives the extra cell-width
factor of a second-order edge rule. Fold seeds and large jumps retain the
first-order topology-warning scale, and the independent small-companion and
tangent-caustic guards remain active. Thus a large but smooth image boundary
does not trigger a retry merely because it crosses many grid rows.

For an explicit tolerance, a Cartesian result that the embedded estimator
would accept is additionally compared with one coarser grid. This catches rare
lattice-aliasing cases where the area indicator is optimistic. The comparison
is deliberately conditional: default calculations and rows already known to
miss their budget incur no second integration.

The exact fitted coefficients and validation metrics are frozen under
`tests/diagnostics/results/finite-source-auto-20260716/`. Independent validation
had zero underpredictions across 3,655 evaluable rows at the calibrated target
of relative `1e-3` plus absolute `1e-4`. Tighter requested tolerances use a
conservative convergence-order scaling but do not inherit that exact
zero-violation claim.

## Finite-source geometry for external hosts

lcbinint never imports or links to VBMicrolensing and has no concept of
backend routing. For a host that implements its own finite-source engine and
needs the same trajectory/orbital-motion transformations lcbinint uses
internally, `lcbi_finite_source_geometry[_array]` returns the engine-neutral,
root-solve-free geometry (separation, mass ratio, source position and radius,
limb darkening, tolerances) for a given epoch, and `lcbi_result` carries the
time-evolved `separation`/`mass_ratio` plus `caustic_distance` alongside every
magnification. Both are cheap enough to call per epoch on a hot path, unlike
`lcbi_magnification[_array]`. moasarc is one such host: it owns its own
calibration rule and VBM-routing decision entirely on its side, using these
primitives.

## Annual and terrestrial parallax

Parallax is active only when `Model(parallax=True, ...)` is attached to the
`LightCurve`. Non-zero `piEN`/`piEE` parameters alone do not activate it.
Annual parallax uses the bundled Earth ephemeris, sky coordinates, and the
fixed reference epoch `t_ref`.

Terrestrial parallax additionally requires `terrestrial=True` and an explicit
ground `obs.Site("ground", lat, lon)`; enabling it without a ground site is an error.
Observatory longitude is east-positive. The geocentric telescope
position is rotated using mean sidereal time and projected onto the same north
and east sky basis as annual parallax. A real site at latitude/longitude
`(0, 0)` is valid; absence of a site is represented separately and produces no
terrestrial offset.

Spacecraft parallax uses `obs.Site("space", table)`, where `table`
has columns `(JD, RA_deg, Dec_deg, distance_AU)`. It follows the
VBMicrolensing satellite-table convention: the geocentric spacecraft position
is linearly interpolated and added directly to the Earth annual-parallax term,
without subtracting its position or velocity at `t_ref`. A single event
`Model` may have `terrestrial=True` and be shared by both ground and space
curves; the terrestrial term is applied only to its ground curves.

The current terrestrial calculation uses a spherical Earth at the equatorial
radius and GMST without polar motion, elevation, or sub-arcsecond nutation.
Those approximations are much smaller than the present microlensing use case,
but should be revisited before claiming precision astrometry at that scale.
