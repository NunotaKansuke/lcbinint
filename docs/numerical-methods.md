# Numerical methods and runtime decisions

## Literature basis

The finite-source decision pipeline is an independent `lcbinint` implementation,
but its terminology and physical safeguards are grounded in the microlensing
literature. In particular, the quadrupole/cusp/ghost-image point-source tests
are informed by the decision tree described by [Bozza et al. (2018)](#references).
The automatic grid-size calibration, tolerance budget, and one-shot policy below
are specific to `lcbinint` and are validated by this repository's regression
and calibration data.

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

This is the quadrupole/hexadecapole finite-source expansion described by
[Gould (2008)](#references). The source-boundary sampling strategy and the
need to exclude caustic and cusp neighborhoods are also discussed there.

The absolute fourth-order contribution divided by `|A|` is the local
self-consistency indicator. A caustic-distance guard and the independent
quadrupole/cusp/ghost topology checks prevent a small fourth-order term from
being trusted when cancellation or a topology change makes the expansion
unsafe. Rejected positions continue to source-plane quadrature or inverse-ray
integration.

The ghost-image guard is especially important near an exterior fold approach:
the physical image count has not yet changed, while the non-physical pair of
polynomial roots signals the impending topology change. This use of the ghost
pair, together with the quadrupole and cusp checks, follows the safety logic in
[Bozza et al. (2018)](#references) (Section 5). Contour-integration error
control and adaptive sampling background are given by [Bozza (2010)](#references).

## Automatic `nbin`

Finite-source accuracy uses one combined absolute budget,

```text
max(finite_source_tol, finite_source_reltol * max(|A|, 1)).
```

If either term is explicitly positive, the other term is exactly zero unless
the caller also sets it. In particular, `tol=1e-5, reltol=0` is absolute-only;
it never inherits the survey default relative tolerance. When both terms are
unset, the calibrated default is
`max(1e-4, 1e-3 * max(|A|, 1))`. This convention is shared by point-source and
hexadecapole acceptance, Cartesian and polar inverse rays, source-plane
quadrature, and `finite_source_converged` diagnostics.

`Options(nbin="auto")` is the default. For each binary-lens source position it
makes one calibrated prediction. Point-source magnification selects Cartesian
below `Apoint=200` and polar at or above it. Separate absolute- and
relative-tolerance power laws are evaluated, the less demanding active branch
is selected, and the continuous result is rounded upward to an integer and
capped by `max_source_bins`.

The calibrated domains are `1e-4 <= reltol <= 1e-2` and
`2e-4 <= tol <= 1e-2`. An unsupported branch is ignored when the other branch
is supported; if neither is supported, automatic binary evaluation reports
`unsupported_tolerance`. In particular, absolute-only `tol <= 1e-4` is not
claimed by the empirical law.

The selected inverse-ray grid is evaluated exactly once. Cartesian and polar
still expose their embedded area-error estimates for diagnostics, but those
estimates do not veto the empirical prediction and do not increase `nbin`.
Image-component support certification remains mandatory because it detects a
different failure mode: a component omitted before integration. A support or
numerical failure is therefore still reported fail-closed.

A fixed integer `nbin` is also one-shot. When combined with an explicit
tolerance, its compatibility diagnostics may compare the selected grid with a
half-resolution grid, but the requested integer is never changed. Triple-lens
automatic resolution follows its separate calibration and is not governed by
the binary empirical law described here.

The exact binary coefficients, domains, and validation evidence are frozen in
`tests/diagnostics/results/recal2026/REPORT_empirical_resolution_law.md` and the
final Apoint validation artifacts linked from that report.

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

## References

1. V. Bozza, E. Bachelet, F. Bartolić, T. M. Heintz, A. R. Hoag, and M.
   Hundertmark, “VBBinaryLensing: a public package for microlensing light-curve
   computation,” *Monthly Notices of the Royal Astronomical Society* **479**,
   5157–5167 (2018), [doi:10.1093/mnras/sty1791](https://doi.org/10.1093/mnras/sty1791),
   [ADS](https://ui.adsabs.harvard.edu/abs/2018MNRAS.479.5157B/abstract).
2. V. Bozza, “Microlensing with an advanced contour integration algorithm:
   Green's theorem to third order, error control, optimal sampling and limb
   darkening,” *Monthly Notices of the Royal Astronomical Society* **408**,
   2188–2200 (2010), [doi:10.1111/j.1365-2966.2010.17265.x](https://doi.org/10.1111/j.1365-2966.2010.17265.x).
3. A. Gould, “Hexadecapole Approximation in Planetary Microlensing,” *The
   Astrophysical Journal* **681**, 1593–1598 (2008),
   [doi:10.1086/588601](https://doi.org/10.1086/588601),
   [arXiv:0801.2578](https://arxiv.org/abs/0801.2578).
