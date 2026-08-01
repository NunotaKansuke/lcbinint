# Certified component support — resolution of the 20260801 handoff

Branch `codex/certified-component-support`.  The earlier handoff note that
described the problem alongside an unfinished separate tile integrator has been
deleted with this branch: the tile integrator is gone, and the fix is in the
seeding, not in a second kernel.  `codex/tile-approach-archive` keeps the
abandoned integrator for reference.

## The defect

Reference geometry `s=1.2, q=0.1, u=(0.653, 0.020), rho=0.020`
(`coordinates="center_of_mass"`).

The caustic passes 1.999177e-02 from the source centre against `rho = 2.0e-02`,
so the five-image cap is **8.226e-06 deep = 4.1e-04 rho** and subtends
**0.0878 rad** of the limb (1/72 of the circle).

Every seeding path reached the cap through a fixed-fraction offset from the
limb: `append_caustic_probe_image_seeds` and `append_boundary_probe_image_seeds`
(`inward_fraction = 0.02`, i.e. `0.98 rho`) both step at least `0.02 rho =
4e-4`, which is **49x deeper than the cap**.  No such probe can enter it.  The
JAX path used an 8/16/32-sample limb raster, whose 45/22.5/11.25 degree spacing
hits the 5-degree arc by luck — that is the historical 8-vs-16 discrepancy.

Consequence: the magnification converged smoothly to 3.94848 instead of
3.96089 (**-3.1e-03 relative**) *with a healthy error indicator*.  A silent
wrong answer, not a noisy one.

## The fix

`src/lcbinint/magnification/component_certificate.{hpp,cpp}`.  The module is
lens-agnostic: it takes a caustic polyline and a disk, and knows nothing about
how many lenses produced the polyline.

Every connected component of `f^-1(D)` has its boundary in `f^-1(dD)`, so a
component is only discoverable from a source point inside the component of
`D \ K` it covers (`K` = caustic).  Conversely every component `C` of `D \ K`
either is all of `D` (when `K` misses `D`) or has a caustic arc on `dC`; on any
such arc `|zeta - centre|` attains a local extremum inside `D`, and there the
caustic is tangent to a circle about the centre, so the normal ray leaves the
caustic on one side only.

**Probing every local extremum of `r(phi) = |zeta(phi) - c|` along `K ∩ D`
therefore reaches every component of `D \ K`.**  The criterion mentions no limb
raster, no integration grid and no refinement level, which is what makes it a
completeness certificate rather than a denser heuristic.

Two details of *how* an extremum is probed:

- **Four directions, not two.**  The tangency argument gives the normal at a
  smooth extremum, but an extremum can also sit at a **cusp**, where the caustic
  has no tangent and the wedge opens along the *tangent* direction instead.  So
  `resolve_certified_probes` walks `{normal+, normal-, tangent+, tangent-}`
  (`kProbeDirections = 4`).  A cusp component is entered from the tangent
  probes; the normal probes at a cusp land in the same region on both sides and
  prove nothing.
- **The straddle criterion.**  An extremum counts as resolved when two of its
  probes return *different* image counts — that is the direct statement that a
  caustic arc separates them.  The first implementation used a fixed floor
  (`count > 3`, "a five-image probe was found"), which is a binary-lens fact:
  nested triple caustics have regions of 4, 6, 8 and 10 images, and no constant
  is right for all of them.  Straddling needs no constant.

Supporting details:

- The transverse polyline error is bounded per vertex by the sagitta
  `|v[i-1] - 2 v[i] + v[i+1]|`, which stays valid at cusps.  Extrema within that
  margin of the limb are kept as near-misses and probed from the facing limb
  point.
- Probe offsets are fractions of the *room available along the normal to the
  disk edge*, not fractions of `rho`, so they scale with the cap instead of
  overshooting it.
- The module performs **no lens-equation solves**.  It returns source-plane
  probe positions only, so the native `double` kernel and the JAX Jet kernel
  consume one identical descriptor while keeping their own root solvers.
  `resolve_certified_probes` is the shared traversal, so a descriptor cannot
  mean two different things to them.

### Fail-closed contract

A caustic arc always separates a three-image region from a five-image one, so a
certified extremum inside the disk with no five-image probe on either normal
means a component exists that the support does not cover.

- Native: `augmented_image_seeds` reports `support_proven`; when it is false the
  `diagnose` lambda sets the error estimate to infinity, so the adaptive loop
  cannot report convergence.
- JAX: `discover_cartesian_support` folds `!support_proven` into
  `root_failure`, which every consumer already reads as "support not
  established" (`support_valid = ~(overflow | root_failure)`).

### Triple lenses

The certificate ports unchanged — `certify_disk_support` takes the triple
caustic branches, `append_certified_triple_component_seeds` resolves the probes
through the triple root solver, and `augmented_triple_image_seeds` reports
`support_proven` exactly as the binary path does.  Nothing in the completeness
argument had to be re-derived, which is the point of the argument.

It matters more here.  Reference geometry
`s=1.0, q=1e-3, q2=1e-4, sep2=0.5, ang=1.2`, source at `(-0.05, 0.02)` with the
disk overlapping the nearest caustic point by 1 % (`rho = 6.497855561e-03/0.99`):

| bins | baseline | certified |
|-----:|---------:|----------:|
| 32 | 17.464616522 | 17.494845531 |
| 64 | 17.468326328 | 17.499112462 |
| 128 | 17.469199505 | 17.500163129 |
| 256 | 17.469392721 | 17.500390489 |
| 512 | 17.469464525 | 17.500473031 |
| 2048 | 17.469485364 | 17.500497508 |
| adaptive 1e-4 | 17.469486455 | 17.500498641 |
| adaptive 1e-5 | 17.469486455 | 17.500498641 |

Both sequences are smooth and both converge; they converge to different numbers
**1.8e-03 apart**, and the baseline reported `all_converged` at both tolerances
— tightening the tolerance could not expose it.  Two more centres on the same
caustic behave the same way (7.455358 vs 7.465791, 12.618701 vs 12.638474).

### Two defects in the fill machinery, exposed by the new seeds

Neither is in the certificate; both were latent and only became reachable once
the seed set changed.

1. **Lattice snapping could carry a seed across the critical curve.**  Seeds are
   snapped to `x = ix*incr, y = iy*incr` so the claimed-cell registry is exact.
   A probe taken just off a caustic arc has an image just off the *critical*
   curve, closer to it than one cell at any usable resolution; rounding could
   put it on the far side, and the fold pair then lost the seed for one of its
   two members.  Symptom: the triple ladder dipped at scattered bin counts
   (48 and 192 gave 17.4822/17.4844 against 17.5002 either side) — a resolution
   lottery, the exact signature the certificate is supposed to remove.
   Fix: `lattice_snapped_seeds` now records `sign(J)` at the raw seed and only
   accepts a lattice cell that reproduces it, falling back to a free snap only
   when the seed is already on the curve.

2. **Seed order decided whether the fold guard engaged.**  A fill whose seed has
   `|J| < kFoldJacobianThreshold` (0.02) is confined to one side of the critical
   curve, so an x-scan cannot bleed from `F+` into `F-` and apply boundary
   corrections at the seam — which is not a source boundary.  The guard is a
   property of the *first* seed to claim a component.  The certified ladder
   starts half a disk away from the caustic, so its images have large `|J|`;
   running the certified stage first made it the first claimant of the fold
   component and the guard never engaged.  Symptom: over-counting at a cusp
   (16.1072 falling to 16.0959 over bins 192..2048, against a converged
   16.094686), and a JAX-vs-native finite-difference gradient of opposite sign.
   Fix: the certified stage runs **last** in both seeding functions, so the ring
   probes — which hug the caustic and therefore carry the guard — keep first
   claim on every component they do reach.  Two cheaper-looking fixes were tried
   and reverted: a second claimed-cell registry for guarded fills (broke
   wide-caustic fold overlap, no measurable benefit) and a `stable_partition` of
   all seeds by `|J|` (guarded too many fills and clipped fold image area, three
   regressions).

### Removed

`binary_tile_kernel.{hpp,cpp}` and its bypass out of
`inverse_ray_cartesian_binary_mag`.  A second `double`-only integrator behind a
geometry test is exactly the "weird fallback" the fix was meant to avoid, and
the existing flood fill integrates the certified seeds correctly.

## Measured result

Native Cartesian flood fill, fixed `source_bins`, uniform source.  Reference is
VBMicrolensing `BinaryMag2` at `Tol=1e-9` and `1e-10` (they agree to 1.1e-10):
**3.960888498085**.  Note `Tol=1e-7` returns 3.960889170843, 6.7e-7 high — the
tangency is where VBM's own accuracy goal stops being met, and the handoff's
reference was that Tol=1e-7 value.

| bins | value | rel. error |
|-----:|------:|-----------:|
| 32 | 3.942493231 | -4.65e-03 |
| 64 | 3.952459074 | -2.13e-03 |
| 128 | 3.958808689 | -5.25e-04 |
| 256 | 3.960254926 | -1.60e-04 |
| 512 | 3.960698849 | -4.79e-05 |

Strictly monotone — the historical signature of the defect was a non-monotone
sequence that dipped wherever the limb raster missed the cap.

Adaptive precision at `finite_source_reltol = 1e-5`: **3.960884165, -1.09e-06
relative, 1.25 s**.  (An earlier revision of this note claimed 1e-4 and 1e-5
"agree to all digits".  They do not: 1e-4 stops at 400 bins and returns
3.960763257, -3.16e-05 relative — inside the tolerance it was given, which is
the contract, but not the same number.  See *Certifying the tolerance* below
for why 1e-5 used to return a NaN instead.)

Linear limb darkening `c = 0.5` converges to **3.836158013** the same way.  VBM
is not usable as an external reference there: `BinaryMagDark` returns 3.9675 for
`a1=0.25` but 4.5751 for `a1=0.5` on this geometry, so its limb-darkened
annulus scheme has already broken down.  (The handoff's 3.836256795465 came
from the unfinished tile kernel and is 2.6e-5 high.)

### Cost

`certify_disk_support` returns no extremum when the caustic misses the disk, so
no probe is solved and clear geometries pay nothing.  A/B against `dcef894`,
same machine, `OMP_NUM_THREADS=1`, median of 7 warm calls.

Native binary (ms):

| case | 128 base | 128 cert | 256 base | 256 cert |
|:-----|---------:|---------:|---------:|---------:|
| tangent_cusp | 3.756 | 3.878 | 7.528 | 7.587 |
| wide_equal_mass | 2.392 | 2.429 | 4.854 | 4.775 |
| close_binary | 1.448 | 1.599 | 1.577 | 1.342 |
| planetary | 1.373 | 1.369 | 1.362 | 1.255 |
| caustic_crossing | 4.950 | 5.019 | 13.034 | 13.241 |
| far_from_caustic | 1.329 | 1.337 | 1.344 | 1.348 |

Native triple (ms):

| case | 128 base | 128 cert | 256 base | 256 cert |
|:-----|---------:|---------:|---------:|---------:|
| tangent_cap | 9.173 | 9.548 | 28.058 | 28.628 |
| near_tangency_clear | 8.039 | 8.101 | 28.083 | 28.078 |
| outer_fold_clear | 3.957 | 4.027 | 12.066 | 12.131 |
| cusp_clear | 5.695 | 5.754 | 19.270 | 19.365 |
| central_caustic | 21.025 | 21.310 | 75.772 | 77.010 |
| far_from_caustic | 0.126 | 0.126 | 0.127 | 0.126 |

JAX (`binary_inverse_ray`, ms, limb-sample independent):

| case | 128 base | 128 cert | 256 base | 256 cert |
|:-----|---------:|---------:|---------:|---------:|
| tangent_cusp | 8.919 | 9.429 | 24.482 | 25.389 |
| wide_equal_mass | 5.123 | 5.163 | 14.129 | 14.210 |
| planetary | 4.669 | 4.550 | 12.518 | 12.583 |
| caustic_crossing | 15.037 | 15.175 | 42.947 | 43.336 |

The worst case is **+4 %**, on exactly the geometries that gain a component;
everything else is inside run-to-run scatter and the far-from-caustic case is
unchanged to the digit.  The certificate is not a per-epoch tax: it costs where
there is a caustic inside the disk, and nothing where there is not.

### The JAX limb raster stays

The JAX values and timings are identical for `limb_samples` 8, 16 and 32, which
looked like an invitation to delete the raster in `discover_cartesian_support`
now that the certificate covers completeness.  Measured, and it should not be
deleted, for two independent reasons.

There is no speedup to take.  Across twelve geometries at resolution 128 and
256, `limb_samples = 1` against `limb_samples = 32` is inside run-to-run
scatter (e.g. tangent cusp 9.36 vs 9.48 ms, fold crossing 21.69 vs 21.75 ms).
The raster is `limb_samples + 1` quintic solves; the fill is millions of cells.

And it is not redundant.  It is a *completeness* criterion no longer, but it is
still doing tile-graph work: the limb images trace the whole boundary of every
image component, so they seed tiles all along it.  Dropping to
`limb_samples = 1` costs 2.2e-04 relative on the tangent cusp at the default
`tile_size = 16`.

That loss is the visible end of a separate defect, of the same class as the one
this document is about.  A tile joins the frontier only if one of nine sample
points inside it maps into the source disk (`tile_has_inside_probe`), and the
extra fold pair here is a 55:1 sliver, so a tile the sliver merely passes
through can fail all nine and stop the expansion.  Tangent cusp at resolution
128, `limb_samples = 8`, against the VBM reference 3.960888498085:

| `tile_size` | value | rel. error |
|------------:|------:|-----------:|
| 2 | 3.960952993 | +1.6e-05 |
| 4 | 3.960856754 | -8.0e-06 |
| 8 | 3.959864455 | -2.6e-04 |
| 16 (default) | 3.955731463 | -1.3e-03 |
| 32 | 3.945948543 | -3.8e-03 |

Monotone in `tile_size`, `support_valid` true throughout, and absent on
geometries with no sliver (`cusp_inside` and a resonant crossing are
`tile_size`-independent to all digits).  This is a fixed sample pattern being
used where a completeness criterion is needed — the frontier test should bound
`min |f(z) - zeta|` over the tile rather than sample it — and it is not fixed
here.  The raster is what currently keeps it small, so removing the raster
would expose it.

### The smooth-route safety constants stay too

`point_source`, `hexadecapole` and `source_plane_quadrature` expand about the
disk centre, so all three are wrong by an amount unrelated to their own error
estimate once a caustic passes through the disk.  The router keeps them out of
that regime with calibrated proxies — `kQuadrupoleCuspSafety = 6`,
`kGhostSafety = 3`, `kPlanetarySafety = 2`, `kPreflightPointSafety = 30`,
`kMeasuredTopologyReleaseDistance = 10` rho, and a 20-rho rebuild window.  The
certificate's `min_caustic_distance` / `caustic_touches_disk` are *proven*
statements about the same question, so the obvious move was to replace the
constants with them.  Measured, and there is nothing to replace.

The certificate's bound is the polyline segment distance minus the local
sagitta.  At realistic `caustic_bins` the sagitta term is already negligible,
because it falls as `n^-2`:

| `caustic_bins` | sagitta / distance |
|---------------:|-------------------:|
| 200 | 1.7e-03 .. 1.6e-02 |
| 400 | 4.3e-04 .. 3.5e-03 |
| 1400 | 3.5e-05 .. 2.0e-04 |
| 4000 | 4.3e-06 .. 2.5e-05 |

(three geometries: `s=1.05, q=1e-3` at 6.24 rho from the caustic;
`s=1.2, q=0.1` at 1.00 rho; `s=1.0, q=1e-3` at 397 rho.)  Two to four orders
below the 10-rho release threshold the constants govern — so making the release
test "proven" via the polyline margin changes no decision.

And the constants are not letting anything through.  Two sweeps, both using the
certificate's own bound as the ground truth:

| sweep | trials | smooth routes | violations |
|-------|-------:|--------------:|-----------:|
| uniform over the box | 4000 | 3952 | 0 |
| 0..30 rho from a random caustic vertex | 5280 | 3684 | 0 |

The near-caustic sweep is the one that matters — uniform sampling lands near
the release boundary almost never — and it drew `s` and `q` log-uniformly over
`0.4..2.5` and `1e-4..1`, `rho` over `3e-4..3e-2`, and routed
`{inverse_ray_cartesian: 1491, hexadecapole: 2654, source_plane_quadrature:
344, point_source: 686, inverse_ray_polar: 105}`.  Not one smooth route was
chosen for a disk the caustic provably enters.

What is left in those constants is Taylor-remainder calibration: how far from a
caustic a quadrupole expansion is accurate *enough*, which is an analytic
question about the remainder, not a geometric one, and caustic geometry cannot
prove it.  So the routing code is unchanged and the property the constants
exist to deliver is pinned instead, as a randomised near-caustic invariant test
(`tests/regression/test_smooth_route_clearance.py`).

## Convergence order — what is left, and why it is not a defect

This section diagnoses the residual order on the **uniform** grid; the numbers
in it are the state before per-component refinement, which is what the next
section then removes 55-60 % of.  They are kept because the diagnosis is what
motivated the fix.

The tangent case converges at **h^1.7**, not `h^2`.  Sweeping the source through
the tangency at fixed 256 bins shows this is confined to the exact tangency:

| `u0` | rel. error |
|-----:|-----------:|
| 0.01900 | -1.82e-05 |
| 0.01950 | -2.53e-05 |
| 0.01990 | -2.92e-05 |
| 0.01999 | -8.36e-05 |
| 0.02000 | -1.60e-04 |
| 0.02001 | -9.61e-05 |
| 0.02005 | -8.47e-05 |
| 0.02010 | +3.51e-05 |

and the same ladder at deeper caps recovers clean second order:

| `u0` | 128 | 256 | 512 | order |
|-----:|----:|----:|----:|------:|
| 0.0200 | -2.08e-03 | -6.34e-04 | -1.90e-04 | 1.7 |
| 0.0195 | -4.91e-04 | -1.13e-04 | -2.60e-05 | 2.1 |
| 0.0180 | -2.04e-04 | -3.85e-05 | -9.38e-06 | 2.2 |
| 0.0150 | -2.81e-04 | -6.73e-05 | -1.51e-05 | 2.1 |
| 0.0100 | -1.91e-04 | -2.54e-05 | -1.88e-06 | 3.3 |

The cause was measured directly (`tests/diagnostics/`-style probe over the cap,
7514 samples).  The extra fold pair is a sliver of principal extents
**3.6068e-02 x 6.5218e-04** (55:1 aspect, image area 1.5591e-05), so on a grid
whose cell is `rho / bins` it is

| bins | cells across | cells along |
|-----:|-------------:|------------:|
| 64 | 2.1 | 115 |
| 128 | 4.2 | 231 |
| 256 | 8.3 | 462 |
| 512 | 16.7 | 923 |

The row scan corrects the x crossings to second order and integrates the row
width by the midpoint rule in y; both degrade when a row is a couple of samples
wide.  This is the uniform image-plane grid meeting a feature 0.03 `rho` across
— a property of the discretisation, not a missing component, and it converges.

That next step is now implemented; the rest of this section is its result.

## Per-component grid refinement

The lattice spacing is `rho / bins`.  That number is derived from the *source*
and says nothing about the images it has to resolve, which is why a 55:1 sliver
is 2.1 cells across at 64 bins while the rest of the grid is comfortable.
Refining globally costs `k^2` over the whole disk to fix a feature occupying
0.3 % of it.

The fill already knows enough to decide for itself.  `fill_image_component` was
split out of `inverse_ray_cartesian_core` and now returns, alongside the area,
the component's row span and mean row width in cells; `narrow_cells()` is the
smaller of the two, because the row scan resolves x by sub-cell edge corrections
and y by the midpoint rule over row widths, and both degrade once *either*
direction is a few samples wide.  A component under `kComponentRefineTrigger =
16` cells across is refilled on its own lattice at `k = ceil(32 / narrow)`, odd,
capped at 32, with the refined cell count scaled by `k^-2` and its footprint
projected back into the coarse registry.

`k` is odd so that the two lattices partition the plane the same way: coarse
`x = i * incr` and fine `x = j * incr / k` put the coarse cell centre exactly on
fine cell `j = i * k`, and with `k` odd the coarse cell covers exactly the fine
cells within `(k-1)/2` of it.  The projection is then `[ceil(lo/k),
floor(hi/k)]` on rows `j % k == 0`, with no slack — in particular nothing
claimed across a critical curve on a fold partner's side.  An earlier version
used `llround(j/k)` and over-claimed by half a cell, which starved the fold
partner and cost 2.7x at 512 bins.

### Why the footprint has to come back, and why it has to be checked

A component thinner than a cell reaches the coarse lattice **in pieces** — at 32
bins the reference sliver breaks into four — and each piece is seeded
separately.  Refined, every piece recovers the whole component, so without
carrying the footprint back the component is counted once per piece: measured as
a +6.2e-04 overshoot at 32 bins, with three pieces each returning the identical
whole-component area 17.80674501.

The converse failure is worse.  A coarse fill whose extent was decided by a
neighbour's claims is not a description of a component at all; refilled alone on
an empty lattice it is free to leave through that seam and trace the neighbour,
whose area is already in the sum.  On the extreme-magnification geometry in
`test_point_source_safety.py` (`rho = 1.4e-4`, `mu = 9000`, 40 bins) two seeds
survived as 3-cell specks between the claims of a huge image; each refined at
`k = 33` into 2.2e7 fine cells and re-banked the whole image, **doubling** the
magnification to 17954.

Both are the same statement — the counted area must remain a measure over a
disjoint union of cells — and both are enforced there:

- `claim_refined_footprint` returns false, and the refinement is discarded, if
  the projected footprint meets a cell some other fill already owns.  This
  needed the claimed-cell runs to carry an `owner`, which is also what tells a
  fill whether its own extent was decided by a neighbour (`foreign_contact`).
- the refined fill is given an explicit step budget, `8 * coarse_cells * k^2 +
  4096`, and returns NaN if it exceeds it.  Unlike the existing `max_steps`
  safety net this is an expected answer rather than a numerical failure, so it
  is not reported as one.  It is what keeps the runaway above from costing 2.2e7
  cells before being rejected.

The parity guard is the third piece.  Components of `f^-1(D)` are disjoint and
the lattice only puts two of them in contact where their images merge, i.e. on a
critical curve; so a coarse fill that ended against another component's cells is
one member of a fold pair, and `sign(J_seed)` is the boundary condition that
holds the refined fill to its own side without the neighbour's claims to lean
on.  It is applied whenever `foreign_contact` is set, regardless of
`kFoldJacobianThreshold`: the evidence there is the contact, not the size of
`|J|`.

### Two variants that were tried and removed

- Gating refinement on `!foreign_contact` alone.  Fixes the triple, but the
  sliver's fold pair *is* in contact by construction, so the binary gain fell to
  1.4-1.7x and the ladder was not monotone.
- Transferring the foreign claims onto the fine lattice instead of the parity
  guard.  Worse still (1.8x / 1.4x / 1.0x), slower, and still not monotone: a
  foreign coarse cell blocks all `k^2` fine cells under it, which over-blocks
  exactly at the fold interface the refinement exists to resolve.

### Measured

Tangent cusp, reference 3.960888498085, A/B on the same build:

| bins | uniform | refined | gain | cost |
|-----:|--------:|--------:|-----:|-----:|
| 32 | -4.646e-03 | -2.176e-03 | 2.1x | 2.53 -> 2.75 ms |
| 64 | -2.128e-03 | -6.966e-04 | 3.1x | 2.83 -> 3.28 ms |
| 128 | -5.251e-04 | -2.302e-04 | 2.3x | 3.91 -> 4.66 ms |
| 256 | -1.600e-04 | -6.401e-05 | 2.5x | 7.67 -> 9.37 ms |
| 512 | -4.788e-05 | -2.102e-05 | 2.3x | 22.57 -> 25.59 ms |

The ladder is monotone and stays below the reference throughout.  Interpolating
the uniform ladder to equal error puts the refined grid at **~2.3x cheaper at
fixed accuracy**.  Limb-darkened `c = 0.5` (converged reference 3.836158013)
moves the same way: 128 bins 3.834896 -> 3.835601, 256 bins 3.835778 ->
3.836007.

The triple five-image cap, reference 17.500498641, improves at every resolution
and stays monotone: 32 -2.807e-04, 48 -1.256e-04, 64 -7.062e-05, 96 -3.232e-05,
128 -1.662e-05, 192 -8.882e-06, 256 -6.180e-06, 512 -1.463e-06.  It reaches the
Cartesian core through its own seed set and image map, so it is the check that
this is a property of the fill and not of the binary caller.

Geometries with no thin component are **bit-identical** with the refinement on
and off: `wide_equal_mass` 1.644285791543 / 1.644310999230, `close_binary`
12.302093936985, `planetary` 17.450204716764, `caustic_crossing`
5.366034638431, `far_from_caustic` 2.177131051261, `triple_clear_a`
7.363689756306 / 7.363822519344.

The gain **saturates**: raising the refinement target from 32 to 128 cells
barely moves the error (128 bins, -2.302e-04 -> -2.259e-04) while the cost rises
sharply (256 bins, 9.34 -> 14.18 ms).  So roughly 55-60 % of what was the
`h^1.7` residual is the sliver's own discretisation and the rest is elsewhere.
`kComponentRefineTarget = 32` is where the curve flattens.

### Effect on the adaptive loop

The adaptive error estimate is the raw difference between the calibrated grid
and half of it.  A more accurate fill at a fixed resolution makes that
difference smaller, and it makes it smaller *honestly* — the estimate remains an
upper bound on the true error in every case measured.  On the tangent cusp:

| `finite_source_reltol` | uniform | refined |
|-----------------------:|--------:|--------:|
| 1e-3 | -2.480e-04, 7.1 ms | -3.155e-04, 4.8 ms |
| 1e-4 | -1.094e-06, 1247 ms | -3.162e-05, 21.6 ms |
| 1e-5 | -1.094e-06, 1239 ms | -1.094e-06, 1243 ms |
| 1e-6 | -1.094e-06, 1241 ms | -1.094e-06, 1243 ms |

Every row honours the tolerance it was given.  The 1e-4 row is the whole point:
the uniform grid missed its 1e-4 budget at 400 bins and escalated straight to
the 4096 cap, delivering 1e-6 for 1.2 s of work; the refined grid meets the
budget at 400 bins and stops, **58x faster**.  The regression suite as a whole
went from 695 s to 246 s for the same reason.

`test_adaptive_precision_reaches_the_reference` asserted 1e-5 at a 1e-4 request
and had been passing on that accidental over-delivery.  It is now
`test_adaptive_precision_meets_the_tolerance_it_is_given`, parameterised over
1e-3/1e-4/1e-5, which pins the contract that actually exists — plus the
certificate, so a loose tolerance cannot be met by losing the image pair.

## Certifying the tolerance — the indicator is not the error

Both acceptance tests that ask the adaptive loop for `finite_source_reltol =
1e-5` failed from the day they were written: the binary one raised
`RuntimeError: numerical error` at the `[1e-05]` parameter, the triple one at
both 1e-4 and 1e-5.  The values were never wrong.  What was wrong was the thing
deciding whether to hand them back.

The convergence gate is `diagnostics.estimated_error`, the boundary-area
indicator.  It counts the cells the source limb and the caustic cut, so it is
`perimeter * cell_width / area` — **first order in the cell width, always**,
however fast the integrated area actually converges.  Measured against the
truth on the two acceptance geometries:

| bins | binary value | binary true rel. | indicator | ratio |
|-----:|-------------:|-----------------:|----------:|------:|
| 50 | 3.9572411139 | 9.21e-04 | 1.867e-03 | 0.5 |
| 400 | 3.9607632573 | 3.16e-05 | 2.664e-04 | 2.1 |
| 4096 | 3.9608841648 | 1.09e-06 | 2.650e-05 | 6.1 |

| bins | triple value | triple true rel. | indicator | ratio |
|-----:|-------------:|-----------------:|----------:|------:|
| 50 | 17.4985278858 | 1.13e-04 | 1.563e-02 | 7.9 |
| 400 | 17.5004492993 | 2.82e-06 | 1.955e-03 | 39.6 |
| 4096 | 17.5004986409 | ~1e-14 | 1.909e-04 | >1e6 |

The ratio grows without bound because the value converges faster than first
order and the indicator does not.  At the 4096-bin cap the indicator cannot
certify better than 2e-4 absolute on the triple, so `reltol = 1e-5` (budget
1.75e-4) fails closed on a value good to fourteen digits.  No amount of
refinement fixes this; the ceiling is structural.

Two changes, both in `finite_source_magnifier.cpp`.

**1. A grid pair measures the coarse error, not the fine one.**  The retry loop
already compared the refined value against the one before it, as
`|A_fine - A_coarse|`.  For a first-order scheme that difference is about
`(r - 1)` times the *fine*-grid error, with `r` the ratio of the two bin counts.
Halving gives `r = 2` and needs no correction — which is why the plain
difference was right for the `/2` check that used it.  But the automatic retry
does not halve: on the binary tangency it jumps from 400 straight to 4096, and
the raw difference `|A(4096) - A(400)| = 1.21e-04` was then charged to `A(4096)`,
whose own error is 4.3e-06.  Ten times too pessimistic, and the more the loop
refined the worse the estimate got.  `grid_pair_error_estimate` divides by
`max(r - 1, 1)`, leaving every `r = 2` call site bit-identical.

**2. When the ladder is exhausted, measure instead of bounding.**
`reconcile_with_half_resolution` spends one half-resolution evaluation and lets
the measured pair rule in both directions: a row the indicator passed still has
to survive it (this is the pre-existing anti-aliasing check, unchanged), and a
row the indicator rejected is admitted only if the measurement is inside the
budget.  On the binary it runs after the retry ladder has run out of grid; the
triple Cartesian route has no ladder at all, so it runs there directly.

What this does **not** touch is the part that makes the branch fail closed.
`support_proven` still forces an infinite error, and the `underresolved`
resolvability guard still vetoes — neither is a statement about grid error, so
neither is something a second grid can overrule.  The door still closes: the
binary geometry at `reltol = 1e-6` returns NaN exactly as before.

Result on the two geometries:

| case | before | after |
|:-----|-------:|------:|
| binary, reltol 1e-5 | NaN (indicator 1.21e-04 vs budget 3.96e-05) | 3.960884165, estimate 2.65e-05 |
| triple, reltol 1e-5 | NaN (indicator 1.91e-04 vs budget 1.75e-04) | 17.500498641, estimate 1.13e-06 |
| binary, reltol 1e-6 | NaN | NaN |

Cost is one extra evaluation at half resolution — 25 % — paid only by a row
that carries an explicit tolerance and would otherwise have returned a NaN.

## Seed sets are built once per epoch

`inverse_ray_cartesian_*_mag` and `inverse_ray_polar_*_mag` now take an optional
precomputed seed set, and `triple_mag` builds one lazily per epoch and passes it
to every resolution.  The seeds — point images, caustic probes, the boundary
ring and the certified probes — are a function of the lens and the source disk
only; nothing about them can depend on the grid, so re-deriving them at each rung
of a retry ladder was pure repetition.  The binary path already did this; the
triple did not.

It is worth ~0.7 % on a 60-epoch triple caustic crossing at
`finite_source_reltol = 1e-4` (12.31 s -> 12.24 s at `caustic_bins = 1400`) and
nothing at fixed resolution (2.56 s either way), with bit-identical checksums.
It is kept because it removes a real binary/triple asymmetry, not for the
number.  The equivalent polar A/B was measured and is marginal.

## Tests

- `tests/regression/test_binary_cusp_component.py` — 10 tests: refinement to
  reference (uniform and `c=0.5`), monotonicity across 32..256 bins, adaptive
  precision at 1e-3/1e-4/1e-5 against the tolerance requested, the tangency
  sweep against VBM, and three clear geometries that must be untouched.
- `tests/regression/test_component_refinement.py` — 12 tests: the thin-component
  refinement beats the uniform grid by at least 2x at each of 32..512 bins and
  never overshoots the reference, the ladder stays monotone with decreasing
  increments, the triple cap behaves the same way through its own caller, and
  five geometries without a thin component are unchanged to 1e-12.  The
  over-count guard itself is pinned by
  `test_point_source_safety.py::test_forced_cartesian_high_magnification_does_not_truncate_image_area`,
  which the missing check doubled.
- `tests/regression/test_triple_cusp_component.py` — 7 tests: the same contract
  on the triple lens (refinement to reference uniform and `c=0.5`, monotonicity
  across 32..256 bins, adaptive precision at both tolerances, three clear
  geometries that route through the Cartesian fill and must be untouched).
- `tests/regression/test_smooth_route_clearance.py` — 2 tests: a fixed-seed
  randomised sweep (binary 80x16, triple 24x10) placing the source 0..30 rho
  from a random caustic vertex and asserting that no smooth-expansion route is
  ever chosen for a disk whose certified clearance is below rho.  936 and 168
  smooth routes respectively, 0 violations, 16 s.
- `tests/jax_ir/test_discovery.py` —
  `test_component_certificate_is_independent_of_limb_seed_count` (was
  `test_sixteen_limb_seeds_recover_component_missed_by_eight`, which asserted
  the bug), plus tangent-cusp limb-sample independence at resolution 64/128/256,
  convergence to the reference, and value/JVP agreement across limb samples.

One existing assertion was relaxed, in
`test_lcbinint_auto_nbin_accepts_second_order_smooth_resonant_boundary`.  The
test sweeps 40 epochs across a resonant caustic and asserted that none of them
refines.  One of them (index 8, `t = -0.1049`) has the caustic **0.016 rho** from
the disk centre, so the disk straddles it: the certified support now seeds the
fold pair, `fold_seed_count` goes 0 -> 1, `cartesian_area_error_indicator` stops
applying its smooth-boundary discount, and that epoch spends one refinement (or,
capped at 40 bins, declares itself unconverged instead of claiming success).
The magnifications are bit-identical to the baseline at every bin count and the
accuracy assertion is unchanged, so what moved is the honesty of the error
indicator at a caustic crossing, not the answer.  The test now excludes that one
epoch from the smooth-boundary assertions and checks it on its own terms.

Exit statuses are recorded in the commit message.
