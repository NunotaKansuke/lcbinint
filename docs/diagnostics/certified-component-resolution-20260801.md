# Certified component support — resolution of the 20260801 handoff

Branch `codex/certified-component-tile-kernel`.  Supersedes
`certified-component-tile-kernel-handoff-20260801.md`, which described the
problem and an unfinished separate tile integrator.  The tile integrator has
been removed; the fix is in the seeding, not in a second kernel.

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

Adaptive precision (`finite_source_reltol` 1e-4 and 1e-5 agree to all digits):
**3.960884165, -1.09e-06 relative, 1.25 s**.

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

Note also that the JAX timings and values are now identical for
`limb_samples` 8, 16 and 32 — the certificate has made the limb raster in
`discover_cartesian_support` redundant, which is the next thing to remove.

## Convergence order — what is left, and why it is not a defect

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

The identified next step, if this ever needs to be cheaper: derive the cell size
for a component from that component's own measured extent (`k = ceil(N_target *
rows / cells)` from a base-resolution pass, refill at `h / k` with its own
claimed-cell registry, scale the returned cell count by `k^-2`).  The sliver is
0.3 % of the grid, so refining only it costs ~45 % more at `N_target = 128`
where refining globally to the same accuracy costs ~10x.  Not implemented: the
component is now correct and converging, and the change touches the claimed-cell
ownership invariants.

## Tests

- `tests/regression/test_binary_cusp_component.py` — 8 tests: refinement to
  reference (uniform and `c=0.5`), monotonicity across 32..256 bins, adaptive
  precision to 1e-5 relative, the tangency sweep against VBM, and three clear
  geometries that must be untouched.
- `tests/regression/test_triple_cusp_component.py` — 7 tests: the same contract
  on the triple lens (refinement to reference uniform and `c=0.5`, monotonicity
  across 32..256 bins, adaptive precision at both tolerances, three clear
  geometries that route through the Cartesian fill and must be untouched).
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
