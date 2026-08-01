# Binary finite-source inverse-ray cusp investigation

## Scope and reproduction

This note records observations only.  It does **not** propose or apply a
numerical fix.

Reproduction geometry (center-of-mass coordinates):

| field | value |
| --- | ---: |
| `source_x` | 0.653 |
| `source_y` | 0.020 |
| `separation` | 1.2 |
| `mass_ratio` | 0.1 |
| `source_radius` | 0.020 |

The supplied uniform-source reference is `3.960889`.  All JAX measurements
below use the direct Cartesian inverse-ray entry point, a 16-cell tile, and
sufficient tile capacity; they do not use a source-plane evaluator.

## Result: both implementations are affected

### Native Cartesian inverse ray

Uniform-source measurements, `inverse_ray_grid="cartesian"`:

| source bins | magnification |
| ---: | ---: |
| 32 | 3.939794855 |
| 64 | 3.945871028 |
| 128 | 3.947677569 |
| 256 | 3.948320749 |
| 512 | 3.948482292 |
| 1024 | 3.948533276 |

At 1024 bins the absolute error against the supplied reference is about
`1.236e-2`.  `LCBININT_AREA_DIAGNOSTICS=1` reports exactly three snapped and
processed seeds at every listed resolution.  Its reported estimated error
nevertheless falls from `6.85e-4` (32 bins) to `6.99e-7` (1024 bins).

This is direct evidence that the native error indicator can certify the
discretization of an incomplete support; it is not an error estimate for the
missing image component.

### JAX Cartesian inverse ray

At a fixed resolution of 128, uniform-source results depend discontinuously
on the source-limb discovery sampling:

| limb samples | magnification | discovered tiles |
| ---: | ---: | ---: |
| 16 | 3.944162788 | 1303 |
| 32 | 3.944162788 | 1303 |
| 64 | 3.944162788 | 1303 |
| 128 | 3.957513553 | 1341 |
| 256 | 3.961093348 | 1356 |

The same rows through the compiled Cartesian FFI are equal to the pure JAX
rows to roughly `7e-12`.  Therefore this failure is shared by the JAX
discovery contract and its FFI implementation, rather than being a JAX
autodiff-only effect.

With 256 limb samples held fixed, uniform-source refinement gives:

| resolution | magnification |
| ---: | ---: |
| 16 | 3.970681723 |
| 32 | 3.962666061 |
| 64 | 3.961373793 |
| 128 | 3.961093348 |
| 256 | 3.960643883 |
| 512 | 3.960702028 |

The values are much nearer the supplied reference once the extra component is
discovered, but the last two rows are not monotone.  Thus this sequence must
not be treated as a validated convergence proof.

## Existing source-plane fallback is not safe for this case

It must not be used as the remedy for this defect.

The native automatic path classifies the reproduction as `caustic_enters_disk`
because the cached scan has an inside vertex and a crossing.  Its grazing-only
source-plane branch is therefore deliberately not selected; it proceeds to
the incomplete inverse-ray support instead.  This is appropriate in the sense
that the fallback's documented precondition is a smooth point-source field
over the disk, which does not hold here.

The JAX chord rule was evaluated directly for the uniform profile:

| coarse/fine order | magnification | estimated difference | converged |
| ---: | ---: | ---: | --- |
| 48 / 96 | 3.958311799 | 1.851e-4 | false |

The result is still `2.58e-3` below the supplied reference and correctly does
not claim convergence at a `1e-6` absolute/relative budget.  A direct
160/256 attempt exceeded the available memory (about 594 GB requested by XLA
in this environment).  Lower-order ring evaluation is also not converged
(`3.936629677`, estimated difference `1.025e-2`).

Thus the fallback has a useful fail-closed outcome at strict tolerance here,
but it is neither an accurate accepted result nor a feasible high-order
resolution route for this cusp case.  It cannot establish safety for automatic
use near a caustic crossing.

## Independent VBBinaryLensing check

The installed `VBBinaryLensing` package was called directly with the same
`(s, q, x, y, rho) = (1.2, 0.1, 0.653, 0.020, 0.020)` inputs.  It is working
for this case and converges as its `Tol` is tightened:

| Tol | uniform | uniform seconds | linear `c=0.5` | linear seconds |
| ---: | ---: | ---: | ---: | ---: |
| 1e-3 | 3.960818897605 | 0.00016 | 3.836727702723 | 0.00112 |
| 1e-5 | 3.960888138495 | 0.00050 | 3.836264547181 | 0.02668 |
| 1e-7 | 3.960889170813 | 0.00378 | 3.836256795465 | 0.72392 |

The uniform `1e-7` result agrees with the supplied reference to about
`1.7e-7`.  This establishes a usable independent value reference for both
profiles.  It does not by itself reveal the package's internal algorithm, so
we should not infer an implementation strategy from the numerical agreement
alone.

For linear limb darkening with `c=0.5`, again with 256 limb samples:

| resolution | magnification |
| ---: | ---: |
| 32 | 3.837325931 |
| 64 | 3.836483389 |
| 128 | 3.836294571 |
| 256 | 3.836017670 |
| 512 | 3.836051089 |

No independent high-precision linear-darkening reference was supplied or
generated for this investigation.  Consequently these numbers establish the
same non-monotone behavior, not linear-profile accuracy.

## Gradient observation

Pure-JAX gradients at 128, 256, and 512 resolution are not stable in this
case, even after forcing 256 limb samples.  For the uniform profile, the
`source_y` derivative is `-4042.13`, `-4711.37`, and `-4999.31`; the
`source_radius` derivative is `3940.28`, `4609.68`, and `4897.71`.
The linear (`c=0.5`) sequence has the same behavior (`source_y`: `-2575.71`,
`-2975.23`, `-3148.22`).  The support/discretization is therefore not yet
suitable for a resolution-independent gradient claim.

## Exact discovery failure

The native pipeline was instrumented temporarily at raw-seed generation and
after lattice snapping, then rebuilt locally.  At 128 bins it reports:

```text
cached-caustic scan: min_distance = 0.0199917500 < rho,
                     any_vertex_inside = true, crossings = 1
raw seeds before snapping: 3
snapped seeds:             3
```

Therefore the native component is **not** being generated and then lost by
lattice snapping in this reproduction.  It is absent before snapping.

The reason is now identified in the native probe policy.  Its cached-caustic
path calls `append_caustic_probe_image_seeds` around the detected caustic
point with the smallest radial offset `0.02*rho`; its boundary ring is at
`0.98*rho`.  Every one of those probes has only three point images.  Direct
sampling of the source limb shows that the extra image pair exists only in a
much thinner annulus (the tested `0.9999*rho` ring contains five-image points,
whereas the `0.999*rho` ring does not).  Consequently neither native probe
family ever produces a raw seed for this component.

JAX has the homologous failure, but via its own policy: discovery samples only
the centre plus `limb_samples` points exactly on the limb.  At 64 samples its
sample phases miss the short five-image arc; at 128 samples they first enter
it, which accounts for the abrupt support increase from 1303 to 1341 tiles.

The native and JAX mechanisms are thus distinct in implementation but the
same in kind: a fixed, non-certified source-plane probe set is being used as
the completeness criterion for image support.

## What the code establishes

1. JAX discovery seeds only the source centre plus a uniformly sampled source
   limb (`python/lcbinint_jax/discovery.py`).  A component absent from all
   samples is absent from the macro-tile BFS support.
2. The native inverse ray uses point/caustic/boundary seeds, then snaps them
   to its Cartesian lattice (`lattice_snapped_seeds` in
   `src/lcbinint/magnification/finite_source_magnifier.cpp`).  The diagnostic
   count above shows that this path retains only the ordinary three components
   for the reproduction case.
3. Neither Cartesian lattice is nested under a twofold refinement: the JAX
   integrator evaluates cell centres at `(i + 0.5) h`, while the native path
   uses a grid spacing `h = rho / nbin` and seed snapping tied to that spacing.
   A coarse/fine difference can therefore be small while both supports omit
   the same component.

## Conclusions and required acceptance conditions

The confirmed defect is **support incompleteness before quadrature**.  Raising
the integration resolution alone cannot detect or bound it.  A fixed increase
in limb samples merely changes the probability of discovery and is not a
solution.

A real fix must satisfy all of the following before it can be accepted:

1. component discovery has a deterministic completeness criterion independent
   of an arbitrary fixed limb-sampling count;
2. any component newly found at a finer level invalidates convergence of the
   coarser support;
3. the refinement estimator includes a support-change term, so shared omitted
   support cannot be reported as converged;
4. native and JAX/FFI implement the same contract;
5. uniform and linear profiles are checked against independent references,
   including value, refinement sequence, and gradients; and
6. when these conditions cannot be established within the requested budget,
   the existing fail-closed behavior leaves the value unaccepted.

This diagnosis establishes the first failure point for the reproduction:
native raw-seed discovery and JAX limb discovery.  It does not yet establish
whether a newly discovered near-critical native seed can always be represented
by the existing Cartesian lattice; that is a separate acceptance test for a
future fix, not the cause of the current omission.

## Implementation checkpoint (2026-08-01)

The contact path now uses a geometry-derived local component witness and a
deduplicated tile support rather than a row walk.  With 16 limb samples (which
do not hit the cap), the native uniform sequence is
`3.961383, 3.961087, 3.960979, 3.960935` at 64--512 cells; the linear
`c=0.5` sequence is `3.837031, 3.836486, 3.836291, 3.836216`.  The JAX CPU
sequence is `3.962666, 3.961374, 3.961093, 3.960900` at 32--512 cells with
sufficient tile capacity.  The remaining acceptance item is the full
component-change certificate across refinement and tighter stability of all
parameter derivatives; neither is claimed complete by these figures.
