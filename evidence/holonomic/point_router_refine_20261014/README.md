# Point-route refinement, baseline 88f7de2c

The point-only ordinary trajectory now has essentially the same native runtime
as VBMicrolensing. This is a fresh-preparation comparison, not geometry reuse
across repeated calls. The public Python validation/wrapper adds about 9 us for
201 epochs. The optimizations are in the shared binary root and routing paths;
the seeded radial and inverse-ray integration algorithms are unchanged.

## Scalar-only matched measurements

Xeon Gold 6530, one CPU thread, Release O3, double precision, VBMicrolensing 5.5.
Static ordinary trajectory: s=1.2, q=.01, u0=.3, alpha=0, rho=.003, t0=0, tE=1,
epochs uniformly spaced in [-1,1], atol=rtol=1e-3. All epochs select point.
101 repetitions in rotating order. Constructor and code warmup are excluded;
the native scalar API clears geometry/seed preparation once per curve.
No gradients, diagnostics or per-epoch timers run inside these timed calls.
VBM is invoked through BinaryLightCurve and returns its native Python list;
lcbinint returns its normal NumPy array. Reference checks are outside timing.

| Epochs | Linear LD | Before public ms | After native ms | After public ms | VBM ms |
|---|---:|---:|---:|---:|---:|
| 201 | 0 | 0.248 | 0.173 | 0.182 | 0.172 |
| 201 | .5 | 0.247 | 0.173 | 0.182 | 0.177 |
| 1001 | 0 | 0.942 | 0.793 | 0.803 | 0.797 |
| 1001 | .5 | 0.943 | 0.793 | 0.803 | 0.817 |
| 2001 | 0 | 1.806 | 1.566 | 1.579 | 1.635 |
| 2001 | .5 | 1.807 | 1.565 | 1.577 | 1.633 |

These are observed medians for these inputs, not a guarantee for every
point-only trajectory. Maximum relative scalar difference from VBM: 4.33e-15.
`point_baseline.json` and `point_optimized.json` retain all samples and hashes.

## Why preparation can be skipped safely

Let the real lens positions be a and 0, with normalized masses m1+m2=1,
separation s=|a|, and z=x+iy on a critical curve. Set
A=m1/|z-a|^2 and B=m2/|z|^2. The critical equation |kappa|=1 gives
|A-B|<=1<=A+B. The lens map has Im(w)=y(1-A-B), so

    |Im(w)| <= 2 min(m1 |y|/|z-a|^2, m2 |y|/|z|^2).

For t=sqrt(m1*m2)/s, the two sets m_i |y|/|z-a_i|^2>t are disjoint
open disks in either half-plane: their radii are m_i/(2t), and the centre
separation equals the sum of those radii. Thus the minimum is <=t everywhere.
Every continuous caustic therefore lies inside the strip

    |Im(w)| <= 2 sqrt(m1*m2)/s.

Real frame shifts and lens-mass swapping preserve the strip. Its distance
lower bound is padded for double rounding. For the ordinary trajectory the
half-width is about 0.165, while the source remains at |y|=.3. The proven
clearance exceeds the existing 10-rho gate, with no critical quartic solves.
This witness only replaces the topology veto: the same strict 30x local
derivative bound, requested error budget and source-radius gate still apply.
If the strip cannot prove clearance, retain the continuous disk enclosure
and detailed-caustic/hex/integration fallback.

The fallback disk enclosure uses conjugation symmetry to prove half the
phase circle and reflect it; a cached clearance ball uses the 1-Lipschitz
distance property. Padded squared-distance pruning skips disks that cannot
improve the minimum. Failed proofs remain conservative.

## Shared root-path changes

- Retain physical-root selection and residuals from warm-root validation for
  the following safety/scalar pass. The same polynomial/Vieta and lens-residual
  guards remain.
- Share reciprocal lens displacements across derivative orders and the ghost
  map; cache the resulting image Jacobian for the trajectory predictor.
- Specialize the degree-five Newton Horner loop while preserving the SG Adams
  stopping rule, final Newton step and fractional jumps. Fall back to the
  original SG routine on unusual/failed iterations.
- Avoid constructing an unused local point magnifier when one is supplied;
  move routing explanations instead of copying their strings.

## Integral-route checks

The previous 16 configurations (four trajectories, LD 0/.5, equal atol/rtol
1e-3/1e-4) have exactly identical selected methods, value/full convergence,
seed/integral support masks and stop labels. Maximum relative before/after
value change is 7.48e-13. None regressed in measured median runtime.
Mixed cases are diagnostic timings because some existing epochs have
uncertified support; the scalar API still rejects those curves.

| Uniform, atol=rtol=1e-3 | Before diagnostic ms | After diagnostic ms | VBM ms |
|---|---:|---:|---:|
| wide planetary | 5.741 | 5.323 | 4.607 |
| resonant | 5.526 | 5.431 | 7.633 |
| close | 6.754 | 6.678 | 7.538 |

21 rotating/alternating repetitions, original uninterrupted trajectories.
`baseline.json` and `optimized.json` include every epoch and all samples;
`provenance.json` checks the exact masks and identifies source/native binaries.
`enclosure.json`, `shared_safety.json`, and `point_*` intermediate results
retain the progression. Scratch trials with finer phase sectors or ordinary
quartic warm starts were not adopted.

Validation: native unit_core passes, including 320 changing-geometry/mass-swap
warm-root checks against cold starts. 46 native/API regressions pass, including
independent NumPy critical quartics (13 geometries), calibrated ghost cases,
near-caustic route checks, reversed/jumping trajectories, scalar fail-closed
support and inverse-ray tests. One optional JAX FFI test is deselected because
this benchmark build explicitly has JAX FFI disabled.

## Reproduce

Run from the experiment checkout:

```bash
export TMPDIR="$PWD/build-seeded-events"
cmake --build build-seeded-events/python-api --parallel 2
python benchmarks/holonomic/bench_point_value.py --native-dir build-seeded-events/python-api --output evidence/holonomic/point_router_refine_20261014/point_optimized.json
python benchmarks/holonomic/bench_router_shared_point.py --native-dir build-seeded-events/python-api --output evidence/holonomic/point_router_refine_20261014/optimized.json --repetitions 21
ctest --test-dir build-seeded-events/python-api --output-on-failure
python evidence/holonomic/router_shared_point_20261014/validate.py
```

The baseline extension was copied from 88f7de2c before edits into the ignored
project build directory. To regenerate baseline results after rebuilding it,
pass that extension directory to the same benchmark commands.
