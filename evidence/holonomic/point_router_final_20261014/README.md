# VBM point-classifier audit and shared predictor refinement

The measured reference is the installed VBMicrolensing 5.5 package. It ships
its C++ source next to the extension; `provenance.json` records that source's
path and hash, together with both measured lcbinint native hashes. Public
upstream source: https://github.com/valboz/VBMicrolensing/blob/main/VBMicrolensing/lib/VBMicrolensingLibrary.cpp

BinaryLightCurve calls BinaryMag2 per epoch. BinaryMag2 first computes the
centre point value via BinaryMag0/NewImages, including derivative and ghost
diagnostics. Point acceptance uses all three conditions:

- corrquad scaled by 6*(rho^2+1e-4*Tol) is below Tol;
- corrquad2 scaled by 256*(rho^2+1e-8) is below 1;
- safedist is greater than 4*rho^2.

The small-q safedist condition is a planetary-caustic-location approximation;
otherwise safedist starts at 10. The point classifier neither constructs a
global caustic curve nor uses the new continuous strip bound. A failed point
test calls BinaryMagDark, the limb-darkened contour integrator. Consequently,
removing lcbinint's formerly necessary caustic preparation primarily removed
an extra cost relative to VBM; it did not remove VBM's quintic-root work.
This audit concerns the shipped 5.5 source, not every future VBM release.

The installed source locations are BinaryMag0 at line 716, BinaryMag2 at 1170,
the NewImages quintic solve at 1517, and BinaryLightCurve at 5664.

## Further shared optimization

The binary image predictor previously used complex absolute values for source,
lens-pole and image-step checks. Replace it by the real inverse lens Jacobian,
reuse the cached image derivative when available, and compare squared norms.
The step limits, polynomial/Vieta checks and lens-residual/cold-retry guards
remain. This common predictor is used by centre roots, hex/ring samples and
image-seed preparation; neither integration algorithm was changed.

## Scalar-only measurements

Same ordinary point-only trajectory, uniform source, atol=rtol=1e-3,
one Xeon Gold 6530 CPU thread, Release O3, double precision, 101 repetitions.
All epochs choose point. Each lcbinint call clears its preparation; constructors
and code warmup are excluded. No gradients or diagnostics in timed calls.
VBM uses BinaryLightCurve directly, returning its ordinary list. lcbinint public
timings include request validation and normal NumPy output. Native and public
results agree exactly; scalar VBM differences remain at double rounding scale.

| Epochs | Before public ms | After native ms | After public ms | VBM ms |
|---|---:|---:|---:|---:|
| 201 | 0.182 | 0.163 | 0.172 | 0.175 |
| 1001 | 0.802 | 0.744 | 0.752 | 0.816 |
| 2001 | 1.575 | 1.471 | 1.479 | 1.626 |

The 201-point difference versus VBM is only a few microseconds; describe it
as comparable, not a robust win. The denser trajectories show a clearer gap.
Matching linear LD=.5 gives public/VBM 0.172/0.175, 0.752/0.818,
and 1.479/1.611 ms at the three respective counts. These are measurements
of these trajectories, not all point-only configurations.

## Integration-path validation

All 16 previously measured configurations retain exactly identical method,
full/value convergence, seed/integral support masks and stop labels. Maximum
relative before/after value change is 1.17e-12. Integration-containing curves
improve 1.01--1.04x versus the preceding strip-bound version. Representative
uniform diagnostic timings at atol=rtol=1e-3:

| Case | Before ms | After ms |
|---|---:|---:|
| wide planetary | 5.331 | 5.269 |
| resonant | 5.429 | 5.238 |
| close | 6.683 | 6.521 |

Mixed diagnostic calls retain existing uncertified-support epochs; the scalar
API still rejects those curves. Native unit_core passes; 46 selected native/API
regressions pass. One optional JAX test is deselected because FFI is disabled
in this benchmark build. Logs and raw samples are stored alongside this report.

Run from the experiment checkout:

```bash
export TMPDIR="$PWD/build-seeded-events"
cmake --build build-seeded-events/python-api --parallel 2
python benchmarks/holonomic/bench_point_value.py --native-dir build-seeded-events/python-api --output evidence/holonomic/point_router_final_20261014/point_optimized.json
python benchmarks/holonomic/bench_router_shared_point.py --native-dir build-seeded-events/python-api --output evidence/holonomic/point_router_final_20261014/optimized.json --repetitions 21
ctest --test-dir build-seeded-events/python-api --output-on-failure
python evidence/holonomic/router_shared_point_20261014/validate.py
```

The immediately preceding native build was copied before this edit into the
ignored project build directory `build-seeded-events/point-router-final/baseline`.
Pass that directory to the same benchmark scripts to reproduce baseline calls.
