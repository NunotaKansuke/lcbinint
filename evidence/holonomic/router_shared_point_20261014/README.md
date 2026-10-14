# Shared binary routing preparation, 2026-10-14

Existing dirty work was retained; this is a before/after measurement of this checkout, not a production-wide performance guarantee.

## Implementation

- Solve the centre roots once and retain magnification, physical-image selection, derivative and topology diagnostics. Reuse the image Jacobian for the next trajectory predictor. Materialize centre-image seeds only after point/hex routing falls through to integration; the augmented seed generator is unchanged.
- Polish retained quintic roots independently instead of deflating the polynomial at every trajectory/hex/ring sample. Keep the same polynomial residual, full-set Vieta and lens-residual checks, with the original cold solver on failure.
- Cover every critical-curve phase with disjoint root disks using Rouche bounds. Bound the rational lens map over each disk, including coefficient reconstruction error and double-roundoff padding. An unsuccessful cover returns zero clearance and falls back to detailed caustics. Point acceptance still requires the strict 30x derivative bound and at least max(10, hex_threshold) source radii of proved clearance; ghost/cusp/planetary constants are unchanged.
- Add `curve.holonomic_magnification(times, params)` for scalar-only values, with geometry/seed caches cleared once per call and the ordinary convergence/support contract. `lightcurve_value` remains the diagnostic API and retains uncertified last iterates. Diagnostic integer arrays are returned as NumPy buffers instead of Python lists.

## Matched benchmark

Xeon Gold 6530, one thread, 201 epochs, 15 alternating repetitions, no gradients. Both methods use equal absolute/relative tolerance and matching linear LD; VBM 5.5 is called through BinaryLightCurve. Construction, compilation and one code warmup are excluded. The diagnostic API clears geometry every call. Scalar baseline cold measurements use fresh OS threads to empty thread-local geometry; the new scalar API clears that geometry itself. The scalar warm measurement deliberately reuses the same lens geometry.

Point-only ordinary case (`s=1.2, q=.01, rho=.003, u0=.3, alpha=0`, atol=rtol=1e-3, uniform):

| Path | Before ms | After ms | VBM ms |
|---|---:|---:|---:|
| scalar, fresh geometry | 1.709 | 0.239 | 0.183 |
| scalar, retained geometry | 0.343 | 0.197 | 0.183 |
| full diagnostics, fresh geometry | 1.696 | 0.259 | 0.183 |

The scalar API remains about 1.3x the VBM time with fresh geometry, and about 1.1x with geometry retained. Both return point-source values; maximum relative difference against matched VBM is 3.7e-15.

Other full-curve timings below include identical native diagnostics before/after. Mixed radial cases have existing uncertified-support epochs, so their scalar API correctly rejects the curve rather than hiding those flags. VBM returns values only.

| Case (uniform, atol=rtol=1e-3) | Before ms | After ms | VBM ms |
|---|---:|---:|---:|
| wide_planetary | 6.034 | 5.729 | 4.604 |
| resonant | 5.954 | 5.522 | 7.616 |
| close | 7.260 | 6.762 | 7.533 |

Across all 16 configurations (four trajectories, uniform/linear LD, 1e-3/1e-4), methods, convergence, support masks and stop labels match the baseline exactly. Maximum relative value change is 1.17e-12. The three integration-containing trajectories show 1.05–1.12x full-curve speedups (about 5–11% less time). Matched-tolerance VBM differences stay within the requested relative tolerance in this matrix; this is not a higher-precision reference certification.

The full public Python API, including request validation and NumPy conversion, was also timed separately in 21 alternating repetitions: **0.249 ms** for `curve.holonomic_magnification`, **0.186 ms** for VBM BinaryLightCurve (uniform, fresh geometry). Linear LD gives 0.249 / 0.185 ms. `public_api.json` records the samples.

## Validation and reproduction

Native core test passes. The Python selection runs 39 native/API regression tests, including independent NumPy critical-quartic phases, calibrated ghost counterexamples, and 1280 caustic-near routing trials. One optional JAX warmup test is deselected: it was attempted, but this matched benchmark build explicitly disables the required CPU FFI backend.

```bash
export TMPDIR="$PWD/build-seeded-events"
cmake --build build-seeded-events/python-api --parallel 2
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmarks/holonomic/bench_router_shared_point.py --native-dir build-seeded-events/python-api --output evidence/holonomic/router_shared_point_20261014/optimized.json --repetitions 15
ctest --test-dir build-seeded-events/python-api --output-on-failure
python evidence/holonomic/router_shared_point_20261014/validate.py
```

`baseline.json` and `optimized.json` retain every epoch and runtime sample. `provenance.json` identifies the source/compiled binaries and exact comparisons. `candidate*.json` retain intermediate timings; an attempted detailed-quartic continuation was discarded because its validation cost offset its benefit.
