# Stable exceptional gradients — 2026-10-15

## Result and scope

Implemented in experiment/algebraic_boundary, HEAD 1d6984f7501066f22ff0143d69c1cd1c45b04d06 plus the preserved existing changes. Both default native value and gradient modules were rebuilt from this checkout. No notebook edits, new worktrees, branch changes, commits or pushes in this work.

- Small-source case_033 at t=-.15: dmu/drho -150.4200742 → +1.1234968126. Independent source-plane integration using VBM point-source calls gives approximately +1.123494…+1.123500. At +.15 the new derivative is +1.1240657429. The new result remains stable for tol=reltol 1e-3, 1e-4, 1e-5, 1e-7.
- All 120 stress configurations now produce finite gradients at all 3960 epochs: 27720/27720 components, previously 13 invalid epochs (91 NaN components). All their gradient-path value-convergence flags are true.
- The original 60-configuration matrix (10 geometries × uniform/LD .5/LD .9 × tol=reltol 1e-3/1e-4) has 168420/168420 finite components. The ordinary value API is bitwise unchanged at all 24060 epochs; all recorded value convergence/support/seed/node/integrator fields are unchanged.
- These are observed regression results, not a general gradient accuracy certificate. Existing support and gradient-precision certification flags were not promoted.

## Implementation

### Source-disk coordinates for cancellation-prone smooth disks

`src/lcbinint/magnification/holonomic/source_disk_gradient.hpp` evaluates the original finite-source integral in normalized source coordinates. With z=sqrt(1-r^2), its measure is

`2*z*(1-c+c*z)/(1-c/3) dz dtheta/(2*pi)`.

The measure is independent of rho. The rho derivative integrates the point magnification gradient dotted with the normalized source displacement, eliminating the subtraction of two terms of order 2*A/rho. Centre gradients are subtracted before the angular first moment; their exact contribution is zero. Point-image derivatives come from an analytic real 2x2 lens Jacobian solve, with q/s and centre-of-mass transformations included. This is not a truncated point-source or single-lens approximation.

The cheap route-selection tests use inverse-image conditioning/thin source geometry and proximity to the symmetry axis. They choose coordinates, not an accuracy bound. The continuous-caustic clearance certificate must additionally prove that the entire disk is smooth. Ordinary entry requires 32 source radii of clearance. Invalid/retried geometry can request a less restrictive but still strictly nonintersecting disk. If the ordinary caustic cover is too broad, only this repair path constructs a cached finer 512-interval cover. Default enclosure settings remain 32 intervals/depth 4 for existing callers.

Radial GL2/4/8/16 and periodic angular rules refine only by magnification differences against the supplied atol/rtol. No gradient difference, gradient tolerance, or gradient-error adaptivity is introduced. Invalid geometry, changed image counts, nonfinite samples, or failed primal convergence return to the existing path. No negative brightness is clamped.

### Exact stationary-root fallback

`benchmarks/holonomic/fused_gradient/fused_geometry_exact.hpp` is called only if the existing double/quad interval Bernstein/Sturm isolation cannot finish. It reconstructs the D14 derivative at the original dyadic PrimaryFrame input with exact rationals, then uses primitive integer pseudo-remainder Sturm chains. All sign decisions and root counts are exact; bisection remains bounded. Integer homogeneous Horner evaluation avoids repeated rational gcd work. Ordinary successful detection does not execute this fallback.

### Boundary precision and geometry retry

`fused_phi.inc` re-evaluates near-zero boundary brightness and its analytic derivatives in binary128, preserving the radial low part. A genuinely negative result remains a failure. Normal brightness evaluations use the existing arithmetic.

If a deferred Gaussian derivative replay cannot evaluate a root near a fold endpoint, it can retry a fixed 14-point interior rule within the remaining evaluation budget. This is a geometry-validity retry, not derivative-error refinement. It is recorded internally by `reliability_retried`. When a smooth disk can be certified, the source-plane result replaces such a retry. This distinction matters: case_081 produced finite but inaccurate values with the interior rule alone; the certified source-plane replacement fixes that counterexample. Merely making outputs finite was not accepted as correctness evidence.

## Accuracy checks and limits

- `check.py`/`fd.json`: 12 selected configurations, four parameters, three epochs, three central-difference steps. They preserve each reference convergence flag and step variation.
- `repaired_fd.py`: repaired cases_004,049,058,081,109. Cases_004/049 agree with stable value finite differences. For case_081, u0 derivative -365.305655 agrees with finite-difference sequence -365.306707,-365.305353,-365.305221; rho derivative 17.967796 agrees with 17.967944,17.967300,17.967095 within the reference's observed numerical variation.
- case_058's strict value references do not report convergence, so its recovered finite derivative is not independently certified. Its smaller-step differences approach the recovered values, but those are diagnostics only.
- case_109 is extremely close to source-edge/caustic contact; rho finite differences vary strongly with step. Its finite derivative is stable across tighter primal tolerances, but no step-independent external precision reference was established. Do not equate the removed NaN with an accuracy proof.
- `api_check` passes coexistence, analytic limit and invalid-input checks.
- All requested options remain atol/reltol only. No scientific support/precision flags were raised.

## Timing

Normal `LightCurve.holonomic_gradient` call, 401 uniformly spaced epochs, LD=.5, tol=reltol=1e-3. One pinned CPU, BLAS/OpenMP one thread, warmup then median of three calls, setup excluded. Baseline module retained in before.so. All individual repetitions are saved.

| Geometry | Before ms | After ms |
|---|---:|---:|
| ordinary |54.82|54.85|
| wide caustic |65.40|65.63|
| equal mass |111.41|113.70|
| broad source |410.39|419.73|
| large source |259.90|260.18|
| tiny source |172.09|64.28|
| notebook alpha=180 radians |101.64|104.76|

The other measured configurations are also in summary.json; ordinary overhead is roughly 0–3%, while the tiny-source example is about 2.68x faster. Exceptional failed-geometry epochs can cost more: this is intentionally confined to the failures. These are gradient timings, not the earlier 156-ms value-only large-source timing. The prior value-only speed improvement remains active.

## Reproduction and provenance

1. `TMPDIR=$PWD/build-algebraic-boundary/fused-gradient/scratch cmake --build build-algebraic-boundary/python-api -j 2`
2. `bash benchmarks/holonomic/fused_gradient/build_algebraic.sh`
3. With `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1`, run matrix.py, sweep_gradients.py, timing.py, check.py and repaired_fd.py in this evidence directory. These scripts explicitly load this checkout's source/default native modules.
4. Run `build-algebraic-boundary/fused-gradient/api_check` and `git diff --check`.
5. Run analyze.py to assert the aggregate counts and write summary/provenance/source snapshots.

provenance.json records HEAD, source/module hashes and timing boundaries. source_snapshot/ retains all relevant final code, including the earlier dirty angular and continuation fixes. Intermediate rejected/partial experiments are retained with their specific logs; only the final summary/provenance and delivery_* logs describe the delivered build. Restart an existing Jupyter kernel before using the rebuilt extension.
