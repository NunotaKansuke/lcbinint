# Forward fused gradients (isolated experiment)

This experimental entry point carries derivatives for `(xs, ys, rho, q, a)`
while computing the magnification. It does not call a VJP or rerun the
solver for each parameter. The main selector has an opt-in diagnostic route
for comparing the seeded radial-frontier integrator; default dispatch remains
unchanged.

`build_material_overlay.py` creates a consistent copied header tree under
`build-seeded-events/fused-gradient/include`. Exact replacement assertions
reject an incompatible source revision. `evaluate.hpp` requires the
experimental build macro and defaults to `Refinement::ValueMesh`.

The fused route shares frontier counts and prepared fold seeds, transports
the motion of the two radial fold endpoints, and differentiates the angular
quadrature at its existing nodes. The nested angular ladder evaluates only
new nodes. Uniform and linear limb-darkening terms use separate product
weights on the same radial mesh.

ValueMesh uses only the value tolerance to select angular/radial work. All
five derivatives are accumulated in the forward quadrature, but derivative
error estimation and derivative-only refinement are skipped. The result's
`gradient_precision_checked` and `all_gradient_tolerances_met` are false;
the underlying derivative error fields contain an unresolved sentinel, not
a measured accuracy bound. Some existing failure paths can return invalid
derivatives. Callers must inspect the result status. `JointTolerance` remains
an optional diagnostic mode and is not the low-overhead default.

From the experiment checkout:

```bash
bash benchmarks/holonomic/fused_gradient/build.sh
build-seeded-events/fused-gradient/baseline INPUT baseline.tsv 3 2
build-seeded-events/fused-gradient/fused INPUT fused.tsv 3 1
```

The benchmark excludes caustic prebuild, includes fresh seed/index
preparation, and alternates value/fused order with independent workspaces.
Argument 2 in the last position selects value only; 1 selects value and
ValueMesh. Three repetitions produce per-input median timings. Inputs use
the saved paper benchmark format; its reciprocal mass-ratio convention is
converted once when constructing LensParams.

Evidence and executable/backend hashes are in
`evidence/holonomic/fused_gradient_20261014`. `audit.py` compares every primal
result, stopping status and node count against the original value build,
then reports paired timing ratios. Saved finite-difference and point-lens
limit probes are preliminary diagnostics; derivative accuracy is not a
release claim in this experiment.

## Optional compiled API

The same build script now produces `liblcbinint_fused_gradient.a`. Callers
include `lcbinint/magnification/holonomic/fused_gradient.hpp` from the usual
source include directory and link that archive alongside the native backend.
They do not need the generated include directory or the experiment macro.
The compiled kernel uses a private namespace, so normal value headers and
the optional gradient API can safely be used in the same program.

```cpp
// primary_image_seeds contains image positions in the primary lens frame.
auto result = lcbinint::holonomic::epoch_fused_gradient(
    params, limb_darkening, primary_image_seeds);
// result.mu and result.grad_mu[0..4]: xs, ys, rho, q, a.
// Check value_converged and gradient_valid before using these outputs.
// gradient_precision_checked is false: gradient accuracy is not certified.
```

`FusedGradientOptions` exposes only the existing value absolute/relative
tolerances. Seed preparation stays with the caller, allowing already
prepared seeds to be reused. `support_certified` reports the kernel's
coverage flag separately from quadrature convergence. No default CMake
target or existing value/derivative entry point is changed.

Run `build-seeded-events/fused-gradient/api_check` for the analytic limit,
normal value/API coexistence and invalid-input checks, and `batch_check`
for the shared root-pair Jacobian check. `api_bench` uses the normal value
kernel and the compiled API within one executable; it accepts the same
arguments as the existing benchmark. An optional sixth argument selects
mu_rtol for small resolution studies.

The value comparison uses `lightcurve_value(curve, times, params)`. It calls
the ordinary LightCurve selector, retaining point-source and hexadecapole
epochs, and invokes the caustic-aware seeder plus radial-frontier integrator
only for epochs selected for finite-source integration. Its tolerances must
match `Options.finite_source_tol` and `Options.finite_source_reltol`; the
comparison notebooks set both from their VBM tolerance. Quadratic limb
darkening fails closed on epochs that need integration; point/hex results keep
the ordinary route. The output separates quadrature `value_converged`, seeder
`seed_support_proven`, local `frontier_discovery_completed`, global
`support_certified`, and `fully_converged`. The current seeded frontier does
not set its global coverage certificate, so radial-integral epochs remain
diagnostic values rather than fully accepted results.

Each `lightcurve_value` call clears shared binary caustic, seed, and result
caches once before its first epoch. This prevents repeated timing calls for the
same parameter set from measuring a seed-cache hit; geometry and seeds are
still reused across epochs within that one light curve. The caustic-aware seed
builder is shared with the fused-gradient value path.

Call `curve.holonomic_gradient(times, params)` for the separate optional fused
derivatives. It supports static binary rectilinear curves and returns `mu`, an
`(n_epoch, 7)` `grad_mu` array, convergence/validity flags, and parameter order
`t0,u0,tE,alpha,rho,q,s`. Parallax, xallarap, orbital motion and binary-source
curves are outside this gradient hookup. The curve's VBM or center-of-mass
convention and linear limb-darkening coefficient are used.

```python
curve = lc.LightCurve(options=lc.Options(param_type="vbm"))
result = curve.holonomic_gradient(
    times,
    {"t0": t0, "u0": u0, "tE": tE, "alpha": alpha,
     "rho": rho, "q": q, "s": s},
)
gradient = result["grad_mu"]  # columns: t0, u0, tE, alpha, rho, q, s
```

The wrapper maps VBM and explicit center-of-mass source coordinates into the
kernel's barycentric lens frame before integration; primary-centered
coordinates stay in the primary frame. It prepares seeds with the existing
caustic-aware finite-source component scan, rather than relying on the point
images at the source center alone. The caustic geometry and point-image
continuation state are reused across epochs in each light-curve call. Returned
`seed_support_proven` reports that preparation result separately from the
kernel's `support_certified` flag.

`holonomic_vbm_gradient_comparison.ipynb` is a single-case notebook for the
fused light curve, VBM overlay, caustic/source trajectory, timing comparison,
and seven parameter-gradient curves. It reports support and gradient
validity separately; fused gradient precision remains unchecked.

## Shared routing preparation (2026-10-14)

`curve.holonomic_magnification(times, params)` is the scalar-only value path.
It shares the ordinary point/hex selector, the caustic-aware seed generator and
the seeded radial integral with `lightcurve_value`, but requires the ordinary
convergence/support contract and does not assemble per-epoch diagnostics.
Geometry and seed caches are cleared once per call, then shared within the
curve. Use `lightcurve_value` when inspecting uncertified last iterates.

The common kernels now share centre-root safety diagnostics and predictor
Jacobians, defer image-seed materialization until integration is selected, and
polish retained quintic roots before the same residual/Vieta/cold-retry guards.
A continuous caustic enclosure can certify remote point-source topology
without constructing the detailed caustic polyline; unproved configurations
fall back to the existing geometry and integration routes.

Matched before/after timings, per-epoch values/support flags, code provenance
and validation commands are in
[`router_shared_point_20261014`](../../../evidence/holonomic/router_shared_point_20261014/README.md).
