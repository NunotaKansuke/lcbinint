# Forward fused gradients (isolated experiment)

This experimental entry point carries derivatives for `(xs, ys, rho, q, a)`
while computing the magnification. It does not call a VJP or rerun the
solver for each parameter. The original `src/` headers are untouched.

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
