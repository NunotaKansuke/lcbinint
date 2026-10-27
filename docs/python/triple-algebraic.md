# Triple-lens algebraic finite-source integration

The explicit `algebraic_boundary` backend supports static, single-source triple
lenses with uniform or linear limb darkening. It integrates image-plane angular
arcs from a sextic boundary polynomial, then integrates radius with nested rules.
It does not use D14 or a sampled angular grid as support authority. Existing
triple defaults are unchanged.

## LightCurve

```python
import lcbinint

options = lcbinint.Options(
    finite_source_backend="algebraic_boundary",
    finite_source_tol=1e-6,
    finite_source_reltol=1e-5,
)
curve = lcbinint.LightCurve(
    lens="triple", param_type="vbm", options=options, ld_c=0.5,
)
params = dict(
    t0=0.0, u0=0.15, tE=1.0, alpha=0.2, rho=0.15,
    q=0.2, s=1.0, q2=0.08, sep2=0.7, ang=2.1,
)
result = curve.info([0.1], params, with_gradient=True)
print(result.magnifications, result.status, result.gradient_valid)
print(result.gradient_parameters, result.gradients)
```

The ten analytic derivative columns are
`(t0, u0, tE, alpha, rho, q, s, q2, sep2, ang)`.
Angles are radians. The existing `vbm` and `lcbinint` lens/trajectory conventions
are preserved; they describe different meanings for `s` and `sep2`, so do not
interchange them without transforming the geometry. Frame motion and normalized
mass derivatives are included. Limb-darkening coefficients are held fixed.

A value-only call does not instantiate parameter derivatives. Native analytic
triple gradients are requested through `with_gradient=True`; JAX tracing is not
part of this native implementation. Orbital motion, parallax, xallarap, binary
sources and quadratic limb darkening are not supported on this triple route.
The finite-source gradient requires positive `rho`. Batch/warmup and other JAX
backends retain their existing, separate contracts.

## Standalone physical-coordinate API

```python
result = lcbinint.triple_algebraic(
    lenses=[0j, 1+0j, -0.4+0.6j],
    masses=[0.72, 0.20, 0.08],
    source=0.1+0.15j, rho=0.15, limb=0.5,
    with_gradient=True, atol=1e-6, rtol=1e-5,
)
```

Masses must be positive and sum to one; positions must be distinct. Coordinates
and `rho` are in total-mass Einstein-radius units. The returned physical-gradient
order is
`(source_x, source_y, rho, m1, m2, lens0_x, lens0_y, lens1_x, lens1_y, lens2_x, lens2_y)`.
For mass derivatives, `m0=1-m1-m2`; lens positions are held fixed. Inspect
`value_converged`, `gradient_reliable`, `support_complete`, `stop` and
`error_estimate`. A finite candidate is not necessarily converged. Missing candidates have NaN values/derivatives and false convergence flags; the LightCurve wrapper raises the existing missing-finite-candidate error. The canonical
8-component gradient/error estimates are also returned for numerical diagnosis.

`gradient_policy="value_mesh"` is the default, matching the binary fused route.
Only the flux error controls refinement. The dedicated gradient frontier carries
flux and all eight canonical derivatives together at each integration point.
Support proofs, coefficient geometry and continued physical endpoints are shared;
completed value/tangent packets remain cached when more radial points are added.
The ordinary value-only frontier evaluates no parameter jets.

For relative tolerances of `1e-3` and `1e-4`, the first-order frontier uses nested
Gauss–Patterson rules, or the existing endpoint-weighted rules for isolated smooth
components. Nearby real and complex radial folds retain hyperbolic charts;
the chart motion and its Jacobian are differentiated with the physical boundary.
Critical angular charts accumulate local coefficient sensitivities together with
the flux, then apply the eight-parameter pullback once per cell. Physical Newton,
orientation, disjointness and certified root-count checks remain active after
endpoint prediction. Uniform full-circle bands use exact annular area and its
moving-boundary derivative.

Tighter requests retain the coupled mixed radial/parameter transport. Both
frontiers control the requested value tolerance; their derivative estimates do
not drive refinement. Binary64 is the ordinary route, with local quad repair for
unresolved endpoints or samples. A failed continuation repairs only that radius.
The radius-normalization derivative uses the flux from the same coupled integral.

This policy reports `gradient_valid=True` for a finite, complete-support candidate,
`gradient_reliable=False`, and `gradient_precision_checked=False`. Canonical
gradient error estimates are infinity (unchecked). Flux convergence is not a
per-derivative accuracy certificate. The LightCurve API uses the same value-only
error policy; its `gradient_valid` flag denotes finite usable output.
The explicit point API retains `gradient_policy="strict"` as a diagnostic route
that also controls all eight estimated derivative errors.

The strict diagnostic may use normalized source-disk quadrature for a caustic-free
source covered by disjoint full-source IFT image disks. The default gradient frontier
uses the algebraic boundary panels; `smooth_source_used` remains false there.

`event_box_budget` bounds support isolation (default 1,000,000 boxes). Exhaustion
is explicit and does not silently switch to another integration method.
Collapsed/zero-mass systems use the existing binary/single-lens APIs; this
standalone triple API requires three distinct positive-mass lenses.

## Precision, refinement and acceptance

Ordinary coefficients, geometry evaluations and quadrature use binary64.
Outward interval arithmetic identifies ambiguous root/event boxes, which alone
are promoted to quad. Sensitive endpoint/sample calculations use local quad
repair; accepted unrelated boxes, radii and samples are retained. Counts include
`quad_samples`, `event_quad_boxes`, `event_quad_refinements`,
`parameter_derivative_evaluations` and `node_evaluations` in standalone results.

On ordinary LD arcs whose concavity is verified, the gradient route integrates
scalar cofactor sensitivity moments and applies the eight parameter directions
once per completed angular integral. Other arcs retain the critical-chart or
physical kernel. The value-only route does not evaluate parameter directions.
The ordinary first-order physical kernel uses explicit real arithmetic for its
complex products and retains the same endpoint roundoff and local repair gates.
If a critical cofactor needs promoted scalar coefficients, its three Fourier
harmonics are rotated directly in that precision; the parameter directions stay
in binary64 and the original physical minimum still checks the result.
Common angular packets use a small inline cache and allocate additional storage
only on refinement; all previously completed nodes retain their identity.

The first-order radial controller shares the value controller's positive
embedded-rule construction and contraction test. Its error estimate uses only
the scalar flux samples already present; the difference safety factor and
interpolation-witness floor are unchanged. Endpoint-weighted rules retain their
existing estimator. Angular stationary-point scans compute only the slope;
accepted minima retain the full physical evaluation and local precision guards.

Angular and radial p-refinement reuse canonical nested nodes. Local angular
repair and radial h-refinement retain their parent work and use epoch-local
radius registries. Cache identities include geometry, source, radius and angular
settings. Event plans cannot be reused for a different epoch. Repeated identical
native C++ LightCurve epochs reuse the support plan.

`support_complete` concerns coverage of stationary boundary events in a bounded
radial domain. It is separate from the estimated flux error and derivative
reliability. Coarse/fine errors use the incumbent factor-two safety policy; they
are estimates, not rigorous enclosures of the final magnification or gradient.
Exact or higher-order degeneracies and resource/precision limits return explicit
failure diagnostics rather than a fabricated converged result.

See `tests/triple_algebraic` for kernel and public-API tests, and
`evidence/triple_algebraic_kernel` for the saved numerical audits. The independent
Gauss–Kronrod flux check shares angular topology/support geometry but uses a
separate quadrature of the original lens mapping. External VBM uniform-source
checks independently test the complete magnification. Saved dark-source VBM
comparisons include unresolved reference-accuracy/robustness limitations; their
requested tolerance alone is not treated as reference certification.

`component_domains` reports how many certified image domains the value
controller selected. A value of 1 denotes the ordinary whole-image path.
Eligible regular LD calls can use the existing weighted panel rule separately
for multiple images, with one shared adaptive error controller. This field
counts image domains, not source annuli, and does not indicate a fixed
annulus or point-source approximation. Convergence and support flags remain
the acceptance checks.
