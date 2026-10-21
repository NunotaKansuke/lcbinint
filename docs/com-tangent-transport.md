# COM-fixed tangent transport for thin images

The primal algebraic integral is unchanged. This chart changes its analytic
derivative integrand, not the physical model or a sampling gradient estimator.

## Coordinate identity

The primary frame places the lenses at 0 and `a`, with masses `m0` and
`m1 = 1 - m0`. Image coordinates satisfy

```
x_primary = x_COM + m1 * a.
```

Holding the COM image coordinate fixed therefore gives the internal operators

```
D_m0 = partial_m0 - a * partial_image_x
D_a  = partial_a  + m1 * partial_image_x.
```

The other three internal derivatives (`X`, `Y`, `rho`) are unchanged. The
existing internal-to-user parameter Jacobian still applies afterward.

For a supported finite-source image-plane integral, adding the derivative
of a constant image-plane translation does not change the integral:

```
integral partial_image_x I(image) dA = 0.
```

The image intensity has compact support and translation has zero divergence.
For uniform intensity the statement includes the boundary distribution; for
linear limb darkening it is the corresponding integrable weak derivative.
No image support or topology certificate is inferred from this identity.
Incomplete coverage must still remain uncertified.

## Conditioning

In the barycentric convention, let `g` be the complex lens-map residual,
`i = 1/conj(z)`, and `j = 1/(conj(z)-a)`. At fixed primary image coordinates
the separation term contains a large translation contribution. After COM
transport its physical combination is

```
D_a phi + m1 * D_X phi
  = 2 * m0 * m1 * dot(g, j*j - i*i) / rho**2.
```

This avoids integrating the dominant lens's large common translation and
cancelling it only after quadrature. That cancellation was particularly
damaging for a thin image and an extreme mass ratio. Increasing quadrature
order or adjusting a gradient tolerance does not remove its conditioning.

The chart is selected once per epoch from source radius, lens scale, and
machine precision. It cannot switch on and off between radial samples. Its
thread-local scope is restored on return and on exceptions. Primal evaluation
never uses the transported derivatives.

## Implementation boundaries

- Transport both the endpoint derivatives and angular intensity derivatives.
- Select the same coordinate and precision scopes at every actual tangent
  node, including direct regular-chart and incomplete-value fallback solves.
  A successful Gaussian replay must not be the only scope owner.
- Do not add a second radial material-coordinate motion while this chart is
  active.
- The existing quartic K-rule derivative describes fixed primary coordinates;
  use the direct angular chain rule for transported tangents instead.
- Differentiate angular-node motion and the angular interval width together.
- Regular-radius replay uses an endpoint-squared quadrature substitution.
- Gradient tolerances do not select or refine the default replay.
- Keep the candidate and its existing uncertified flags. The coordinate
  identity is not a new numerical derivative-error bound.

## Thin angular boundaries

A Sturm certificate for rounded double coefficients does not certify the
physical polynomial before coefficient formation. Near a radial fold, rounding
can remove an actual positive arc. Its value contribution may be tiny while
its endpoint derivative is not. Accepting that node as empty silently loses
its tangent contribution.

Thin replay forms local coefficients in extended precision and computes every
dynamic crossing count from physical inputs in binary128. A rounded polynomial
can incorrectly certify four roots when the physical polynomial has two, not
only incorrectly certify an empty arc. If the ordinary angular solve disagrees with the required crossing
count, binary128 physical coefficients use the existing Sturm root isolation
algorithm. This is a root-recovery path, not an angular grid, numerical-gradient
fallback, or gradient-tolerance controller.

Unpartitioned full-span replay panels also use an endpoint-squared variable
substitution. The default fixed replay finishes its finite set of panel rules
even if the primal exhausted the shared refinement budget. Its candidate stays
uncertified with an unbounded derivative error estimate; the actual extra node
count remains visible.

Replay is not conditional on primal convergence. An unconverged primal must
still produce an actual full-domain tangent candidate rather than expose the
deferred tangent's zero initialization. Its complete replay value is retained
as a candidate with an unbounded error estimate; convergence is not promoted.
An empty active workspace is not accepted as a zero integral.

Validation compares with lcbinint's own value differences across several step
sizes. Finite differences are diagnostic only and never supply a sampler's
gradient. Numerical timing and remaining full-likelihood checks are recorded
in the MB23169 research artifacts, not promoted to universal certification.
