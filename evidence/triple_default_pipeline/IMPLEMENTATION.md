# Ordinary triple lightcurve pipeline

The Python LightCurve default now selects the same algebraic controller family
as binary for supported native, static, single-source triples with uniform or
linear limb darkening. Users call magnification()/info()/the curve normally;
no backend argument is needed. Explicit historical backend selection remains
compatible. Dynamic/parallax, binary-source, nonlinear-LD and JAX triple modes
retain their existing backend instead of being routed into unsupported code.

Value-only and value+10-parameter-derivative calls share one templated scalar
selector. Value-only instantiates no tangent arrays, root tangents, or parameter
pullbacks. The selected scalar error budget is atol+rtol*abs(mu). No runtime
derivative-error estimator, tolerance or adaptive mesh is introduced.

1. Reuse the existing validated central point roots. A positive-radius point
   shortcut requires the scalar curvature indicator 6*Cq*rho^2 to fit the value
   budget, continuous caustic clearance above max(20,kinji_threshold)*rho and a
   matching certified physical-image count.
2. Hex uses the incumbent 13-point axial/diagonal stencil and LD formula. It
   requires certified clearance above max(5,hex_threshold)*rho, the same certified
   image count at every sample, and a fourth-order scalar correction within the
   requested value budget. Tangents reuse each sample's physical roots through
   the implicit lens equation, including the rho motion of the sample positions.
3. Refused or unresolved shortcuts use the existing algebraic integral and
   analytic transported tangent. Reuse lens preparation, continued image/seed
   proposals, critical-phase enclosures and per-epoch support checks. Derivative
   evaluation does not first integrate the value and then restart the integral.

The geometry cache is the continuous critical-curve enclosure tree already
ported from binary into PreparedRegularSupport. The same tree and witness
clearances are now used by routing and by subsequent support proofs. Cache keys
are normalized lens masses/positions, never source fluxes or derivative values.
A new source needs its own proof; nearby source clearance can be inherited only
with the existing conservative movement margin. Budget exhaustion refuses a
shortcut. Unsupported floating environments refuse clearance. Full plotting
polylines are built only by the plotting/legacy APIs; their prior 10.65ms cost is
not artificially added to a route that uses the enclosure tree instead.

The integral value-error regressions exposed an unsafe reduction: replacing the
old/fine nested difference by a smaller embedded/fine difference after apparent
contraction, despite the latter sharing nearly all nodes. Both scalar and
first-order streams now retain the independent nested difference and the
existing interpolation detail floor. No requested tolerance or shared binary
safety factor is changed. Previously computed quadrature packets are retained.

Original dirty changes are preserved in before.diff. In particular,
patterson_stream.hpp already contained unrelated first-order work; this task
changes only its scalar embedded-error reduction. Other existing triple and JAX
work is not reverted or included in an automatic commit.
