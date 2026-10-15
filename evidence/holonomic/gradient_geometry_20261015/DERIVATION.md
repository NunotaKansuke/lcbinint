# Geometry-based gradient regularization: mathematical contract

## Goal and scope

Use only the existing magnification atol/rtol API. Do not estimate gradient
error to select a mesh. Additional work follows algebraic root geometry.
The current implementation is an experimental overlay, not the default API.

## Detect a local pair rather than a large derivative

For the angular boundary quartic P(t,R;p), radial branch points satisfy
P=P_t=0. With v=R^2 their discriminant is D(v;p)=D14(v;p).
A close pair has a real stationary point v_c of D, with quadratic proposal

    D(v_c+z) = A + B z^2 + higher terms,
    eta = A/B = 2 D(v_c)/D''(v_c).

eta>0 proposes conjugate roots; eta<0 proposes two real roots.
Root proposals are polished against the actual discriminant. This expansion
never substitutes an approximate value or derivative into the flux integral.

The detector searches D' in existing occupied support bands. Bernstein
coefficients enclose the exact polynomial on a real interval. Zero sign
variations exclude stationary roots; one variation isolates one; multiple
variations require subdivision. Uncertain binary64 interval signs retry in
binary128. A bounded search reports incompleteness instead of assuming no root.
It does not certify image-support completeness outside the seeded bands.

For a proposed pair, interval Taylor coefficients d_k on the disk |z|=r,
r=2 sqrt(|eta|), must satisfy

    inf |d_2| r^2 > sup |d_0| + sup |d_1| r
                       + sum_{k>=3} sup |d_k| r^k.

Rouche's theorem then gives exactly two roots inside that disk. This is a
sufficient condition for an isolated quadratic cluster. Failure is not proof
of safety: higher-order clusters remain unsupported by this normal form.
The polished root locations are numerical approximations, not root enclosures;
the disk root count alone does not certify every downstream gradient error.

## Two coordinate charts of one normal form

Let the corresponding radial roots be R_c +/- i b or R_c +/- b.
Locally the angular root separation is proportional to
sqrt((R-R_c)^2 + eta_R). Its derivative contains the inverse square root.

For complex roots use

    R = R_c + b sinh(y),
    dR / sqrt((R-R_c)^2+b^2) = dy.

For two real roots the natural continuation is

    exterior: R = R_c +/- b cosh(y),
    interior: R = R_c + b sin(theta).

These regularize the normal-form square root on its respective real domains.
The current prototype implements sinh for complex pairs and existing fold
maps plus explicit physical-event insertion for real pairs. It DOES NOT yet
implement the full cosh/sin continuation or analytic singular-term subtraction.
At exact coalescence neither chart should be evaluated by clipping b to an
arbitrary positive floor. A derived limiting expression, or an explicit
unresolved result, is required.

## Differentiation rule and artificial boundaries

For I(p)=integral_{a(p)}^{b(p)} f(R,p) dR choose any smooth radial material
velocity w(R,p) satisfying w(a,p)=a_p and w(b,p)=b_p. Then

    I_p = integral_a^b [ f_p + w f_R + (partial_R w) f ] dR.

This follows directly from the Leibniz rule and integrating partial_R(w f).
For an internal artificial cut use matching velocities on both sides, so its
boundary terms cancel. The current all-span linear material velocity obeys
this condition. Changing the quadrature coordinates at a fixed p does not
change this identity or require differentiating the detector's branch choices.
The quadrature uses the actual transported derivative density and the actual
Jacobian. No interpolation, clipping, or averaging of reported gradients is
part of this rule. Finite quadrature still requires validation across chart
changes; the identity is not a finite-order error guarantee.

## Where additional resources belong

1. Exclude safe intervals with interval polynomial bounds.
2. Locate and validate only isolated local clusters.
3. Add both physical folds when present; preserve their implicit motion.
4. Regularize the cluster contribution; integrate the smooth remainder on
   the value-selected mesh. Do not raise the order on an entire occupied band
   merely because one narrow cluster lies inside it.
5. Use stable fold limits in the angular kernel when mapped nodes approach a
   physical endpoint beyond binary64 resolution. More Gaussian points alone
   can make this conditioning worse.

The prototype currently overallocates steps 2 and 4. Its fixed 64/128 orders
are experimental replay choices, not mathematically guaranteed optimal orders.
They are not proposed as API tolerances or calibrated defaults.
