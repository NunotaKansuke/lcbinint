# Resolution/error estimate: local boundary tube prototype

## Default route and experimental isolation

As requested, normal source kernels, headers, bindings and core tests have been
restored to master baseline `b51c00c566667fc2ed1fe640543219f480642002`.
Their tracked content is identical to that commit. CMake alone adds an OFF-by-
default `LCBININT_BOUNDARY_SUPPORT_EXPERIMENTS` option. Existing research changes
were copied to `experiments/boundary-support-overlay/`, with their original
manifest and full patch, before restoring the normal files. No unrelated event
inputs or figures were removed. Algebraic discriminant/minimal-seed archives
remain available.

Normal build: `build-master-main-20261012` (option OFF).
Experiment build: `build-boundary-support` (option ON).
New experiment bindings belong to the overlay, not the normal Python module.

## New implementation

`binary_boundary_tube` in `binary_boundary_enclosure.hpp` evaluates

    g(x,y) = |lens_map(x,y) - source|^2 - rho^2

and encloses its first/second derivatives over an image-plane rectangle using
outward interval arithmetic. Opposite endpoint signs uniformly over the
vertical interval, and a nonzero g_x throughout the rectangle, establish a
unique smooth boundary graph x(y) there. It then encloses

    x'  = -g_y/g_x
    x'' = -(g_yy + 2 g_xy x' + g_xx (x')^2)/g_x.

The geometric midpoint remainder for this ONE boundary graph over height h is
at most h^3 sup|x''| / 24. A pair of certified left/right graphs can bound a
uniform-brightness strip's midpoint area error by adding their contributions.
This is a local building block, not full flux acceptance. Turning points,
poles, uncertain signs and unsupported arithmetic return unresolved. No epoch
restart or second-nbin evaluation is required by the helper.

It does NOT yet bound Bennett Eq. (15)'s actual outer-coordinate rule,
limb-darkened brightness quadrature, missed boundaries, or all endcap areas.
Changing nbin or accepting a full magnification on this helper alone would be
incorrect. Runtime tolerances and automatic nbin therefore remain unchanged.

## Independent bounded audit

`boundary_tube_audit.py` reuses 64 recorded boundary segments and three tube
heights (0.25, 0.05, 0.01 times horizontal segment width). This is deliberately
a local pilot, not an acceptance-rate claim at actual production strip heights.

192 candidates: 157 certified, 35 unresolved. At five y positions in every
accepted tube, independent 60-digit differentiation of the original lens map
found no slope/curvature enclosure violations. Independent 8/16-point Gaussian
quadratures of the boundary graph found no midpoint-bound violations. These
numerical cross-checks audit the analytic local bound; quadrature convergence
alone is not a proof of a global flux bound. No scientific acceptance criterion
was loosened.

Artifacts: `tests/diagnostics/results/recal2026/boundary-tube-20261012-v1/`.
The first implementation uses interval evaluations in addition to the fill;
its cost in a production error estimator has not been measured.

## Next integration step

Use recorded boundary fractions/rows to propose tubes at actual strip heights.
Combine left/right local contributions, keep the already integrated rows, and
assign turning/endcap intervals conservative local budgets. Refine only the
unresolved or dominant intervals. Account for actual Bennett weights and
brightness remainder before using the sum as a tolerance acceptance bound.
Only after that compare lower initial nbin by total elapsed time and independent
accuracy. Whole-grid repeated nbin/Richardson convergence is not a runtime
acceptance mechanism.

## Restoration checks

Default master build versus the preserved original master build: 96 saved
nbin=256 Cartesian/polar inputs, zero value and acceptance mismatches.
`master-route-isolation-20261012-v1/` retains actual inputs, outputs, native
hashes and the default core unit log. The default core test passes; the
experimental core/boundary-support tests also pass (2/2). These are restoration
and local prototype checks, not a new nbin-law validation or speedup claim.
