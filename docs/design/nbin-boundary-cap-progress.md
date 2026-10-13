# Clipped endpoint area enclosure (2026-10-13)

## Result and scope

The experiment-only `binary_boundary_cap` encloses the uniform-brightness source
preimage area INSIDE a supplied image-plane rectangle. It handles regular
turning points by using monotonicity in y, rather than assuming x(y).
It does not require a boundary root to exist throughout the rectangle: columns
entirely inside/outside the source saturate at the row edges.

All 24 previously unresolved, Eq. (15)-weighted endpoint rectangles have finite
area intervals. This is a bounded geometric result, NOT tolerance acceptance.
When monotonicity cannot be certified, the affected column retains [0, full
column area]. It is not discarded or labeled accurate. Missing area outside
these rectangles and missing image components remain separate obligations.

Main kernels, the existing Bennett integration rule, and the nbin law are
unchanged. Code is in the explicit overlay/header and is not called by default.

## Basis

Let g(x,y)=|lens_map(x,y)-source|^2-rho^2. In each column, interval evaluation
bounds g and g_y over the entire rectangle. If g has a strict uniform sign,
its area is either the full column or zero. Otherwise, if g_y excludes zero,
for each fixed x the mean value theorem bounds any root by

    y_root in y_mid - g(x, y_mid) / g_y(column).

Clip the resulting root interval to [ylo,yhi]. With g_y positive, inside lies
below that transition; with g_y negative, inside lies above it. The clipped
interval remains valid even when no root exists in the row. For example, if
positive g_y and g(yhi)<0, applying the mean value theorem between y_mid and
yhi forces the root interval's upper endpoint above yhi. Thus the enclosure
includes the fully inside height. The wholly outside case is analogous.
Multiplication by the column width and outward summation gives an area interval.

The natural interval form for g(x,y_mid) is intersected with its mean-value
form about the column center. Near a turning point, small g_x removes much of
the dependency inflation. The proof only uses derivatives within the row;
it does not assume a smooth continuation outside the clipping rectangle.

Refinement chooses the column with the largest area uncertainty, retains
all other computed bounds, and splits only that column. Already classified
inside/outside columns are terminal. Work has an explicit column budget
(clamped to 1..256); exhausted budget means a wider interval, not false success.
There is no full-grid rerun or runtime Richardson comparison.

## Independent checks

Input: the 24 Eq. (15)-weighted rectangles from the preceding v6 audit,
three geometries, nbin 16/64, source (.1,.04), rho .12.
Reference: clear the lens denominators to a real sextic for each horizontal or
vertical slice; find candidate intersections with a separate polynomial solver,
refine using 60-digit original lens evaluation, integrate clipped slice lengths
with 16/32/64-point Gaussian quadrature. Split at intersections with row edges.
Largest observed quadrature change: 5.44e-19 area units.
These are independent numerical checks, not an interval-certified reference.
The sextic candidate discovery and numerical quadrature are limitations of the
audit, rather than assumptions in the interval-area implementation.

All 72 area intervals and derived error bounds for the recorded corrected
contributions passed. An additional 35 checks passed: an analytic single-lens
annular cap at four budgets, invalid inputs, empty and fully inside rectangles,
a pole-containing rectangle, and 24 reflected upper-end configurations.
Existing experimental unit_core and unit_boundary_support both passed.

| Maximum columns | Mean interval width / true rectangle area | Maximum ratio | Time for 24 calls |
| --- | ---: | ---: | ---: |
| 8 | 57.9% | 98.6% | 0.422 ms |
| 32 | 6.20% | 31.3% | 1.39 ms |
| 128 | 1.10% | 3.27% | 6.42 ms |

Timings include Python calls and cover only the local helper. They are single
pilot measurements, not a total solver speedup. The width is the entire
interval width, not relative magnification error or a tolerance achieved.

## Remaining work and decision

This removes the outright inability to bound endpoint rectangles, but the
bounds are too loose to promote a small-tolerance nbin acceptance rule. Merely
raising the column budget has an unfavorable cost/accuracy tradeoff.

Before total-flux acceptance:

- Treat the two Eq. (15) endpoint samples as one panel. Summing absolute
  per-row errors throws away the deliberate cancellation in that rule.
- Include the true endpoint cap outside the midpoint-row footprint and
  establish the panel's image-plane extent. The current supplied rectangle
  alone does not do this.
- Tighten panel bounds with integrated boundary curvature or a validated
  local affine enclosure, rather than many more columns.
- Account for arithmetic in the actual stencil accumulation. The recorded
  weighted-row products reproduce its sum to floating-point roundoff, but
  are not an exact decomposition of every rounded addition.
- Combine disjoint complete panels, then measure integration plus estimator
  cost before lowering initial nbin. Limb darkening remains separate.

Final artifacts:
`tests/diagnostics/results/recal2026/boundary-cap-20261013-v3/` contains summary,
edge checks, source snapshots/hashes, and provenance/commands. v1/v2 retain
previous stages; v3 is the current terminal-column optimization.

Signed endpoint-pair follow-up: see `nbin-bennett-pair-error-progress.md`.
Same-domain pairing halves the typical bound in the pilot without repeating
boundary work. Complete endpoint domain coverage remains unresolved.
