# Actual-row local area bounds (2026-10-12)

## Implemented

The explicit experiment build adds `binary_boundary_strip`: it combines left
and right certified boundary graphs over an actual Cartesian row height,
contracts each midpoint boundary position with interval Newton, and sums
outward-rounded area intervals. Vertical subintervals are retained once
certified; only unresolved subintervals are split. An unresolved subinterval
prevents acceptance of the strip. Partial sums never count as full-row bounds.

The midpoint remainder includes both graph-curvature terms and the rounding
shift between the binary64 evaluation midpoint and the exact subinterval
midpoint. Midpoint root contractions are not attempted when either tube failed.
No whole image refill or second-nbin runtime estimate is introduced.

For uniform brightness, existing boundary fractions give the pre-outer-rule
row estimate h^2 (hi-lo+delta_left+delta_right). The audit compares that estimate
with the certified area interval, obtaining an upper bound on its local area
error by taking the largest distance to either interval endpoint. Independent
high-precision areas confirm these derived bounds as well.

This certifies the integral BETWEEN two supplied smooth graphs. Interpreting it
as physical run area requires excluding extra interior boundaries. This helper
does not certify missing images or hidden holes and cannot accept total flux.

## Actual-width pilot

Five binary geometries, nbin 16/64, 12 sampled run records per evaluation: 120
actual-height rows. Centre/extremum seed reduction is disabled for the trace
collection. Uniform brightness, component refinement disabled to isolate rows.

| Bounds / vertical subdivision | Fully resolved rows |
| --- | ---: |
| Original cell brackets, no subdivision | 10 / 120 |
| Original brackets, depth 3 | 27 / 120 |
| Original brackets, depth 6 | 33 / 120 |
| Brackets widened by up to one cell, depth 3 | 104 / 120 |
| Brackets widened by up to two cells, depth 3 | 100 / 120 |

Widening is capped to keep left/right rectangles separate. A wider rectangle is
not always better: derivative interval inflation can prevent certification.
Across all configurations, 108 distinct rows have a resolved area interval.
Deep subdivision alone cannot fix a bracket from which the boundary exits.

Independent 60-digit lens evaluation and 8/16-point Gaussian quadratures found
zero violations for every accepted area interval and every derived pre-outer
row-error upper bound. Numerical quadrature stability is audit evidence, not a
proof of complete magnification acceptance. This is a bounded pilot, not a
population coverage rate.

One pilot measured all five configurations on all rows in 0.0593 s including
Python call overhead. In a preceding identical-configuration v2 run, the
one-cell widening configuration took 0.00428 s over 120 rows; original brackets
at depth 6 took 0.0408 s. These are local helper measurements, not a production
speedup. The audit deliberately measures multiple alternatives per row; a
runtime estimator should choose a short ordered fallback, not run them all.

## Remaining scope

No automatic nbin law, tolerance gate or default master kernel changed. Next:

- Narrow boundary boxes around the already available root fractions for very
  short runs, rather than attempting deeper subdivision of whole cell brackets.
- Account for turning/endcap regions and any extra interior boundary obligation
  with explicit unresolved budgets or localized refinement.
- Relate certified strip intervals to the actual Bennett Eq. (15) stencil and
  add brightness-quadrature remainder before summing a full magnification bound.
- Measure estimator plus integration time before lowering initial nbin.

Current final evidence:
`tests/diagnostics/results/recal2026/boundary-strip-20261012-v4/`.
Native SHA256:
`fbb01333c729454b57ebe9d9764fe89f47f273e45c178a94a4e67e5ce9f09319`.
v1-v3 artifacts are retained as earlier stages; v4 checks every accepted
configuration against independent area and pre-outer row error, rather than
only the first accepted configuration.

Reproduction (explicit experiment build required):

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
LCBININT_PROBE_BUILD=/rogue1_8/nunota/lcbinint/build-boundary-support \
/home/nunota/.miniconda3/envs/myenv/bin/python \
-m tests.diagnostics.recal2026.boundary_strip_audit
```

## Actual Eq. (15) trace audit

The experiment now records the actual horizontal run contribution (converted
from lattice units to physical area) and the sum of applied Eq. (15) weight
increments. Ordinary calls do not allocate the diagnostic weight vector.
Summing these corrected contributions reproduces the returned magnification
in all ten uniform-brightness, refinement-disabled pilot evaluations; maximum
relative difference is 2.39e-16. This checks actual implementation weights,
including both endpoints, rather than duplicating the weight formula in Python.

The new 120-row audit deliberately includes corrected endpoint rows first.
24 rows have non-unit outer weights. None of those 24 has a resolved two-graph
strip certificate, even with the existing widening/depth alternatives. Of all
120 rows, one-cell widening resolves 81. Independent 60-digit area checks found
zero violations on accepted alternatives. The previous 104/120 figure sampled
mostly uncorrected rows and is not an endpoint result. All 600 local calls cost
0.0681 s in this pilot, including Python overhead; no total-speed claim follows.

For a fully certified panel with true area in [L,U] and its actual corrected
contribution Q, max(|Q-L|,|Q-U|), rounded upward, bounds that panel error.
This identity does not establish that all physical image area is partitioned
into certified panels. Endpoint caps and additional boundaries remain open.
The observed endpoint failures prevent promoting this to a total error gate.

Next local remedy: around a regular turning point, change graph direction
from x(y) to y(x), where g_y can stay nonzero even though g_x vanishes.
Clip the resulting graph enclosure to the row edges and account for the cap
area. This requires a separate proof of the clipped integral; merely swapping
coordinates in the existing smooth-strip remainder is insufficient. Degenerate
turning points remain unresolved. Deep whole-row subdivision is not the remedy.

Evidence: boundary-strip-20261012-v6/summary.json and source-sha256.json under
 tests/diagnostics/results/recal2026/. v5 is retained as a failed diagnostic stage:
its raw run areas omitted h^2 conversion, and the reconstruction check caught
that units error. v6 fixes it. Main master kernels and nbin law remain unchanged.

Endpoint rectangle follow-up: see `nbin-boundary-cap-progress.md`. A clipped
y-monotonicity enclosure now bounds all 24 weighted rectangles, but its width
and the unaccounted outer cap prevent a global tolerance gate.
