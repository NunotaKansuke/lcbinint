# Tangency correctness fixes — native binary lens

This is a post-fix companion to `REPORT_tangency_defects.md`.  It does not
replace the speed report and none of the measured speed/grid/route rules were
changed.

## Independent regression corpus

`tangency_correctness_before.json` records the installed pre-fix extension.
`tangency_correctness_after.json` records the rebuilt extension and is checked
by `tests/regression/test_tangency_correctness.py`.  Arbiter values and their
`self_gap` are read from the existing JSON files; VBMicrolensing is accepted as
the closest party only in the nine verdicts for which the independent arbiter
has a decade of decision margin.

| corpus | before relative gap | after relative gap | certified | converged |
|---|---:|---:|---:|---:|
| A, Cartesian, 11/11 | 3.75e-3–1.62e-2 | 1.35e-5–2.19e-4 | 11/11 → 11/11 | 11/11 → 11/11 |
| B, both grids, 9/9 | 7.69e-4–2.27e-3 | 7.98e-8–7.78e-5 | 9/9 → 9/9 | 4/9 → 9/9 |

All A results remain `inverse_ray_cartesian`; all B results remain the shared
`source_plane_quadrature` route.  The flags were not suppressed.  `certified`
still means that disk support/topology was proven.  `converged` is independently
credible on these cases only because the values now lie inside the arbiter
envelope.

All 12 rows in `tangency_arbitration_vbm.json` were rerun.  The three rows in
which the arbiter had already found VBM pathological remain consistent with the
native grids (native gaps 2.86e-5–1.46e-4); they are recorded in
`defect_b_all` but deliberately excluded from the nine-row VBM-right assertion.

## C — seed invariance

Two distinct defects contributed to the observed dependence:

1. Probe-image proximity deduplication replaced an earlier representative with
   a later one.  At finite lattice spacing the replacement need not reach the
   same connected run, so a deeper certificate could remove coverage.  Probe
   seeds are now retained as a union and deduplicated only after lattice
   snapping.
2. The legacy single-run walk lets the first seed determine ownership at seams.
   Production seed cells are now put in a canonical order.  The multi-run
   candidate follows seed parity independently of existing ownership.

The recorded seed-addition case changed from a non-monotonic
`823.163656699 → 822.016016337` to identical values
`823.163656750 → 823.163656750`.  On the explicit order case the legacy walk
gives `112.918051928` versus `113.624718163` when sorted by increasing versus
decreasing `|J|`; the fixed path gives `113.624712640` in both orders.

## A — disjoint row runs

The old per-fill scratch still supplies the established boundary correction as
one candidate.  A second scanline flood fill stores a private set of merged runs
per lattice row (`ClaimedCellRuns`) and discovers every vertically connected
run.  The two candidates start from identical claimed-cell state; the larger
inside-support footprint is committed.  This preserves long, sub-cell fold
images for which the legacy continuation is better resolved, while allowing a
pinched row to contribute multiple disjoint runs.

For the first A arbiter case at nbin 400, diagnostics report 226 snapped seeds,
4 processed component starts and 368 scanline intervals among the selected
component footprints.  Its value moved from `1.73084582857` to
`1.75936853176`; the arbiter is `1.75934221825` (`self_gap=1.86e-5`).  Across
all eleven cases the former 0.37%–1.62% Cartesian deficit is gone.

The investigation also showed that the report's original single-cause account
was incomplete: retaining the full seed union alone repairs all eleven stored A
cases even through the legacy walk.  The multi-run representation is retained
because it removes the structural one-interval limitation rather than relying
on redundant seeds to partition a pinched component.

## B — actual shared upstream cause

In the current tree the nine cases do not reach either image-plane grid.  Both
explicit grid requests are intercepted by the shared grazing
`source_plane_quadrature` route, which explains their bit-identical answers.
The equal-area midpoint rings treat a caustic just outside the limb as smooth,
but its off-centre fold spike is not resolved.  This was not a flood-fill or
certificate-boundary failure.

Both the chord and grazing branches now use composite order-eight
Gauss–Legendre panels at 48/96, escalating to 192 when their absolute difference
exceeds the requested budget.  This is the arbiter's discretisation but is
implemented in the native batch point-source path.  Median per-grid runtime on
the nine stored cases rose from about 0.021 s to 0.400 s; that is the deliberate
cost of removing a silent 1e-3 error.  No speed rule was changed.

## Regression and scan status

- Native CTest: 1/1 passed.
- Build-extension Python regressions: 224 passed, 3 skipped.
- The 115 existing tangency cases were preserved.  The five missing cases were
  retried with eight workers; after more than five minutes all five remained in
  computation and no complete case file was produced, so the retry was stopped.
  The directory therefore remains 115/120 and must not be described as a full
  post-fix scan.

The speed campaign should be regenerated for the grazing source-plane route and
the Cartesian tangency band.  Existing aggregate speed tables were not edited.
