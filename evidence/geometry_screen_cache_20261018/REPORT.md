# Removing duplicated difficulty-screen work

2026-10-18. Checkout experiment/algebraic_boundary, HEAD e740210b plus preserved prior optimizations. This turn changes only tangent/fused_geometry_screen.hpp; this-turn.diff is relative to the actual starting source (before.hpp), not HEAD. Native SHA256 is recorded in after5/summary.json and parity.json. No commit or push.

## Adopted implementation

The old difficulty screen first expanded and tested the discriminant with an expression-majorant estimate, then repeated the expansion/test using binary64 interval arithmetic when inconclusive. On the saved simple integral group, the first pass certified counts on only 37 of 276 attempted disks. Start directly with interval arithmetic instead. The same strict root-count and distinct real-fold witness conditions apply. Inconclusive interval calculations still escalate to the existing higher-precision calculation and detailed detector. This simplifies the common screening path without adding difficulty thresholds or skipping a required proof.

Interval multiplication now chooses its two extremal endpoint products when signs are known, rather than always computing four products. Both-sign intervals retain the required extremal comparisons; bounds still round outward. A deterministic 200,000-case test over wide exponents checks enclosure of all four quad-precision endpoint products (check_products.cpp, products.log).

Accepted value meshes, retained continuation state and the difficult-case replay logic are untouched by this turn. Point and hex routes do not enter this screen. This reduces the cost of deciding that no special treatment is needed; it does not make that decision free.

## Matched timing

Public LightCurve API, float64, LD=0.5, atol=rtol=1e-4, one pinned CPU, OMP/OpenBLAS=1, nine repetitions with alternating value/gradient order. Same inputs and timing boundaries as evidence/easy_gradient_20261018/public_epochs.py. Before: that directory's after/summary.json and after/*.npz (the actual starting optimized build). After: after5/summary.json and NPZs. No derivative work included in value-only timing.

| Curve | Gradient before ms | Gradient after ms | After value ms | After gradient/value |
|---|---:|---:|---:|---:|
| Notebook 401 epochs |46.5155|44.3585|18.5713|2.389|
| Ordinary 401 epochs |6.0075|5.8224|4.7308|1.231|
| Crossing 101 epochs |5.7365|5.5430|4.1885|1.323|
| Point/hex mixed 201 epochs |3.8348|3.8115|3.5562|1.072|

Fixed baseline mechanism masks; sums of per-epoch medians (not whole-curve medians):

| Notebook group | Epochs | Gradient before ms | Gradient after ms | Value after ms |
|---|---:|---:|---:|---:|
| Simple integral, no extra processing |37|13.6291|11.8542|6.3521|
| Integral with extra processing |10|23.9877|23.5468|4.6638|
| Hex |354|8.6187|8.6272|7.3036|

The simple integral group improves by 13.0%, but remains 1.866x value cost, not 1.2x. The changes support the measured configurations, not a universal overhead bound. Historical 1.23x aggregates use different workloads and cannot establish per-epoch regressions.

## Validation

62 Python regression tests pass; C++ unit_core passes. 120 stress configurations x33 epochs (including case107 and117) retain bitwise-identical value-only outputs, fused values, all gradients and gradient-valid flags against the starting build, for every configuration. All values and gradients finite, 3960/3960 numerical convergence, unchanged 3928/3960 gradient-valid flags. The remaining32 were already seed_support_unproven. See parity.json, stress.json, tests.log and core.log. This is baseline preservation, not independent gradient accuracy certification.

## Rejected experiments

Local polynomial expansion caching worsened cancellation enclosures and runtime (52.91ms notebook; bounded reanchoring49.66ms), so it was removed. Starting with the subdivided disk cover also slowed the notebook (50.06ms), so the original proof ordering was restored. Neither experiment remains in production. after, after2 and after3 hold these rejected measurements; after4 is the adopted interval-first change before interval-multiplication optimization. The division-bound experiment and its bound.log are also not part of the final source.

## Reproduce

Build with cmake --build build-route-unified -j2. Run evidence/easy_gradient_20261018/public_epochs.py --label ../geometry_screen_cache_20261018/NEW_LABEL using the myenv Python. Run compare.py NEW_LABEL for group timing and four-curve parity. Use stress.py for the120-case matrix (writes stress.json and stress-results), and the six regression files listed in evidence/gradient_optimization_20261018/REPORT.md. Existing parallel dirty work was preserved.
