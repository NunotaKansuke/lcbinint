# Easy versus difficult gradient cost, 2026-10-18

Checkout: experiment/algebraic_boundary, HEAD e740210b plus preserved previous-turn optimizations and this-turn.diff. Native SHA256: b01ab0c35a3ceb30c0b7fa14ac09fb8942ed10eb881a658980037c875cf32095.

## Findings and changes

The historical ~20% overhead was an aggregate result, not a bound for every integral epoch. evidence/holonomic/fused_gradient_followup/README.md reports 7,216 inputs, aggregate 1.230x including seed preparation but excluding caustic prebuild. Current simple point/hex epochs retain low overhead. Integral epochs that ultimately need no extra treatment still incur conservative geometry screening.

Two changes preserve the existing tolerances and detailed difficult-case calculation:

1. After the coarse root-count disk proof is inconclusive, prove the same rectangle clear with smaller overlapping circumscribed disks. Every disk must pass the root-count and real-fold witness proof; inconclusive cases retain detailed detection. Existing polynomial blocks and witnesses are reused.
2. Newly detected complex pairs do not change physical boundaries. Retain the accepted value mesh and continuation workspace for their tangent replay. Newly detected real folds retain the existing topology repair.

## Matched measurements

Public LightCurve API, float64, LD=0.5, absolute and relative tolerance 1e-4, one pinned CPU, OMP/OpenBLAS threads=1, nine repetitions with alternating order and cleared epoch caches. Before is the previous-turn optimized version, not HEAD alone. See public_epochs.py and before/after/summary.json.

| Curve | Gradient before ms | Gradient after ms | After gradient/value |
|---|---:|---:|---:|
| notebook, 401 epochs | 50.5413 | 46.5155 | 2.508 |
| ordinary, 401 epochs | 6.2086 | 6.0075 | 1.267 |
| crossing, 101 epochs | 5.7620 | 5.7365 | 1.358 |
| point/hex mixed, 201 epochs | 3.8359 | 3.8348 | 1.083 |

Notebook groups below use baseline mechanism classification, not a timing threshold. Numbers are sums of per-epoch medians, not whole-curve medians.

| Group | Epochs | Value after ms | Gradient before ms | Gradient after ms | After ratio |
|---|---:|---:|---:|---:|---:|
| hex |354|7.3366|8.6520|8.6187|1.175|
| integral, no extra processing |37|6.3585|15.3445|13.6291|2.143|
| integral, extra processing |10|4.6343|26.2871|23.9877|5.176|

Instrumented native profiling: detailed detection for the simple 37 epochs falls from 22 to 8; their nodes remain 2071. Geometry checking falls from 6.955 to 5.067 ms. Difficult-group nodes fall from 8506 to 7278 through workspace reuse; detailed detection stays at 9 epochs. Profiling timers exist only in copied evidence sources, not production sources.

Remaining limitation: simple integral epochs are still 2.14x rather than 1.2x. The change reduces conservative screening and repeated work; it does not eliminate the remaining screening and tangent evaluation cost. No universal 20% claim is supported.

## Validation

62 Python regressions and C++ unit_core pass. The 120-case x33-epoch matrix (including case107 and case117) has finite values and gradients throughout, numerical convergence 3960/3960, gradient-valid 3928/3960; the unchanged 32 flags are seed_support_unproven. Against the saved previous-turn optimized build, all 120 cases have bitwise-identical value-only values, fused values, gradients, and gradient-valid flags (direct-regression.json). This establishes preservation of that baseline, not independent gradient accuracy certification.

See tests.log, core.log, stress.json, direct-regression.json, public-groups.json, group-profile.json and this-turn.diff. No commit or push performed.
