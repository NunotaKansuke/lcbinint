# Local gradient replay refinement, 2026-10-15

## Result

At unchanged tol=reltol=1e-3, replaying the smooth background as well as close-pair spans removes most of the remaining historical u0 derivative error. This extra replay runs only at epochs where a close pair is detected. Ordinary epochs preserve their earlier gradient values bitwise. Normal-case speed improvement is small; broad-source accuracy costs additional runtime.

Implementation is experimental: build_geometry.sh / geometry-python. The default build and its module are unchanged. No gradient tolerance or gradient-driven adaptive criterion was added. Gradients retain FiniteUncertified status.

## Accuracy

Relative u0 derivative error in percent, against the prior three-step independent VBM finite-difference median:

| Time [day] | Before this refinement | Final |
|---|---:|---:|
| -.72 | .012650 | .007894 |
| -.704 | .425412 | .009692 |
| -.7 | .008511 | .009222 |
| -.696 | .323893 | .007077 |
| -.68 | .153725 | .013755 |
| +.78 | .012197 | .011333 |

The two main residuals decrease about 44 and 46 times. Small regressions at -.7 are retained in the table. All five directly referenced parameter derivatives and reference step ranges are in accuracy.csv. The t=.1 VBM reference has previously observed disagreement for small derivatives; this point is not used to claim general accuracy. Tight native finite differences in the old reference report did not converge, so they are not an independent certificate either.

### Why the background matters

The old special replay retained depth-zero panel derivative estimates outside the close-pair spans. Increasing close-span Gaussian order 64 to 256 did not remove the residual (order*.tsv). At -.704, Cartesian derivatives are approximately 222 and 122, whereas the projected u0 derivative is only .392: subtractive cancellation amplifies small background quadrature errors. Replaying the other accepted panels with their existing level-based Gaussian rule resolves most of this error (smooth_trial.py). This is a quadrature mechanism supported by the intervention, not a proof that every remaining error has this origin.

The adopted smooth-panel rule has 2*(2^level-1) Gaussian nodes, with polynomial exactness through degree 4*(2^level-1)-1. The value-selected mesh and level determine the rule; no derivative estimate selects the work. Trials with uniformly smaller 7/10/14-node rules degraded some points and were rejected (smooth_orders.py). Close-pair treatment remains at its existing 64-node exterior and mixed seven-node interior rules.

## Speed

One pinned allowed CPU, O2, identical inputs/tolerances, warmup then five repetitions. Times are median API milliseconds for 401 epochs, including seed preparation and parameter mapping.

| Case | Previous geometry screen | Final |
|---|---:|---:|
| Ordinary | 59.97 | 59.35 |
| Current notebook | 79.44 | 78.58 |
| Historical broad source | 426.72 | 451.95 |

These roughly 1% normal-case changes are small and should not be read as a robust broad performance gain. The original pre-geometry ordinary build remains faster, about 47.71 ms. The historical case becomes about 6% slower because 126/401 epochs need the added background replay.

Screen disabled, same final algorithm: historical 439.62 ms, current 119.58 ms, ordinary 107.49 ms. Thus the conservative screen still adds overhead when every broad-source epoch needs detailed detection. Reusing the epoch D14 derivative polynomial reduces repeated construction in forced full-detection workloads, but does not remove this limitation.

Detailed detector / special replay counts remain:

| Case | Detailed detector epochs | Replay epochs |
|---|---:|---:|
| Ordinary | 2/401 | 2/401 |
| Current notebook | 25/401 | 2/401 |
| Historical | 401/401 | 126/401 |

Only these replay epochs change gradient values from the previous build. All magnifications remain bitwise identical. Historical/current times are 401 uniform epochs in [-.8,.8] day; ordinary times are [-10,10]. Parameters are recorded in evaluate.py and the preceding gradient_screen_20261015 report.

### Cheap arithmetic changes

- Construct the screen disk radius with outward binary64 arithmetic and correctly rounded IEEE square root, enclosing the original quad extents. Avoid one software quad square root per proof disk. Overflow stays inconclusive.
- Use symmetric convolution for polynomial squares, computing each off-diagonal product once, with the same scalar enclosure machinery.
- Reuse coefficient lower bounds and prefix/suffix lower sums when a Pellet test cannot pass.
- Build each precision of the D14 derivative once per epoch and reuse it across occupied bands.

These alter proof cost, not acceptance requirements. Any inconclusive proof still invokes detailed detection. The screen domain remains restricted to neighborhoods of existing occupied bands; global image support and gradient error are not certified.

## Verification and reproduction

Screen on/off: bitwise identical magnifications and gradients on three 401-point curves, an 89-point difficult scan, and 60 configurations. All values converge and all tested gradient components are finite. Values also match the preceding build bitwise. A successful convergence flag does not certify gradient accuracy.

Bounds checks: 99954 IEEE successor/predecessor checks; 15000 interval coefficients; majorant containment against quad interval calculations over 1000 geometries; 100000 outward disk-radius checks including extreme scales. See bounds_check.txt, radius_check.txt and their C++ sources.

Run from this checkout:

```bash
bash evidence/holonomic/gradient_refine_20261015/run_final.sh
```

Existing archive libraries and baseline modules are required, as in the preceding report. TMPDIR is the project-local compiler scratch directory. Source/module hashes and Git revision are in provenance.json; the dirty checkout is not represented by HEAD alone. summary.json contains invariant checks; timing_*.json contains individual repetitions; radius_* and square_* preserve intermediate build results. Exploratory order scripts modify generated experimental headers and are not part of the adopted build; rebuild using build_geometry.sh after running them.

## Remaining work

Normal screen overhead and historical conservative failures remain. Error on the tested problematic u0 derivatives is now roughly 1e-4 relative, but this is not a uniform bound. Investigate the residual numerical/quadrature floor and validate additional fold/cusp geometries before default adoption. Avoid reducing special quadrature order solely because one benchmark happens to improve.
