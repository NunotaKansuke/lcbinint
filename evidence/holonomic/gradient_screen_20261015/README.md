# Geometry screening and remaining gradient error

Checkout: experiment/algebraic_boundary, HEAD b31db3b3be16f9c573dc81d9c87775a045a4aacc plus dirty experimental sources (hashes in provenance.json).
All API comparisons use tol=reltol=1e-3, one pinned CPU, identical inputs, five warm repetitions; numbers below are median milliseconds for 401 epochs including parameter mapping and seed preparation.

## Changes

A conservative root-count screen runs before detailed D14 stationary-root enumeration. It proves that all discriminant roots in the detector neighborhood are already represented by distinct known real folds. Inconclusive proofs fall back to detailed detection. Existing close internal folds are separately recognized and regularized even when no missing roots exist. Their neighboring exterior spans use a cosh coordinate; the interior retains the mixed seven-node rule. Accepted value meshes are reused when topology is unchanged.

There is no new tolerance, gradient-error estimator, or gradient-driven adaptation. These changes are in the experimental geometry build; the default build/module has not been switched.

## Runtime

| Case | Original gradient | Previous full detector | Final screen disabled | Final screen enabled |
|---|---:|---:|---:|---:|
| Historical broad source | 234.47 | 375.23 | 413.93 | 426.72 |
| Current notebook | 58.46 | 125.38 | 126.61 | 79.44 |
| Ordinary event | 47.71 | 113.93 | 115.84 | 59.97 |

Ordinary event: 397/401 epochs use only cheap processing, detailed detection at 2 epochs, local replay at 2 epochs. Current notebook: detailed detection at 25/401, replay at 2. Historical broad source: detailed detection at all 401, replay at 126. The screen is conservative and costs about 3% extra in that historical case. Ordinary overhead versus the original algorithm remains about 26%; matching its speed is not yet achieved.

Historical parameters: s=.6, input q=1e-4, rho=.05, u0=-1e-5, tE=10, alpha=.5, t0=0, linear limb darkening=.5. Current notebook changes input q to 1e-3 and rho to .0005. Ordinary event uses s=1.2, input q=.01, rho=.001, u0=.2; other parameters unchanged. Historical/current curves use 401 uniform times from -.8 to .8; the ordinary curve uses -10 to 10.

## Accuracy and checks

At t=-.68, historical u0 derivative improves from .41918567 to .39676685; independent VBM finite-difference median is .39737772. Absolute error decreases about 35 times, leaving 0.154% relative error. Errors still remain near -.704 (about 0.43%) and -.696 (about 0.32%). These are observations against the prior multistep VBM reference, not a general gradient guarantee. See accuracy.csv for reference step ranges; the near-zero s derivative at t=.1 has reference disagreement and should not be treated as certified.

Screen enabled/disabled gives bitwise identical magnifications and gradients for all three 401-point curves, an additional 89-point historical scan, and a 60-configuration matrix. All curve values report convergence and all 2807 gradient components per curve are finite. This establishes tested screen invariance, not certified gradient accuracy or global support completeness.

99954 IEEE successor/predecessor checks and 15000 interval coefficient comparisons agree with the existing interval implementation. Majorant enclosures were also checked against quad interval coefficients over 1000 geometries. See check_bounds.cpp and bounds_check.txt. git diff --check passes.

## Reproduction

Build experimental module:

```bash
bash benchmarks/holonomic/fused_gradient/build_geometry.sh
```

Module directory: build-algebraic-boundary/gradient-probe/geometry-python. Original module: build-algebraic-boundary/fused-gradient/python. The preserved previous detector module is geometry-before-screen-python under gradient-probe.

Use evaluate.py for original/full/screen/screen_off comparisons, profile.py for detector counts, matrix.py for the 60-case comparison, and analyze.py for summary.json, accuracy.csv and source/module hashes. Scripts are self-contained relative to this checkout; use the project compiler scratch directory and PYTHONDONTWRITEBYTECODE=1. Exploratory trial files are evidence only and are not active build inputs.

Remaining work: reduce conservative-screen overhead and historical fallbacks; investigate remaining radial quadrature error without adding gradient-driven adaptation; validate more geometries before adopting a default.
