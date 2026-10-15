# Large-source shared seed acceleration — 2026-10-15

## Change

The algebraic value/gradient seed helper now retains a separate PointSourceMagnifier for each certificate probe ordinal across epochs. Previously certificate probes shared one last-root set, continually replacing one another. Per-probe continuation preserves locality and avoids repeated cold polynomial solves.

All probes, seed filtering, residual and root-usability checks, and cold solver fallback are preserved. No tolerance is relaxed and no gradient-based adaptivity is added. The bank is limited to 1024 entries; overflow probes still execute with the original provider. Epoch-cache clearing also clears the bank. Legacy inverse-ray callers retain their previous path.

## Full public API timing

401 uniformly spaced epochs, normal LightCurve.info call, same pinned CPU, one warmup and median of five calls, tol=reltol=1e-3. Constructors excluded. Times are milliseconds per entire curve.

| Case | Before | After | Speedup |
|---|---:|---:|---:|
| rho=.5, linear LD .5 |233.17|155.97|1.49x|
| rho=.5, uniform |220.09|143.74|1.53x|
| rho=.05, linear LD .5 |202.03|163.81|1.23x|
| ordinary, linear LD .5 |0.364|0.365|~unchanged|
| notebook alpha=180 radians, linear LD .5 |16.37|16.70|~2% slower|

This improves a shared large-source cost; the benefit is not restricted to uniform brightness. Remaining radial/angular integration and boundary work are unchanged.

## Verification

- 60 standard configurations: 10 trajectories, LD coefficients 0/.5/.9, tol=reltol 1e-3/1e-4.
- 16 additional reverse/random-with-repeat epoch-order configurations.
- Maximum relative magnification change: 2.1874e-11 across all 76 configurations.
- Every recorded seed count, node count, integrator choice, convergence and support flag remained identical.
- All 168420 gradient components finite; gradient value-convergence counts unchanged. Maximum scaled gradient change 2.756e-8, using denominator max(abs(previous gradient),1e-8). Maximum absolute change 9.742e-5 occurs for a large gradient with scaled change below 8.3e-9.
- These are before/after parity checks, not a new certification of gradient accuracy. Existing integral support_certified=false flags remain false.
- Both default value and gradient native modules rebuilt; git diff --check passed.

## Rejected experiment

Caching squared caustic distances to avoid square roots preserved 1.6 million comparison outcomes across four rounding modes, but slowed the large-source pilots by roughly 6%. Reverted from active source; retained as rejected_scan_trial.cpp.

## Artifacts

bench.py and extra_checks.py reproduce measurements. Individual *_before/*_final JSON/NPZ files contain timings and outputs; comparison.json and summary.json aggregate them. provenance.json records HEAD, source and native hashes. Baseline native module is retained under build-algebraic-boundary/speed-before. The preceding lightcurve_matrix_20261015 evidence contains trajectory definitions and prior VBM/independent LD comparisons. The user comparison notebook was not edited for this change.
