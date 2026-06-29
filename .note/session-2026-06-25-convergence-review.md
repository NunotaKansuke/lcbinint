# 2026-06-25 convergence review

## Context

The default finite-source configuration was changed after this review to the
fixed-bin policy:

- `source_bins = 50`
- `adaptive_source_bins = 0`
- `max_source_bins = 400`
- `finite_source_reltol = 0`
- `adaptive_hex_threshold = 1e-3`

The goal of this pass was to reduce false `finite_source_converged=false`
flags without reintroducing false accepts.  Low-bin behavior is not used as the
main tuning target; bins below 50 are known to be less stable.

The public default is therefore a stable fixed `source_bins=50` inverse-ray
grid.  `adaptive_source_bins=1` remains available for expert diagnostics, but
`Options(reltol=...)` no longer enables it implicitly.  The default tolerance
that remains active is the hexadecapole/point-source selection tolerance
(`adaptive_hex_threshold`, exposed in Python as `hex_tol`).

## Current convergence logic

For Cartesian inverse ray, the error estimate is built from two pieces:

1. `cartesian_area_error_indicator`
   - uses boundary rows, gap repairs, overlaps, seed count, and max row jump;
   - estimates image-area discretization error in magnification units.
2. local adaptive refinement estimate
   - records near-boundary cells during the coarse scan;
   - splits the largest estimated-error cells;
   - stops when the safety-scaled local error falls below the target.

The final convergence test remains:

```text
error_estimate <= 0.999 * (tol + reltol * max(|A|, 1))
```

The `0.999` margin is intentionally slightly strict.  Changing it to `1.0`
accepted known local-underestimate regression cases.

## Change made

The high-magnification topology floor was too conservative.  It used large
`A / source_bins` style floors for warning patterns with high magnification,
large row jumps, or many gap repairs.  In current source_bins=50 diagnostics,
those floors caused many points with actual VBM-relative errors well below
target to be marked unconverged.

The floor coefficients were reduced, while keeping the local adaptive safety
factors unchanged:

- local safety remains `2.5` for `rho >= 8e-3`;
- local safety remains `1.8` for `rho >= 3e-3`;
- local safety remains `1.25` otherwise;
- high-magnification/topology floor coefficients were softened.

An attempted reduction of local safety produced false accepts in regression
tests, so it was reverted.

## Validation

Regression subset:

```text
tests/regression/test_vbm_consistency.py
tests/regression/test_adaptive_precision_redesign.py
tests/regression/test_component_union_validation.py

83 passed
```

Current global diagnostic:

```text
.note/diagnostic_runs/20260625-031352-conv-review-final-candidate/points.csv
```

Summary:

- cases: 72
- points: 360
- source_bins: 50
- reltol: 1e-3
- accepted_bad: 0
- inverse_ray_cartesian points: 42
- inverse_ray_cartesian unconverged: 14

Before this pass on the same diagnostic setup:

- inverse_ray_cartesian unconverged: 19
- accepted_bad: 0

So the change removes some false-unconverged points without creating observed
false accepts.

## Remaining issue

The remaining IR unconverged points are mostly borderline:

- several have `estimate / target ~= 1`;
- actual VBM-relative errors are typically far below target;
- one LD case still has `estimate / target ~= 1.5` at local refinement level 3.

Further reducing these flags would require a better local error estimator, not
just a looser acceptance margin.  The margin cannot simply be relaxed to `1.0`,
because known-underestimate tests then accept points whose actual error exceeds
the requested tolerance.

## Next direction

The next real improvement should target the local estimator:

- track whether the largest remaining local-error cells are boundary-crossing
  cells or smooth limb-darkening variation cells;
- use different safety factors for those two categories;
- keep the stricter factor only for topology/boundary cells;
- avoid changing the global convergence margin.
