# case_117 near-limb gradient replay

## Finding

At `t=-rho*tE=-1.671007233467463`, the source center is on the limb to binary64 precision (`|center|-rho = 2.78e-17`). The cached caustic polyline places branch 1 at distance `0.167098434878553` from the source center, about `2.2885e-6` inside the source radius; its bounding-box diagonal is `7.3950e-6`. The polyline estimate is a geometry diagnostic, not a support certificate.

The `tol=1e-3` value mesh converged with all four active panels at depth zero. Before the change, gradient Gaussian replay ran only when the value mesh had deferred child panels. Therefore it did not run for this geometry-sensitive but unsplit mesh. Value convergence did not bound gradient accuracy. The close-root screen had identified the geometry, but its signal was not included in the replay gate.

## Change and measurements

When the per-epoch close-root geometry screen is active, the accepted value mesh now also gets the existing fixed-Gaussian tangent replay, even if it has no deferred children. The trigger is geometric; no derivative-error estimate or derivative-driven tolerance adaptation was added. The value calculation and public tolerances are unchanged.

| tolerance | before gradient `(t0, rho, q)` | after gradient `(t0, rho, q)` | node evaluations after |
|---:|---:|---:|---:|
| `1e-3` | `(-18.4818, 140.9901, 33.9735)` | `(-21.3339, 169.5170, 60.9727)` | 342 |
| `1e-4` | — | `(-21.3339, 169.5170, 60.9727)` | 534 |
| `1e-5` | — | `(-21.3550, 169.7276, 60.9369)` | 2030 |
| `1e-6` | — | `(-21.3550, 169.7276, 60.9369)` | 2696 |

At `tol=1e-3`, the revised gradient differs from the `tol=1e-6` result by 0.099% (`t0`), 0.124% (`rho`), and 0.059% (`q`). The special case uses 342 nodes instead of the old 136 (2.51x for this epoch).

A same-engine central finite difference at value tolerance `1e-7` gives `-21.33514`, `169.52858`, and `60.97420` for `t0`, `rho`, and `q`, respectively, at the representative small steps recorded in `case117_after.json`. The analytic replay agrees closely at those steps. This is a parity check, not an independent reference: finite differences use the same solver, the finite-difference result varies with step, and `support_certified` is false for the center, strict-tolerance evaluations, and perturbed evaluations.

An ordinary 401-epoch gradient control was measured in separate processes against a reconstructed pre-change extension and the new extension. Both used the same native extension SHA256 (`b6ba4c1a...99258d`), one pinned CPU, one warm-up, then five timed calls. Median time was 54.912 ms before and 54.823 ms after; both returned 2807 finite gradient components. This run shows no ordinary-case slowdown, but the sub-percent timing difference is within measurement noise and is not evidence of a speedup.

## Reproduction and artifacts

Run the public-API tolerance sweep, same-engine finite differences, geometry diagnostic, and node counts with:

```bash
python -B evidence/holonomic/case117_near_limb_20261016/case117_after.py
```

`case117_before.json` and `case117_after.json` record the old and revised gradients, finite differences, geometry, node counts, and extension hashes. `ordinary_control.json` records both ordinary-case timing samples and the shared input/native-build provenance. The old extension was reconstructed from the same generated API headers with only the geometry replay gate reverted; old and new extensions were loaded in separate processes to avoid Python/C++ module symbol interposition.

## Status

The large `tol=1e-3` discrepancy is substantially reduced for this case, and the ordinary control shows no measurable cost increase. Numerical acceptance remains bounded: all support checks are uncertified, and the finite-difference reference is same-engine. The evidence does not establish a general guarantee for every close-contact geometry.
