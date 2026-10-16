# Stratified boundary follow-up — 2026-10-16

## Checkout and scope

- Checkout: `/rogue1_8/nunota/lcbinint/experiments/algebraic_boundary`
- Branch/HEAD: `experiment/algebraic_boundary` / `65faef7a3f5ade838119c39b77634a5ccbdf1ee3`
- The existing notebook and `python/lcbinint/holonomic_fused.py` edits were preserved.
- Both inputs are linearly limb-darkened (`Gamma=0.9` and `Gamma=0.5`), not uniform-source (`Gamma=0`).

## Changes

- `source_disk_gradient.hpp`: on a failed-gradient repair only, refine the certified continuous-caustic cover from 512 through 2048 to 8192 phase intervals. This lets the smooth source-disk derivative route prove positive clearance in `stratified_02`; caustic-intersecting disks remain rejected.
- `adaptive_epoch.hpp`: with `fold_maps=false`, keep a refined event estimate as an integration split when available, but do not require its fold-endpoint uncertainty. This map does not use the singular fold substitution.
- `fused_gradient.cpp`: only unresolved epochs retry through full D14 topology with the regular radial map at initial level 8. An `EventLocationLimited` result adopts the retry gradient only when the retry converges. For `TopologyUnresolved`, a converged retry may update the value while the original gradient remains invalid.

No derivative-error tolerance was added; the retry uses the caller's value tolerances. Support and gradient-precision certification flags remain false.

## Results

### `stratified_02`

At the eight epochs `[-0.275, -0.25, -0.225, -0.2, -0.075, -0.025, 0, 0.025]` days, all 8/8 epochs report converged values and valid gradients at each of `tol=reltol=1e-3, 1e-5, 1e-7`.

At `t=-0.275 d`, analytic versus central finite differences from the ordinary value path gave maximum relative differences across two step sizes of: `t0 1.67e-7`, `u0 8.0e-9`, `rho 2.07e-4`, `q 3.45e-7`, `s 1.44e-8`, and `alpha 8.63e-9`. The largest discrepancy is `rho`, about `0.021%`.

### `stratified_09`

Input: `s=0.95`, physical `q=1e-7` (solver-frame `q=1e7`), `rho=1e-6`, linear limb coefficient `0.5`; epochs are `[-1.7455113096375089e-6, 0, +1.7455113096375089e-6]` days.

| tol=reltol | converged values | valid gradient epochs | stop at negative / center / positive contact |
|---:|---:|---:|---|
| `1e-3` | 3/3 | 2/3 | `Converged / Converged / TopologyUnresolved` |
| `1e-5` | 3/3 | 1/3 | `Converged / TopologyUnresolved / TopologyUnresolved` |
| `1e-7` | 1/3 | 0/3 | `EventLocationLimited / TopologyUnresolved / TopologyUnresolved` |

At `tol=1e-3`, the negative-contact `t0` gradient is `-4.48766396e10`; the no-fold value-route one-sided differences at `h=1e-9` are `-4.50134764e10` and `-4.50316979e10` (about `0.32%` from their mean). At `tol=1e-5`, the returned gradient is `-4.46547588e10`; the same-route finite-difference mean is about `-4.5021e10` (about `0.81%`). These are finite/usable gradients, not error-certified gradients.

At the positive contact, the no-fold value route's one-sided `t0` slopes (in `1e9` units) move apart as the step shrinks:

| h (days) | negative-side slope | positive-side slope |
|---:|---:|---:|
| `1e-7` | 20.21 | 16.81 |
| `3e-8` | 21.83 | 15.31 |
| `1e-8` | 27.93 | 9.61 |
| `3e-9` | 47.80 | -10.84 |
| `1e-9` | 106.39 | -69.43 |

The symmetric averages stay near `18.5e9`, while the one-sided slopes do not show a common finite limit. This is strong evidence that the classical derivative at this exact limb-contact point is undefined (or at least not numerically resolvable by this value path). The implementation therefore recovers the value but keeps the gradient invalid; it does not substitute the symmetric derivative as an undocumented convention. This diagnosis is evidence-based but not a formal proof of non-differentiability.

## Verification

- Rebuilt with `bash benchmarks/holonomic/fused_gradient/build_algebraic.sh`.
- `build-algebraic-boundary/fused-gradient/api_check` passed coexistence, analytic-limit, and invalid-input checks.
- Re-ran the public `LightCurve.holonomic_gradient` API on both cases and all three tolerances.
- `git diff --check` passed.

The positive exact-contact gradient and the strict `1e-7` negative-contact result remain unresolved. No commit or push was made.

## Implementation review follow-up

- The regular-map retry is generic by stop reason and uses complete D14 topology; it has no case identifiers. It still uses a fixed `initial_level=8`, so its exceptional-case cost is a conservative empirical choice, not a derivative-accuracy guarantee.
- The earlier generic `TopologyUnresolved` value retry did not promote gradients and ran only when the original value was unconverged. The later edge-pair route recheck below adds a narrowly certified exception: a completed frontier plus an interval-certified upper-edge pair may use the regular chart; the exact positive contact fails that certificate and remains invalid.
- The refined caustic enclosure is cached by lens geometry. Repeated failed epochs now scan the finest cached enclosure once instead of rescanning the same 8192 disks at the 512, 2048, and 8192 loop entries.
- Current route recheck: `stratified_02` remains 8/8 value-converged and 8/8 gradient-valid at all three tolerances. `stratified_09` is 2/3 gradient-valid at `1e-3` and `1e-5`; at `1e-7` only the positive-contact value converges and no gradient is valid. The exact positive contact stays invalid.

### Same-session timing check

One pinned CPU, LD=.5, `tol=reltol=1e-3`, setup excluded, warmup then median of three calls. The retained `before.so` is an older archived module, so this is a same-session broad comparison rather than a build-identical measurement of only the last patch.

| Input | Archived `before.so` ms | Current ms | Current result |
|---|---:|---:|---|
| ordinary, 401 epochs | 157.0 | 159.0 | 401/401 valid gradients |
| large source, 401 epochs | 798.1 | 522.8 | 401/401 valid gradients |
| exact contact, 3 epochs | 54.8 | 229.2 | values 3/3, valid gradients 2/3 |

The exact-contact retry is about 4.2x slower than the archived module on this three-point input, while improving value convergence from 2/3 to 3/3 and valid gradients from 1/3 to 2/3. It is confined to unresolved epochs. Absolute timings differ substantially from the October 15 report in the current execution environment, so only the same-session paired comparison is used here.

### Current edge-pair route recheck

The completed seeded frontier carries a certified isolated D14 edge pair, but its sparse event list may not reproduce that pair through `known_upper_edge_pairs`. The old public gate required both flags, so it skipped the special chart. The corrected gate uses the completed frontier pair plus one of: a usable incumbent gradient, `EventLocationLimited`, or a separately interval-certified pair in the topology event list. The last condition recovers the strict center while keeping the exact positive contact fail-closed: its pair does not pass that interval certificate.

At `tol=reltol=1e-3`, the rebuilt public API returned valid gradients:

- Negative contact: `t0=-4.4992155638594772e10`. Same-build no-fold central differences were `-4.5024007879391624e10` at `h=3e-9` and `-4.5023984566912987e10` at `h=1e-9`; the latter differs by `0.07069%`. Both shifted values converged.
- Center: `t0=-1.3320962692839111e10`. Same-build no-fold central differences were `-1.3339207961282227e10` at `h=1e-8` and `-1.3339274463360197e10` at `h=1e-9`; the latter differs by `0.13728%`. Both shifted values converged.
- Positive exact contact remains `TopologyUnresolved` with invalid gradients.

At `tol=reltol=1e-5`, both center and negative contact also returned valid gradients. Their same-build no-fold central differences at `h=1e-9` were `-1.3332643047673628e10` and `-4.5020319461007599e10`, respectively; relative differences are `0.17521%` and `0.09513%`. Both shifted values converged. The positive exact contact remained invalid.

At `tol=reltol=1e-7`, the negative value still stops at `EventLocationLimited` and the center value at `TopologyUnresolved`; neither gradient is adopted because the value has not converged. The positive exact-contact gradient remains invalid. A regular non-contact public-API control still converged with `retry=0`, 21 node evaluations, and all five gradients valid. `api_check` passed after rebuilding.
