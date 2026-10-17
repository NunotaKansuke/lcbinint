# API timing audit: lcbinint and autodiff-vbm

The first three sections are historical measurements from 2026-10-16. They
compare the then-existing lcbinint value router with a separate
`holonomic_gradient()` route and are not measurements of the current unified
`LightCurve(..., with_gradient=...)` API. The separate route was removed during
the branch API consolidation. Current API measurements are appended below.

Date: 2026-10-16. Same checkout, CPU affinity, x64, `tol=retol=1e-3`, physical inputs and public light-curve calls as the main benchmark. The audit uses 15 steady repetitions per mode/case; JAX outputs are blocked before stopping the timer. First-call compile times are recorded separately. The repetitions are sequential, not paired, so differences of a few percent should be read as approximate.

## Findings

### lcbinint value and gradient calls follow different solver paths

With `finite_source_backend='algebraic_boundary'`, `LightCurve.__call__` delegates to `holonomic_info` and uses its value router. The router can return a fast point-source result: `ordinary_point` selected `point_source` for all 401 epochs. In contrast, `holonomic_gradient` computes trajectory geometry and loops over epochs in Python, calling `BinarySeededFusedGradient.evaluate` for each one. That route evaluates the seeded finite-source gradient and does not reuse the value router's result.

| Case | Epochs | Value only | Value + gradient | Ratio |
|---|---:|---:|---:|---:|
| ordinary_point | 401 | 0.361 ms | 55.646 ms | 154x |
| large_source_ld05_user | 401 | 155.821 ms | 273.442 ms | 1.76x |
| near_limb_contact_117 | 27 | 11.383 ms | 82.212 ms | 7.22x |
| near_double_root | 30 | 18.000 ms | 49.873 ms | 2.77x |

For `large_source_ld05_user` and `near_limb_contact_117`, the value router selected `algebraic_boundary_seeded_radial` for every epoch. `stratified_02` selected `hexadecapole` for all 25 epochs, while `stratified_00` selected the seeded radial value route for all 27. The two stratified cases were also rerun for 15 repetitions: 9.793 ms vs 1,245.081 ms (127x) for `stratified_00`, and 2.150 ms vs 139.676 ms (65.0x) for `stratified_02`. The latter is largely a value-selector versus gradient-route comparison; the former remains very slow even though its value route is already seeded radial, indicating substantial work in the gradient solver for that topology. These are end-to-end costs of the current public APIs; they are not the incremental cost of adding a derivative to an identical value algorithm. The very large ordinary-point ratio is mostly explained by point-source value evaluation versus the distinct finite-source gradient path.

### autodiff-vbm value + gradient is approximately the same speed as value only

A 15-repetition rerun on four cases found no substantial or systematic gradient speedup. Combined/value-only median ratios range from 0.984 to 1.033; p95 timings overlap. Summing these four per-case medians gives 814.2 ms with gradients versus 821.7 ms for values only, a difference under 1%. The medians are within 3.3%; with sequential, non-paired samples and no confidence intervals, this is not evidence that gradients are inherently faster.

| Case | Value only | Value + gradient | Combined/value ratio |
|---|---:|---:|---:|
| ordinary_point | 0.393 ms | 0.406 ms | 1.033 |
| large_source_ld05_user | 244.139 ms | 244.882 ms | 1.003 |
| near_limb_contact_117 | 556.320 ms | 548.459 ms | 0.986 |
| near_double_root | 20.834 ms | 20.493 ms | 0.984 |

The earlier three-repetition 12-case aggregate showed a small apparent win for value + gradient (1.281 s vs 1.368 s). The larger rerun indicates this is not a meaningful, general speed advantage.

### autodiff-vbm sometimes beats official VBM, but not on every case or at matched values

A 15-repetition value-only rerun on six cases gave 1.127 s summed per-case medians for autodiff-vbm and 1.501 s for official VBMicrolensing, a 25% reduction on this selected subset. The per-case result is strongly workload-dependent:

| Case | VBMicrolensing 5.6.1 | autodiff-vbm | AD/VBM |
|---|---:|---:|---:|
| ordinary_point | 0.384 ms | 0.399 ms | 1.04x |
| wide_caustic | 9.660 ms | 8.664 ms | 0.90x |
| large_source_ld09 | 808.184 ms | 268.939 ms | 0.33x |
| large_source_ld05_user | 619.508 ms | 244.484 ms | 0.39x |
| near_limb_contact_117 | 26.360 ms | 583.723 ms | 22.14x |
| near_double_root | 37.002 ms | 21.011 ms | 0.57x |

autodiff-vbm is a JAX implementation whose public vector light-curve function is JIT-compiled for steady-state timing; official VBM uses its separate native implementation. JAX compilation and vector execution make it plausible for autodiff-vbm to win on some batches. This is a comparison of implementations, not evidence that a wrapper around the same kernel is faster. It is also not accuracy-matched: the autodiff-vbm checkout vendors VBM 5.5 while official VBM is 5.6.1, and equal `Tol`/`RelTol` settings do not force the two solvers to return the same values.

Direct per-epoch value differences, relative to official VBM:

| Case | Median relative difference | 95 percentile | Maximum |
|---|---:|---:|---:|
| ordinary_point | 5.55e-14% | 3.87e-13% | 7.86e-13% |
| wide_caustic | 9.83e-13% | 5.37e-11% | 6.47e-10% |
| large_source_ld09 | 0.105% | 0.144% | 0.150% |
| large_source_ld05_user | 0.0839% | 0.200% | 0.211% |
| near_limb_contact_117 | 0.0281% | 0.816% | 21.65% |
| near_double_root | 0.0279% | 0.414% | 0.506% |

The contact discrepancy is concentrated in difficult epochs; these are solver-to-solver differences, not errors against a high-resolution truth calculation. Thus the timing lead on the two large-source cases cannot be called a same-accuracy win yet.

## 2026-10-17 pre-fix unified API timing and derivative-route audit

This subsection records the implementation before the fused-kernel restoration
below. It is historical after the 2026-10-17 code change.

These measurements use the pinned library checkouts and the current public APIs. They do not combine the historical lcbinint `holonomic_gradient()` route with the current `LightCurve(..., with_gradient=True)` route. x64 CPU, one-core affinity, synchronized JAX output; steady timings use 15 repetitions. First-call compilation is stored separately in `issue6_7_gradient_api_audit_20261017/metadata.json`.

### autodiff-vbm value-only versus value plus full Jacobian

Value-only and value plus all seven derivatives were interleaved in 15 alternating pairs per case. Magnification outputs matched exactly in all four cases.

| Case | Value median | Value + gradient median | Ratio |
|---|---:|---:|---:|
| ordinary_point | 0.391 ms | 0.399 ms | 1.019 |
| large_source_ld05_user | 243.717 ms | 244.740 ms | 1.004 |
| near_limb_contact_117 | 533.955 ms | 533.163 ms | 0.999 |
| near_double_root | 20.863 ms | 20.410 ms | 0.978 |

The sum of these selected per-case medians is 798.926 ms for value-only and 798.711 ms for value plus gradient (ratio 0.9997). This does not reproduce the earlier 6.03x ratio or show a systematic gradient speed advantage.

The four cases above span different geometries, not a matched limb-darkening on/off comparison; the original audit does not document why those exact four were selected. They also measure the external `autodiff-vbm.BinaryLightCurve` API, not this branch's default `lcbinint.LightCurve(..., with_gradient=...)` API. A separate matched pair uses the same large-source parameters and 401 shared epochs, changing only the profile from uniform to linear limb darkening with `a1=0.5`. The serialized epoch lists differ by at most `3.33e-16 d`; both profiles were evaluated on the exact same uniform-case epoch array.

| Profile | Value only | Value + full 401x7 Jacobian | Gradient/value |
|---|---:|---:|---:|
| Uniform | 17.813 ms | 24.199 ms | 1.358 |
| Linear LD, `a1=0.5` | 244.143 ms | 244.772 ms | 1.003 |

With LD, the value-only call is `13.71x` slower than the uniform case, while value plus gradient is `10.12x` slower. In the uniform case, adding the full Jacobian costs `6.386 ms` (`35.8%`); with LD it costs `0.630 ms` (`0.26%`). Thus the external `autodiff-vbm` case's ratio moves toward one after adding LD, but this does not reproduce or identify the provenance of the historical `6.03x` result. The Python FFI selects `include_jacobian=False` for values and `include_jacobian=True` for the custom JVP ([kernel](/rogue1_8/nunota/moasarc-ecosystem/integration/autodiff-vbm/src/autodiff_vbm/_kernel.py:216), [limb wrapper](/rogue1_8/nunota/moasarc-ecosystem/integration/autodiff-vbm/src/autodiff_vbm/_limb.py:69)). Native source shows the concrete difference: uniform values use `finite_contour_primal`, while its Jacobian path uses the Jet-based `finite_contour_magnification` ([native kernel](/rogue1_8/nunota/moasarc-ecosystem/integration/autodiff-vbm/native/vbm_ffi.cpp:1065)); linear-LD value and Jacobian FFI both call `evaluate_limb_epoch`, which returns a Jet result, and the value FFI discards its derivative lanes ([native LD FFI](/rogue1_8/nunota/moasarc-ecosystem/integration/autodiff-vbm/native/vbm_limb_ffi.cpp:1682)). Therefore LD values-only already pays for derivative-lane arithmetic, directly explaining the `1.003x` ratio. Uniform uses separate scalar and Jet contour work, consistent with its `1.358x` ratio.

This external result is not the current branch's algebraic-boundary API. Before restoring the fused route, the same saved pair was measured through the then-current `lcbinint.LightCurve` implementation with the branch default resolving to `finite_source_backend='algebraic_boundary'`:

| Profile | Value only | Value + gradient | Gradient/value | Value status | `gradient_valid` |
|---|---:|---:|---:|---|---:|
| Uniform | 306.560 ms | 802.203 ms | 2.617 | unconverged 401/401 | 0/401 |
| Linear LD, `a1=0.5` | 354.528 ms | 1,052.785 ms | 2.970 | unconverged 400/401 | 0/401 |

This is the pre-fix snapshot: the ratio did not move toward one. Both profiles returned finite magnifications; values from the two modes and status arrays matched exactly. These were failed-convergence timing probes, and every gradient row was marked invalid, so they do not establish scientific usability or production performance. The implementation below now dispatches supported eager calls to the existing fused native kernel.

In that pre-fix implementation, the API's single call surface did not mean a single evaluation internally: `LightCurve.__call__` invoked `_evaluate_values` and then `_evaluate_gradients`; the latter ran `jax.jacfwd`. The ordinary value-only call returned before entering the gradient function, while the value-plus-gradient call did both evaluations. The pre-fix algebraic-boundary FFI's ordinary primal used `include_jacobian=False`; its JVP requested the Jacobian output. Those duplicate passes were the implementation error corrected below.

### Forward versus reverse mode for the full light-curve Jacobian

The same vector output and full 27x7 or 30x7 Jacobian were measured for `jacfwd` and `jacrev` on two cases. The mode order alternated during 15 steady repetitions.

| Case | Library | `jacfwd` median | `jacrev` median | Reverse / forward | Max relative gradient difference |
|---|---|---:|---:|---:|---:|
| near_limb_contact_117 | autodiff-vbm | 526.507 ms | 528.486 ms | 1.004 | 1.6e-10 |
| near_limb_contact_117 | microlux | 691.137 ms | 1,567.668 ms | 2.268 | 2.25e-3 |
| near_double_root | autodiff-vbm | 20.378 ms | 20.370 ms | 1.000 | 5.0e-12 |
| near_double_root | microlux | 874.877 ms | 2,207.555 ms | 2.523 | 2.4e-8 |

These measurements are for vector outputs: 27 or 30 magnifications and their complete Jacobians. They do not test the scalar-output case behind the usual reverse-mode rule of thumb. A matched scalar follow-up selects one epoch from the same 27-point contact input, returns `mag[0]` (function shape `[]`, gradient shape `[7]`), and measures 15 alternating pairs:

| Library | Output | `jacfwd` median | `jacrev` median | Reverse / forward | Max relative gradient difference |
|---|---|---:|---:|---:|---:|
| autodiff-vbm | one scalar | 0.865 ms | 0.869 ms | 1.004 | 7.21e-13 |
| MicroLUX | one scalar | 35.840 ms | 24.409 ms | 0.681 | 3.26e-4 |

For this one scalar MicroLUX evaluation, reverse mode is `31.9%` faster, while its full 27-output Jacobian had reverse mode `2.27x` slower. autodiff-vbm is essentially tied in both shapes, consistent with its custom-JVP FFI path producing native Jacobian data for the differentiated call. The scalar MicroLUX run emitted a sampling-capacity warning (581 requested/current samples versus a limit of 478); this timing does not certify gradient accuracy. The mode-dependent gradient difference is recorded as a diagnostic, not evidence that either gradient is correct.

The MicroLUX source checkout was at `711245b6`, but its native `.so` is an untracked local artifact with SHA-256 `8819fcf5…447e9`; its build provenance is not recorded. That exact artifact hash is saved in the new metadata. The earlier vector audit did not save a native-binary hash, so equality of the binary used by the two historical measurements cannot be independently verified.

Therefore the vector results support forward mode only for those measured vector shapes. The scalar observation agrees with the usual input/output dimensionality heuristic for MicroLUX, but it is one input and one implementation; JAX batching and solver internals also affect actual timings. MicroJAX's CPU public API supports and recommends forward mode; reverse mode through its CPU scheduler is outside that API. lcbinint provides one `LightCurve(..., with_gradient=...)` call surface; supported eager static-binary calls use the existing fused native value/Jacobian kernel, while other calls use the JAX differentiation route.

## 2026-10-17 restored fused native route

For supported eager static-binary calls, `LightCurve(...,
with_gradient=True)` now calls the existing `BinarySeededFusedGradient.evaluate`
kernel. The former public gradient function remains removed. This corrects the
pre-fix path that evaluated the value and then ran `jax.jacfwd` as a second
solver pass.

The matched 401-epoch uniform and linear-LD inputs were rerun in 15 alternating
pairs on one CPU core at `tol=reltol=1e-3`, with the same shared epoch array.
Uniform medians were 306.483 ms (value-only) and 229.652 ms (value plus
gradient), ratio 0.749. Linear LD `a1=0.5` medians were 355.161 ms and 265.129
ms, ratio 0.747. All 401 magnifications and gradients were finite, and
`gradient_valid` was true for 401/401 rows.

This is not a like-for-like gradient overhead measurement: value-only uses the
algebraic-boundary integrator, while value plus gradient uses the seeded radial
fused kernel. Maximum relative value differences were 1.366e-2 (uniform) and
7.525e-3 (LD). The value-only route was unconverged on 401/401 uniform rows and
400/401 LD rows. The fused route reported value convergence for 401/401 rows,
but support certification for 0/401 rows, so all its statuses were
`support_unproven`. The timing confirms that the separate `jax.jacfwd` pass is
gone; it does not certify value parity, support, or gradient accuracy.

The historical internal matched audit on 7,216 saved paper inputs reported a
fused/value summed-median ratio of 1.239 (uniform median 1.201, linear-LD
median 1.370). That audit used the seeded-radial value route on both sides and
is not the same timing boundary as the current public API comparison above.

### Why autodiff-vbm's LD timing ratio approaches one

The matched `autodiff-vbm` result is explained by its native source. The
uniform forward-value handler uses `finite_contour_primal`, while the Jacobian
handler uses `finite_contour_magnification`, which propagates Jet derivatives
through the contour walk. The linear-LD forward and Jacobian handlers both
call `evaluate_limb_epoch` and obtain a Jet result; the forward handler reads
only `.value`, while the Jacobian handler also copies its derivative lanes.
Thus the LD value-only measurement already pays for native derivative
propagation, explaining the near-one ratio. This is confirmed by source in
`autodiff-vbm` revision `69002603035eaddcdbad3d3b62e4f570a7cf9b85`; the
measured source hashes are recorded in
[OPEN_ISSUES.md](OPEN_ISSUES.md).

The restored-route measurement, arrays, and pinned extension hashes are in
[issue6_lcbinint_fused_api_20261017](issue6_lcbinint_fused_api_20261017/metadata.json)
and can be reproduced with
[`audit_lcbinint_fused_api_20261017.py`](audit_lcbinint_fused_api_20261017.py).

The current `stratified_09` timings and per-epoch status are in `issue5_stratified_09_remaining_apis_20261017/summary.json`; VBM and autodiff-vbm did not complete that input within the bounded attempts recorded in `OPEN_ISSUES.md`.

## Provenance

- AD/LC timing: `autodiff_lc_timing_audit_15x/summary.json` and `raw/*.npz`.
- Official VBM/AD value timing: `vbm_autodiff_value_audit_15x_clean/summary.json` and `raw/*.npz`.
- Stratified LC timing: `stratified_lc_timing_audit_15x/summary.json` and `raw/*.npz`.
- Reproduction scripts: `benchmark_timing_audit.py` for AD/LC and `stratified_lc_timing_audit_15x`; `benchmark_timing_audit_no_lc.py` for VBM/AD (15 steady repetitions; the main `benchmark.py` was preserved). The preliminary `vbm_autodiff_value_audit_15x/` folder is superseded: its skip filter did not suppress the later LC calls, so the clean folder is used for conclusions.
- Historical route inspection: the pre-consolidation `python/lcbinint/holonomic_fused.py` and `src/lcbinint/magnification/holonomic/fused_gradient.cpp` snapshots; those routes are no longer part of the active branch API.
- Current-route audit: `audit_gradient_routes_20261017.py`, `issue6_7_gradient_api_audit_20261017/metadata.json`, and its `raw/` Jacobian arrays. External matched LD and scalar output follow-up: `audit_ld_and_scalar_jacobian_20261017.py`, `issue6_ld_scalar_output_audit_20261017/metadata.json`, and `matched_ld_raw.npz`. Branch-default matched LD follow-up: `audit_lcbinint_matched_ld_20261017.py` and `issue6_lcbinint_matched_ld_20261017/metadata.json`. For other current lcbinint evidence use `benchmark_algebraic_boundary_api.py` and the `issue5_stratified_09_*_20261017/` outputs.

The timing audit itself made no solver changes. The separate #5 finite-value
contract correction is documented in `OPEN_ISSUES.md` and verified with the
current branch API; it changes the Python diagnostic fallback only.
