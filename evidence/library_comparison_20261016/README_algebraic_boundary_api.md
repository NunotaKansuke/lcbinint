# Current algebraic-boundary API comparison

`benchmark_algebraic_boundary_api.py` reads the saved 12-case input set and
also includes the preserved `stratified_09` input. For every selected
lcbinint case it measures
`curve(times, params, with_gradient=False)` and
`curve(times, params, with_gradient=True)` on the same curve. It asserts that
the branch default resolves these static single-source binary cases to
`algebraic_boundary`, and stores the per-epoch value status, support state,
selected method, stop reason, gradient validity, values, and gradients.

This backend retains its point-source and hexadecapole fast paths in the
shared selector. Those epochs are recorded by method name; the backend excludes
the standard Cartesian and polar inverse-ray routes, while deep finite-source
epochs use the image-boundary integral.

If image-boundary support is unproven before a finite-source estimate is
available, `LightCurve` returns a finite point-source fallback. The epoch keeps
`status="support_unproven"`, diagnostics identify
`value_fallback_kind="point_source"`, and its gradient is invalid. This is not
accepted as a finite-source value. Value-only and value-plus-gradient modes
return the same estimate.

The script pins MicroJAX, MicroLUX, autodiff-vbm, and the archived official VBM
5.6.1 package to the source paths/revisions recorded with the comparison. This
avoids importing unrelated editable worktrees or the environment's default VBM
5.5 package. The other-library calls remain each library's own implementation;
only lcbinint's route is fixed to algebraic boundary. The previous
[`benchmark.py`](benchmark.py) and its saved outputs remain the historical
comparison; its lcbinint gradient measurement used `holonomic_gradient()` and
must not be combined with this current API run.

Use this runner for new comparisons. The older benchmark and timing-audit
scripts are preserved as historical provenance; their lcbinint gradient data
uses `holonomic_gradient()` and is not a current API benchmark.

By default, outputs go to `algebraic_boundary_api_20261017/` beside the script.
The old raw directories are not used or overwritten. A case subset can be
selected with `LCBININT_BENCH_CASES`, for example:

```sh
LCBININT_BENCH_CASES=ordinary_point \
LCBININT_BENCHMARK_OUT=evidence/library_comparison_20261016/algebraic_boundary_api_smoke_20261017 \
python evidence/library_comparison_20261016/benchmark_algebraic_boundary_api.py
```

The full 12-case run has not been executed. `stratified_09` has been measured
with the current API. Per-epoch isolation obtained values from all five
implementations on 24/27 epochs. The lcbinint
value/fallback contract was checked over all 27 epochs in both API modes; the
values and statuses match exactly, with 24 accepted rows and three
`support_unproven` point-source fallbacks. Per-epoch VBM 5.6.1 and autodiff-vbm
calls finish on neighboring epochs but exceed bounded 2 s / 5 s limits at the
three central epochs. The two VBM-family implementations match within
`2.21e-9` relative on their 24 completed epochs, but are not independent
references. The center values and an independent reference remain unresolved.
Partial data are preserved under
`issue5_stratified_09_remaining_apis_20261017/`,
`issue5_stratified_09_fallback_verification_20261017/`, and
`issue5_vbm_epoch_isolation_20261017/`; the matched 24-epoch summary is in
`issue5_stratified_09_24_epoch_comparison_20261017/`.

A separate MicroLUX capacity audit raised `default_strategy` to a total length
of 6000. Its three center values moved to within `0.112%` of MicroJAX and no
capacity warning was emitted. However, `return_info=True` reports a very large
estimated error sum for the final annulus (`3.70e8` to `5.18e9` at tolerance
`1e-3`); this state does not certify the limb-darkened total. Treat the values
as a diagnostic improvement only, not as converged finite-source references.
The raw diagnostic and state are preserved in
`issue5_microlux_capacity_6000_info_20261017/metadata.json`.

The runner's first-call timing includes compilation where the public API
creates a new JAX graph; steady-state timing is reported separately.
