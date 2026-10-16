# Near-double-root gradient regression check (2026-10-16)

## Result

The historical `t=-0.7 d` gradient spike is absent in the current default
algebraic-boundary gradient build for the checked near-double-root window. The
same default path also resolves the neighboring `t=-0.696 d` failure. No
case-specific constants or new production-code patch were needed in this check:
the generic close-pair geometry treatment is already enabled by the current
branch, including the depth-zero replay fix at HEAD `65faef7a3f5ade838119c39b77634a5ccbdf1ee3`.

| Epoch | Historical default `dmu/du0` | Current default, tol=1e-3 | VBM FD median | Current vs VBM |
|---:|---:|---:|---:|---:|
| -0.700 d | -5.7882218 | 0.393246554 | 0.393210291 | +0.00922% |
| -0.696 d | -9.7891368 | 0.394351690 | 0.394323783 | +0.00708% |

The historical values and VBM finite-difference references are preserved in
[`gradient_geometry_20261015/comparison.csv`](../gradient_geometry_20261015/comparison.csv).
The old experimental-prototype table in that directory's README is not a
measurement of this current default build; it is retained as historical
research evidence.

A tolerance sweep (`tol=retol=1e-3, 1e-5, 1e-6`) gives `dmu/du0` at `-0.700 d`
of `0.393246554`, `0.393238227`, and `0.393238225`, respectively. At `-0.696 d`
the corresponding values are `0.394351690`, `0.394339004`, and `0.394339008`.
All five checked epochs report `value_converged=true` and finite usable
gradients. The largest absolute relative difference from the saved VBM
finite-difference medians over all compared parameters at `-0.704, -0.700,
and -0.696 d` is 0.0232% (`s` at `-0.704 d`).

A 65-epoch scan from `-0.706` to `-0.690 d` has finite `u0` gradients spanning
`0.3913404` to `0.3967674`; a separate 21-epoch scan from `+0.775` to `+0.785 d`
has finite `u0` gradients spanning `0.4287866` to `0.4348618`. Across all 89
scan epochs, all 623 gradient components are finite and there are no invalid
epochs. This checks for a renewed spike over these sampled windows; it is not a
continuous-domain guarantee.

## Cause and current handling

At `-0.700 d`, the diagnostic D14 roots contain a close complex-conjugate
radial pair. At `-0.696 d`, the corresponding pair is two real folds; the
value-seeded topology omitted both events. The local angular arc width has a
square-root form near the pair, so its parameter derivative can vary sharply
even while its contribution to the magnification is small. A value-only
quadrature estimate can therefore accept a mesh that is too coarse for the
gradient.

The current default gradient build screens D14 geometry, isolates roots in
regions where the screen cannot settle the topology, and replays the close-pair
contribution with a regularizing radial map. At special epochs it also replays
the smooth background using the value-selected quadrature level. The latest
unsplit-mesh fix lets this geometry-gated gradient replay run even when the
value integrator accepted depth-zero panels. Ordinary epochs keep the regular
path. The trigger comes from root geometry, not a measured gradient error; no
derivative tolerance or derivative-driven adaptivity was introduced.

The implementation is already in the existing default build path:

- [`fused_gradient/README.md`](../../../benchmarks/holonomic/fused_gradient/README.md#default-close-pair-treatment-2026-10-15)
- [`build_api_overlay.py`](../../../benchmarks/holonomic/fused_gradient/build_api_overlay.py#L22-L24)
- [`build_geometry_overlay.py`](../../../benchmarks/holonomic/fused_gradient/build_geometry_overlay.py#L18-L67)
- HEAD commit: `65faef7a` (`Fix close-root gradient replay on unsplit meshes`)

## Reproduction and provenance

The check used the existing project-local compiled modules from HEAD above.
Their SHA-256 hashes are recorded in `current_validation.json` and
`alpha_vbm_fd.json`. The tolerance sweep and 89-epoch scan can be regenerated
with:

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python evidence/holonomic/gradient_near_double_root_20261016/verify_current.py
```

The independent VBM central-difference sweep for `alpha` should be generated
first (the current-validation script reads its saved result):

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python evidence/holonomic/gradient_near_double_root_20261016/verify_alpha_fd.py
```

That sweep uses VBM `Tol=RelTol=1e-10`, linear limb darkening `a1=0.5`, `a2=0`,
and central-difference steps `1e-4`, `3e-5`, and `1e-5`. VBM references for
`u0`, `rho`, `q`, and `s` come from the saved step-study table linked above.
For a fresh build, run `bash benchmarks/holonomic/fused_gradient/build_algebraic.sh`
and restart Python/Jupyter before running the scripts.

## Limits

Every current API result in this check reports `support_certified=false`. The
finite values, tolerance stability, dense sampled scan, and independent VBM
finite differences establish that the old spike is not present in these tested
inputs and epochs; they do not constitute a formal bound on gradient error or
a certification for all close-root configurations. This close-root regression
is resolved in the tested case, while global support certification remains
open.
