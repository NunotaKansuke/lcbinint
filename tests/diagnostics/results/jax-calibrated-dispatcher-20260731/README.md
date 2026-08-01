# JAX calibrated binary dispatcher stress result (2026-07-31)

This result freezes the calibration evidence for the opt-in
`binary_magnification_calibrated` scalar API.  The dispatcher uses native
`lcbinint` routing diagnostics and the native 14-bucket source-resolution
selector, then executes differentiable JAX point-source, hexadecapole,
Cartesian, polar, or source-plane kernels.  Routing decisions are
stopped-gradient; the selected physical kernel is not.

## Accuracy sweep

The seeded sweep used ten close/resonant/wide binary lenses, six positions per
lens (five caustic-normal distances and one field point), and uniform, linear,
and square-root source profiles: 180 rows total.  Native Cartesian
coarse/fine convergence and Cartesian/polar agreement trusted 174 rows.

All 174 trusted rows passed the common absolute-plus-relative error budget and
reported valid, converged support.  The selected method counts were:

| Method | Code | Rows |
| --- | ---: | ---: |
| hexadecapole | 0 | 45 |
| Cartesian inverse ray | 1 | 37 |
| polar inverse ray | 2 | 26 |
| source-plane chord 48/96 | 3 | 39 |
| point source | 4 | 27 |

The median scalar wall time was 31.92 ms, including per-row geometry and
routing reconstruction.  Method medians were 4.85, 219.06, 936.17, 32.49,
and 5.29 ms in code order.  These are single-host measurements, not portable
performance guarantees.  The polar tail deliberately pays for a 65,536-angle
floor needed to remove a shared coarse/fine bias at a small-mass-ratio grazing
cusp.

The sweep is reproducible with:

```sh
python tests/diagnostics/jax_ir/sweep_dispatcher.py \
  --lens-cases 10 --points-per-case 6 \
  --profiles uniform,linear,square_root \
  --caustic-bins 512 --native-coarse-bins 128 \
  --native-fine-bins 256 --gradient-stride 2 \
  --repeat 1 --seed 20260726 --output sweep.json
```

The runner writes each completed row atomically and supports `--resume`.

## Representative positioning and microLUX

The final positioning run covered regular, resonant-cusp, planetary-far, and
planetary-cusp geometries with uniform and linear profiles.  All eight core
jobs completed under a 240 s per-job limit and the calibrated JAX result
passed its accuracy budget.  Warm calibrated JAX forward/JVP/gradient ranges
were 2.43--215.30, 3.14--280.27, and 12.22--513.68 ms.

microLUX was isolated into separate children.  With 80 annuli, all four linear
jobs completed and passed: whole-job times were 69.31--74.08 s, while warm
forward/JVP/gradient ranges were 7.58--331.72, 12.83--399.81, and
25.50--519.46 ms.  All four uniform jobs reached the realistic 60 s whole-job
limit and are recorded as lower bounds, not discarded measurements.  The
80-annulus configuration is intentionally the common limb-darkening reference
and is not an optimized uniform-source baseline.

Reproduce or resume the isolated runs with:

```sh
python tests/diagnostics/jax_ir/stress_positioning.py \
  --cases regular,resonant_cusp,planetary_far,planetary_cusp \
  --profiles uniform,linear --operations core \
  --timeout-core 240 --repeat 1 --inner 1 --output core.json

python tests/diagnostics/jax_ir/stress_positioning.py \
  --cases regular,resonant_cusp,planetary_far,planetary_cusp \
  --profiles uniform,linear --operations microlux \
  --timeout-uniform 60 --timeout-limb 120 \
  --repeat 1 --inner 1 --output microlux.json
```

## AD checks

Five far-field point-routed source-coordinate gradients agreed with native
central differences to \(6.1\times10^{-11}\)--\(2.3\times10^{-8}\).
Representative Cartesian and polar gradients passed their native-difference
budgets.  The source-direct tangent check requires a local
\(h=3\times10^{-7}\) step: JAX gave -1279.103466397 and native
-1279.103466327.  A wider step crosses the limb-contact layer and is not a
derivative reference.  The hexadecapole check is limited by native Cartesian
grid noise; forced native Cartesian-1024 at \(h=9\times10^{-5}\) gave
-8.95855 versus JAX -8.96530.

## Failure semantics

- A JAX point-root failure prevents point acceptance.
- A source-direct root or convergence failure falls back to the independently
  checked image-plane path.
- Cartesian/polar values require valid support and an independent adjacent
  resolution comparison.
- If the lower comparison has invalid support or changes method, the next
  higher bucket is used.
- Unsupported or non-converged values are exposed through result flags; no
  partial support is silently accepted.

