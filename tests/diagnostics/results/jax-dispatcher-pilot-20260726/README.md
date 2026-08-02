# JAX binary dispatcher pilot calibration

This pilot is the first stratified holdout for the JAX CPU magnification
dispatcher.  The collection command was:

```bash
PYTHONPATH=python python tests/diagnostics/jax_ir/sweep_dispatcher.py \
  --output /tmp/jax-dispatcher-pilot-20260726.json \
  --lens-cases 8 --points-per-case 6 \
  --profiles uniform,linear,square_root \
  --caustic-bins 512 \
  --native-coarse-bins 128 --native-fine-bins 256 \
  --gradient-stride 18 --repeat 2
```

The eight lenses cover close, resonant, and wide geometries, mass ratios from
`1e-5` to `1`, and source radii from `1e-4` to `2e-2`.  Each lens contributes
source centres at `0.5`, `1`, `2`, `5`, and `20` source radii from a sampled
caustic plus one field control.  The three profiles produce 144 rows.  A
reference is trusted only when native Cartesian 128-to-256-bin convergence and
native Cartesian/polar 256-bin agreement both fit twice the common
`1e-4 + 1e-4 max(|A|, 1)` budget.  This accepted 141 rows.

`baseline-rows.json.gz` contains the complete pre-calibration rows.  Its SHA256
is `231dabcb92c7cc1cc6cb5d896071fa36f2ad95f934c5f75aa7d5d285182bba53`.

## Findings

- The hexadecapole guard had zero false accepts for every tested safety factor
  from 2 through 64.  Safety factor 4 accepted 60 baseline rows.
- The original automatic dispatcher returned 60 hexadecapole, 39 Cartesian,
  and 42 polar results.  Thirty-six results had invalid support.  Among
  support-valid results, 12 exceeded the accuracy budget: four Cartesian and
  eight polar.
- Broad polar use is unsafe.  The 64/4096 polar configuration failed accuracy
  or support on 131 of 141 trusted rows.  Increasing radial-band capacity from
  4 to 16 reduced overflow but left 129 failures and nearly tripled median
  latency.  The issue is coordinate suitability and angular convergence, not
  radial-band capacity.
- Cartesian 64/1024 failed or overflowed on 66 rows after the root fallback was
  added.  Expanding to 64/4096 reduced this to 56; 128/16384 reduced it to 32
  but raised median latency from 9.7 to 68.8 ms.  Capacity and resolution must
  therefore be separate buckets.
- The bounded Ehrlich--Aberth solver lost physical limb roots for extreme
  planetary mass ratios.  A companion-matrix fallback recovered 3/5 images for
  all frozen failing limb samples.

## Dispatcher changes derived from the pilot

The production hybrid dispatcher no longer sends arbitrary Cartesian overflow
to polar.  Polar preselection now additionally requires stable image topology
across the thirteen multipole samples and a symmetric mass ratio of at least
`5e-3`.  Unsupported Cartesian results are returned as `support_valid=False`
with a `NaN` hybrid magnification instead of a plausible biased value.

Replaying the 141 trusted rows with those changes selected 69 hexadecapole and
72 Cartesian results.  There were 36 explicit invalid results and six
support-valid accuracy misses; all six were marginal, at 1.01--1.16 times the
budget.  No broad-pilot row selected polar.  The separately validated
`s=0.95, q=0.01, rho=0.005, A≈95` elongated-arc case still selects polar and
passes its common accuracy budget.

Smooth field-gradient checks use a three-resolution native central-difference
sequence because the native grid error is non-monotonic at tiny differencing
steps.  The rotating eight-case field set passes after requiring sequence
agreement.  Gradients within a few tens of source radii of a caustic remain a
separate convergence problem and are not certified by this pilot.

## Consequence

The dispatcher is now fail-closed, but the JAX inverse-ray backend does not yet
cover the complete sampled caustic domain.  The next calibration increment
must add Cartesian resolution/capacity buckets and an explicit coarse/fine
value-and-gradient convergence route.  Polar should remain a specialized
elongated-arc backend rather than a general overflow fallback.
