# Limb-Darkened Finite-Source Performance Sweep

Date: 2026-06-22

Purpose: measure raw fixed-bin inverse-ray performance for limb-darkened finite
sources against `VBBinaryLensing.BinaryLightCurve`, before tuning adaptive
`source_bins`.

Diagnostic script:

```bash
python tests/diagnostics/ld_source_bins_rho_sweep.py \
  --rhos 1e-4,3e-4,1e-3,3e-3,1e-2,3e-2 \
  --source-bins 35,50,70,100,140 \
  --times 61 \
  --random 12 \
  --seed 20260622 \
  --top 14
```

Settings:

- limb darkening: linear coefficient `c=0.5`
- VBBL tolerance: `Tol=1e-3`
- lcbinint: fixed `source_bins`, `adaptive_source_bins=0`, `vbbl_compatible=1`
- geometry set: 6 fixed cases plus 12 random geometries; `rho` is swept
  independently.

## Aggregate Result

By `source_bins`:

| bins | cases | lc/VBBL median | lc/VBBL geo | lc ms/pt med | VBBL ms/pt med | max rel med | worst max rel |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 35 | 108 | 1.921 | 1.868 | 0.4411 | 0.2470 | 2.119e-04 | 9.227e-01 |
| 50 | 108 | 2.127 | 2.027 | 0.5070 | 0.2470 | 1.156e-04 | 2.297e-03 |
| 70 | 108 | 2.366 | 2.269 | 0.6846 | 0.2470 | 8.004e-05 | 2.297e-03 |
| 100 | 108 | 2.748 | 2.735 | 0.8806 | 0.2470 | 6.518e-05 | 2.297e-03 |
| 140 | 108 | 3.795 | 3.448 | 1.3156 | 0.2470 | 6.313e-05 | 2.297e-03 |

By `rho`:

| rho | cases | lc/VBBL median | lc/VBBL geo | lc ms/pt med | VBBL ms/pt med | max rel med | worst max rel |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1e-4 | 90 | 2.619 | 2.977 | 0.1188 | 0.0438 | 7.354e-06 | 4.142e-05 |
| 3e-4 | 90 | 3.936 | 3.324 | 0.1539 | 0.0414 | 1.611e-05 | 1.132e-03 |
| 1e-3 | 90 | 2.491 | 3.120 | 0.4196 | 0.1428 | 5.781e-05 | 2.297e-03 |
| 3e-3 | 90 | 2.607 | 2.051 | 0.9354 | 0.3681 | 9.506e-05 | 9.227e-01 |
| 1e-2 | 90 | 2.474 | 1.867 | 2.2789 | 0.8547 | 1.851e-04 | 6.339e-01 |
| 3e-2 | 90 | 1.643 | 1.650 | 4.7107 | 2.9021 | 2.951e-04 | 3.229e-02 |

## Interpretation

- Globally, fixed-bin lcbinint LD is still slower than VBBL at the median:
  roughly `2x` at `source_bins=35-70`, and worse as bins increase.
- lcbinint can be much faster for high-magnification planetary/low-q cases
  where VBBL LD contour integration becomes expensive. Examples from the sweep:
  `random_007`, `random_011`, and `planetary_close` reach `lc/VBBL = 0.09-0.25`
  at usable bins.
- lcbinint is much slower when the source is effectively easy for VBBL:
  small `rho`, low magnification, or ordinary binary geometries. Some rows have
  `lc/VBBL > 80` only because VBBL returns in about `0.002 ms/pt`, while
  lcbinint still pays the fixed inverse-ray overhead.
- `source_bins=35` is too aggressive. It has catastrophic outliers
  (`max rel` up to `0.92`) even though the median error looks good.
- `source_bins=50` looks like the first broadly usable floor in this sweep:
  median max relative error `1.16e-4`, worst max relative error `2.3e-3`.
- Increasing beyond `50` gives diminishing accuracy returns for many cases but
  a direct speed penalty. This supports keeping the default floor near 50 and
  using adaptive refinement only when diagnostics indicate unresolved image
  structure.

## Tuning Direction

The next useful optimization is not simply lowering `source_bins` globally.
Instead:

1. Keep a conservative floor around `source_bins=50` for caustic/IR mode.
2. Avoid IR entirely when hexadecapole is reliable, especially for small `rho`
   and ordinary low-magnification points.
3. For LD IR, refine only on image-resolution diagnostics, because most of the
   median benefit from `70-140` bins is small compared to its cost.
4. Treat `source_bins=35` as an optional fast preview setting, not a default,
   unless a stronger failure detector catches the catastrophic outliers.

## Adaptive Mode Investigation (2026-06-23)

Timing breakdown for `planetary_large_source_ld` (s=1, q=1e-3, rho=0.01, LD=0.5)
with 241 time points over [-0.8, 0.8], source_bins=50→200, reltol=1e-4:

| mode | ms/pt | vs VBBL |
|---|---|---|
| VBBL Tol=1e-3 | 2.01 | 1× |
| fixed@50 | 1.10 | 0.55× (faster!) |
| fixed@200 | ~5.1 | 2.5× slower |
| adaptive 50→200 | 5.74 | 2.85× slower |

Key finding: adaptive overhead comes entirely from 27 caustic-crossing points,
each doing 50+100+200 bin computations (seeding is shared). Cost model per IR point:

- Seeding (Phase 1: 1400-sample caustic scan): ~3ms (75% of 50-bin cost)
- Integration at 50 bins: ~1ms (25%)
- Integration at 100 bins: ~4ms (4×)
- Integration at 200 bins: ~16ms (16×, quadratic scaling)

For the 27 refined points: each costs ~21ms extra (100+200 bin integration).
The 214 non-refined points run at fixed@50 speed.

**Phase 1 skip optimization analysis:** skipping the 1400-sample caustic scan for
clearly-far points (caustic_distance > 5×rho) would save ~4% of total time. Not
worth the implementation complexity given the seeding is not the scaling bottleneck.

**Path to beating VBBL for large-rho LD:** requires incremental flood-fill (reuse
lower-resolution result for finer grid) or SIMD vectorization of the inner scan loop.
A 3× speedup in integration would make adaptive competitive with VBBL LD for rho=1e-2.

**Current wins against VBBL (confirmed in named-case sweep):**
- `wide_low_q` (s=1.5, q=1e-3, rho=1e-4): adaptive 0.027ms vs VBBL 0.05ms → 1.85× faster
- High-mag planetary/low-q with LD: lc/VBBL reaches 0.09-0.25 in specific cases

**Convergence criterion (commit 7337a60):** size>=3 + min-history guard eliminated
planetary_large_source accepted_bad=1. All 27 refined points still go to level 2
because max(diag, hist) exceeds target at 100 bins — the size==2 vs size>=3 distinction
has no practical effect when both conditions fail to accept at 100 bins.

**Boundary weight:** bw=0.03 for rho<2e-2 is the stable value. Reducing to 0.005
would save ~12 unconverged across named cases but introduces 2+ new accepted_bad in
random sweeps. The high unconverged counts for close_binary (29/241), resonant_low_q
(26/241), wide_equal_mass (36/241) correctly reflect genuine 200-bin accuracy limits
for complex caustic topologies — not false positives.
