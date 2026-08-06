# Speed report — lcbinint against VBMicrolensing, microlux, and the JAX backend

August 2026 recalibration campaign. This file answers one question and its
corollaries:

> **Under what conditions is lcbinint faster than VBMicrolensing, and what does
> limb darkening do to the answer?**

The short version, stated once and supported below:

**On a uniform source VBM is faster essentially always — by a median factor of
17 to 31, and lcbinint wins under 1% of blocks at every tolerance. Limb
darkening reverses this, and it reverses it through a single mechanism: VBM's
cost multiplies by 3.2× at 1e-3 and 11.7× at 1e-4 when the profile goes from
uniform to linear, while lcbinint's cost multiplies by 1.00×. The crossing
therefore happens wherever the limb-darkened contour cost is already high — a
limb-darkened source at 1e-4 accuracy overlapping the caustic (`d/rho < 0.95`),
where lcbinint wins 69.3% of blocks by a median factor of 1.8, rising to 74.9%
and 2.1× once the magnification exceeds 3.**

Everything else in this file is the evidence, the boundaries of that claim, and
the same measurement against microlux, the JAX backend, and the triple lens.

Companion files: **`REPORT_master.md`** (the consolidated report — start there;
it supersedes §8's corpus table and adds a claims-and-qualifications section),
`README.md` (resolution rules, routing rules, corpus construction, reproduction
commands), and `REPORT_tangency_defects.md` / `REPORT_tangency_fixes.md` (two
correctness defects found while checking the references this report rests on,
and their fixes — read those before quoting any single-block number here).

---

## 1. How cost is measured

Three conventions, identical to the ones `README.md` states for Stage 2, and all
three matter for reading the tables.

**Timing is per block, not per epoch call.** A block is 24 consecutive epochs
spanning 0.4 source radii, timed three times, median reported. Single-epoch
timings on this problem are dominated by cache state, and a light curve is a
block, not an isolated call.

**Cost is read at *delivered* accuracy, never at a requested tolerance.** Every
engine is swept over its own accuracy knob — nine bin counts for the lcbinint
grids, three `reltol` values for `lcbinint_auto`, five `RelTol` values for VBM —
and for each block the reported cost is the cheapest setting whose *measured*
worst-case error over the block actually met the target. Comparing at nominal
tolerance would compare two engines' calibration habits, not their speed.

**Blocks whose converged reference is coarser than the accuracy under discussion
are excluded and counted.** This is why the 1e-4 rows judge ~1320 of 1440
blocks. The exclusion is applied identically to every engine.

One consequence must be stated because it works against VBM: the
`lcbinint_cartesian` / `lcbinint_polar` columns are an **oracle lower bound**.
They pick the cheapest of nine bin counts *after* seeing which one met the
target, which no caller can do at runtime. `lcbinint_auto` is the honest
pipeline number and is reported separately in §6. VBM's column is an oracle in
exactly the same way — cheapest of five `RelTol` settings after the fact — so
the two oracles are comparable to each other, and neither is a runtime number.

Corpus: 160 binary cases, seed 20260803, `s ∈ [0.6, 3.4]`, `q ∈ [1e-5, 0.3]`,
`rho ∈ [3e-5, 0.5]`, 8 caustic distances each (`d/rho` from 0.8 to 30 plus
far-field), every case under both a uniform and a linear (`c = 0.5`) profile.
2880 blocks, 69,120 timed epochs, 72,919 engine-setting measurements.

Ratios below are always **VBM / lcbinint**, so **a value above 1 means lcbinint
is faster**. "win" is the share of blocks with ratio > 1.

One difference from `README.md` Stage 2, which reports the same corpus: that
table reads the **Cartesian grid alone**, this one reads the **cheaper of the two
grids per block**. The numbers therefore differ slightly and in lcbinint's
favour — 69.3% against 63.5% in the headline `linear / 1e-4 / d/rho < 0.95` cell.
Both are correct for what they measure; this file uses cheaper-of-two throughout
because the grid switch (`README.md` Stage 3) is a rule the runtime applies.

---

## 2. The headline table

Cheaper of lcbinint's two grids against the cheapest qualifying VBM setting, at
matched achieved error:

| profile | accuracy | blocks | lcbinint win rate | median VBM/ours | p10 | p90 | VBM missed | ours missed |
|---|---|---|---|---|---|---|---|---|
| uniform | 1e-2 | 1351 | 0.0% | 0.048 | 0.006 | 0.311 | 3 | 0 |
| uniform | 1e-3 | 1347 | 0.0% | 0.058 | 0.007 | 0.312 | 3 | 0 |
| uniform | 1e-4 | 1317 | 0.3% | 0.032 | 0.007 | 0.317 | 24 | 4 |
| linear | 1e-2 | 1348 | 1.3% | 0.097 | 0.009 | 0.375 | 3 | 0 |
| linear | 1e-3 | 1344 | **12.5%** | 0.152 | 0.022 | **1.216** | 3 | 0 |
| linear | 1e-4 | 1320 | **23.8%** | 0.350 | 0.033 | **2.703** | 20 | 2 |

Read the p90 column. On uniform sources it never reaches 0.32 — there is no tail
in which lcbinint competes. On limb-darkened sources at 1e-3 it crosses 1.0, and
at 1e-4 it reaches 2.7. The distribution does not shift uniformly; it grows a
tail, and the rest of this section is about which blocks are in it.

"missed" counts blocks where the engine reached the accuracy at no setting it
was given. At 1e-4 VBM misses 20–24 blocks against lcbinint's 2–4. **The engine
that is faster on the median is the engine that more often fails to deliver the
requested accuracy** — a pattern that repeats independently against microlux in
§7.

## 3. The mechanism: what limb darkening costs each engine

Same block, same geometry, same accuracy, uniform versus linear profile:

| accuracy | engine | blocks paired | median linear/uniform cost | p90 |
|---|---|---|---|---|
| 1e-2 | VBM | 1348 | 1.02 | 6.80 |
| 1e-2 | lcbinint grid | 1351 | 1.01 | 1.07 |
| 1e-2 | `lcbinint_auto` | 1351 | 1.00 | 1.20 |
| 1e-3 | VBM | 1344 | **3.15** | 16.87 |
| 1e-3 | lcbinint grid | 1347 | **1.00** | 1.08 |
| 1e-3 | `lcbinint_auto` | 1347 | 1.00 | 1.20 |
| 1e-4 | VBM | 1317 | **11.66** | 52.49 |
| 1e-4 | lcbinint grid | 1338 | **1.00** | 1.19 |
| 1e-4 | `lcbinint_auto` | 1338 | 1.00 | 1.37 |

This is the whole result in one table, and it is a structural difference rather
than a tuning difference. A contour method integrates the limb-darkened profile
by stacking annuli, so its work is multiplied by the number of annuli the
tolerance demands — and that number grows as the tolerance tightens, which is
why the multiplier is 1.02 at 1e-2 and 11.66 at 1e-4. An inverse-ray grid
already visits the whole disk; applying a radial weight to rays it has shot
anyway is a multiply per ray. lcbinint's cost is **flat in the profile to within
1%** at every accuracy, with a p90 of 1.19 at the worst.

Everything in §4 follows from this table. lcbinint does not get faster on
limb-darkened sources; VBM gets slower, by a factor that the tolerance controls.

## 4. Where lcbinint wins — the conditions

### By distance to the caustic

| d/rho | accuracy | profile | blocks | win rate | median VBM/ours | p90 |
|---|---|---|---|---|---|---|
| < 0.95 (source over the caustic) | 1e-2 | linear | 301 | 5.0% | 0.266 | 0.729 |
| < 0.95 | 1e-3 | linear | 300 | **46.3%** | 0.953 | 2.346 |
| < 0.95 | 1e-4 | linear | 300 | **69.3%** | **1.815** | 6.129 |
| 0.95 – 1.5 | 1e-4 | linear | 319 | 17.2% | 0.321 | 2.025 |
| 1.5 – 4 | 1e-4 | linear | 260 | 16.9% | 0.357 | 1.515 |
| > 4 (far field) | 1e-4 | linear | 287 | 1.7% | 0.095 | 0.589 |
| < 0.95 | any | uniform | 302 | 0.0% | 0.041–0.059 | ≤ 0.130 |

The gradient is monotone and steep: the winning regime is where the source disk
**overlaps** the caustic, and it disappears by `d/rho > 1.5`. Far from the
caustic VBM's contour is a handful of image-boundary points and nothing lcbinint
does can compete.

### By magnification (linear profile)

| A | 1e-2 | 1e-3 | 1e-4 |
|---|---|---|---|
| < 1.2 | 0.0% / 0.112 | 0.8% / 0.115 | 3.7% / 0.300 |
| 1.2 – 3 | 0.7% / 0.099 | 8.0% / 0.104 | 21.5% / 0.328 |
| 3 – 10 | 2.4% / 0.088 | 21.3% / 0.156 | **34.4%** / 0.515 |
| 10 – 100 | 2.0% / 0.075 | **27.0%** / 0.255 | **40.4%** / 0.647 |
| > 100 | 3.1% / 0.068 | 12.6% / 0.179 | **33.3%** / 0.493 |

(cells are `win rate / median VBM-over-ours`; the uniform equivalents are
0.0–3.7% and 0.011–0.300 and are omitted.)

Magnification helps up to `A ≈ 100` and then turns over. That turnover is the
`A_point > 200` grid-switch boundary from `README.md` Stage 3 showing up on the
speed axis: above it the Cartesian grid is the wrong grid, and although the
oracle here is allowed to pick polar instead, the polar grid at those
magnifications is itself expensive.

### By rho and by mass ratio (linear profile, 1e-4)

| rho | win rate | median | | q | win rate | median |
|---|---|---|---|---|---|---|
| < 1e-3 | 21.9% | 0.304 | | < 1e-4 | 20.7% | 0.352 |
| 1e-3 – 1e-2 | 23.8% | 0.327 | | 1e-4 – 1e-2 | 21.9% | 0.320 |
| 1e-2 – 1e-1 | **31.2%** | **0.567** | | > 1e-2 | **26.9%** | 0.390 |
| > 1e-1 | 18.1% | 0.369 | | | | |

Both axes are nearly flat — a factor of 1.7 in win rate across four decades of
rho, against a factor of 40 across the `d/rho` axis. **rho is not the variable
that decides this**, which is worth saying plainly because the natural
expectation is that a big source favours a grid. It does not, at fixed `d/rho`:
what favours the grid is the source *covering the caustic*, and `d/rho` measures
that directly while rho does not.

### The joint condition

| condition (linear profile) | blocks | win rate | median VBM/ours | p90 |
|---|---|---|---|---|
| 1e-3, `d/rho < 0.95` | 300 | 46.3% | 0.953 | 2.346 |
| 1e-3, `d/rho < 0.95`, `A > 3` | 211 | **54.5%** | **1.099** | 2.519 |
| 1e-4, `d/rho < 0.95` | 300 | 69.3% | 1.815 | 6.129 |
| 1e-4, `d/rho < 0.95`, `A > 3` | 211 | **74.9%** | **2.120** | 6.397 |
| 1e-4, `d/rho < 0.95`, `A > 10` | 130 | 70.8% | 2.055 | 7.101 |
| 1e-4, `d/rho < 1.5`, `A > 10` | 247 | 49.4% | 0.946 | 5.086 |
| 1e-4, `A > 10` (no distance cut) | 395 | 37.7% | 0.576 | 3.700 |

**The best-supported statement of the winning regime is:
`limb-darkened ∧ tolerance ≤ 1e-4 ∧ d/rho < 0.95 ∧ A > 3` → lcbinint wins three
blocks in four, by a median factor of 2.1 and a p90 of 6.4.** Relaxing the
distance cut to 1.5 drops it to a coin flip; dropping the distance cut entirely
drops it to 38%.

The size of that regime is 211 of the 1346 limb-darkened blocks at 1e-4, i.e.
about 16% of the corpus — but the corpus samples `d/rho` uniformly in a
log-spaced ladder out to 30, which is not how a real event distributes its
epochs. In an event with a caustic crossing, the epochs that matter for the
model are exactly the ones inside the winning regime.

### The condition stated without `d/rho`

`d/rho` is a corpus label. It is computed from a caustic-distance query the
runtime has already made when it picks a route, so the rule is implementable —
but only inside lcbinint, and it is not a quantity a user chooses. Restated in
user-facing terms:

> lcbinint is the faster engine for **limb-darkened sources during a caustic
> crossing at tolerances of 1e-4 or tighter**. Outside a crossing, or on a
> uniform source, or at 1e-2, VBM is faster by one to two orders of magnitude.

## 5. Reliability runs the other way from speed

At 1e-4, VBM fails to reach the target on 20–24 blocks where lcbinint succeeds;
the reverse happens on 2–4. Within the `rho ∈ [1e-2, 1e-1]` band the counts are
11 and 8 against 0. This is not a large fraction of 1320 blocks, but it is
consistently signed, and it lands where the speed advantage is largest — which
is the same place §7 finds microlux failing 35% of the time.

## 6. The routed pipeline, and a comparison that must not be made

`lcbinint_auto` — the shipping pipeline, choosing point source / hexadecapole /
polar / Cartesian / source-plane quadrature by itself — against the same VBM
oracle:

| accuracy | profile | blocks | win rate | median VBM/auto | p90 |
|---|---|---|---|---|---|
| 1e-2 | uniform | 1351 | 16.7% | 0.021 | 14.15 |
| 1e-2 | linear | 1348 | 17.0% | 0.047 | 14.21 |
| 1e-3 | uniform | 1347 | 16.8% | 0.026 | 14.16 |
| 1e-3 | linear | 1344 | 20.5% | 0.121 | 14.21 |
| 1e-4 | uniform | 1316 | 14.9% | 0.012 | 14.05 |
| 1e-4 | linear | 1319 | **26.9%** | 0.257 | 14.14 |

The p90 pinned at ~14.1 in every single cell is the tell. **Those are the blocks
the pipeline answered with the point-source formula**, where the comparison is
between a closed-form expression and a contour integral and says nothing about
either engine's finite-source algorithm. The flat median of 14× is the cost of
VBM's own point-source-safe path, not a speed result.

So this table should be read as: *the pipeline's routing is worth roughly a
factor of 14 on the blocks it can route away, and on the blocks it cannot, §2
applies.* It must not be quoted as "lcbinint wins 27% of blocks" alongside the
grid numbers — putting a routing decision and a quadrature on the same axis is a
category error. `README.md` Stage 3 makes the same point about the
`A ∈ [1, 1.13]` bucket.

The one honest signal in the table is the 1e-4 column, where the linear win rate
(26.9%) exceeds the uniform one (14.9%) by more than the routing can explain —
that is §3's mechanism surviving into the routed pipeline.

## 7. microlux

An independent contour implementation, from `ext_discovery/`. Calibrated for
the cross-run clock difference (measured at 1.312 on 5504 paired native
measurements — see `README.md` Stage 4; uncalibrated numbers would flatter
microlux by 31%). Ratios are microlux/ours, so **below 1 means microlux is
faster**:

| profile | accuracy | judgeable | microlux reached | missed | lcbinint win rate | median microlux/ours | p10 – p90 |
|---|---|---|---|---|---|---|---|
| uniform | 1e-2 | 1350 | 1334 | 16 (1.2%) | 63.1% | 0.749 | 0.105 – 1.802 |
| uniform | 1e-3 | 1349 | 1332 | 17 (1.3%) | 64.5% | 0.706 | 0.102 – 1.662 |
| uniform | 1e-4 | 797 | 769 | 28 (3.5%) | 79.2% | 0.456 | 0.084 – 1.178 |
| linear | 1e-2 | 1347 | 1319 | 28 (2.1%) | 28.4% | **3.076** | 0.573 – 12.856 |
| linear | 1e-3 | 1346 | 1247 | 99 (7.4%) | 28.9% | **3.108** | 0.569 – 12.696 |
| linear | 1e-4 | 909 | 589 | **320 (35.2%)** | 37.8% | **1.159** | 0.565 – 5.807 |

**The limb-darkening asymmetry reproduces through a second, independent contour
code, and more strongly than against VBM.** That is the strongest available
evidence that §3 is a property of the method class and not of one
implementation.

The 1e-4 limb-darkened row understates lcbinint rather than overstating it.
microlux fails to reach 1e-4 on 320 of 909 judgeable blocks — 114 exhausted the
adaptive sampler's budget (`No enough space to insert new samplings`), 89 were
censored for already exceeding 0.25 s/epoch at a looser setting, the rest simply
never met the target — and the 1.159 median is computed over the 589 where it
succeeded. The blocks it drops are the hard ones.

Two separate effects shrink that row and must not be conflated: judgeable falls
from 1346 to 909 because of the reference floor (a property of the reference,
applied identically to every engine), and 909 falls to 589 because of microlux.

## 8. The JAX backend

> **The corpus table immediately below is superseded.** The missing `jax.jit`
> diagnosed in §8.2 has since been applied — together with four further changes,
> of which a tile capacity that silently capped the reachable magnification near
> A ~ 47 is the one that matters for reliability — and the full corpus
> re-measured (`ext_capfix/`). The backend now costs **4.8–6.2×** the oracle grid
> and wins **~20%** of blocks, against the 43–62× and 0.0–2.2% recorded here.
> Against the honest `lcbinint_auto` denominator this section argues for below,
> it is **5.5–5.8×**, against the 23–32× recorded there. The fail-closed rate
> fell from 9.3% to 2.6% at reltol 1e-2 and the corpus counts rose accordingly;
> a later fix — the polar route had no way back to the Cartesian ladder — took
> it to **0.50%** (`REPORT_master.md` §5.5.1). Removing those NaNs exposed a
> fail-*open* in the polar route that the certificate cannot detect, §5.5.2;
> it is unresolved and it does not affect any timing in this section.
> `REPORT_master.md` §5.5 carries the current numbers and the decomposition.
> §8.1–8.5 below remain valid: they are the diagnosis that produced the fix.
> Quote 5.5–5.8×, not 4.8–6.2× — the caveat in the next subsection did not stop
> applying just because the numerator got smaller.

**Read §8.1–8.4 before quoting anything in this section.** The ratios below are
a property of the shipped wrapper, not of the JAX or FFI integrator: the same
compiled C++ kernel runs on both sides, it costs 3.1–3.9× native, and the rest
is a ~590 ms per-call constant caused by a missing `jax.jit`.

Warm timings; first-call compilation is measured separately and excluded.

| profile | accuracy | blocks | JAX win rate | median JAX/native | p10 – p90 | median first call |
|---|---|---|---|---|---|---|
| uniform | 1e-2 | 1224 | 0.1% | 61.9× | 9.7 – 237 | 0.96 s |
| uniform | 1e-3 | 1220 | 0.2% | 55.5× | 9.0 – 235 | 0.96 s |
| uniform | 1e-4 | 690 | 2.2% | 44.9× | 8.2 – 249 | 1.21 s |
| linear | 1e-2 | 1220 | 0.0% | 59.2× | 9.8 – 240 | 0.88 s |
| linear | 1e-3 | 1211 | 0.0% | 57.0× | 9.9 – 240 | 0.92 s |
| linear | 1e-4 | 790 | 1.6% | 43.1× | 8.9 – 251 | 1.14 s |

### The denominator matters, and the table above uses the wrong one

`ext_analysis.py:196–200` divides by `min(cheapest lcbinint_cartesian, cheapest
lcbinint_polar)` — the **oracle over 18 forced-grid settings** (9 bin counts ×
2 grids), post-selected per block. But the thing being timed on the JAX side,
`binary_magnification_native_pipeline_trajectory`, is a **routed pipeline**: it
runs its own point-source, hexadecapole and band decisions. Its structural
counterpart on the native side is `lcbinint_auto`, not a forced grid.

Recomputed against `lcbinint_auto` on exactly the same rows, same accuracy
matching, same per-row cross-run calibration:

| profile | accuracy | blocks | median JAX / oracle grid | median JAX / `lcbinint_auto` | p10 (auto) | JAX win rate (auto) |
|---|---|---|---|---|---|---|
| uniform | 1e-2 | 1224 | 61.9× | **30.9×** | 7.8× | 0.7% |
| uniform | 1e-3 | 1220 | 55.5× | **30.9×** | 7.8× | 0.8% |
| uniform | 1e-4 | 693 | 44.9× | **23.4×** | 2.3× | 3.9% |
| linear | 1e-2 | 1220 | 59.2× | **30.5×** | 7.8× | 1.2% |
| linear | 1e-3 | 1211 | 57.0× | **32.1×** | 7.8× | 1.2% |
| linear | 1e-4 | 791 | 43.1× | **24.3×** | 2.5× | 4.7% |

**The oracle denominator inflates the ratio by about 1.8×. The honest
pipeline-to-pipeline number is 23–32×, not 43–62×.** It does not change the
sign — the 10th percentile is still 2.3–7.8× and JAX wins under 5% of blocks —
but 43–62× should not be quoted, and the rest of this section uses 23–32×.

(The `lcbinint_auto` p90 is ~10⁴ and is not reported above for the same reason
§6 gives: those are blocks the native pipeline answered with the point-source
formula. The median is the statistic that survives that.)

Two further caveats, neither of which changes the sign either:

*The cost model double-counts one route.* In
`binary_magnification_native_pipeline_trajectory` the fused trajectory is
evaluated unconditionally and the source-plane chord quadrature then *replaces*
the result on epochs the band test selects, so those epochs pay both the
Cartesian grid and the quadrature. The route appears in 32.9% of blocks. That is
worth a factor of two on a third of the corpus and cannot account for fifty.

*The comparison is CPU-only and single-trajectory* — the regime the JAX backend
is least suited to. Its reasons to exist are differentiability and batching
across many trajectories on an accelerator, neither of which appears on this
axis. What the table establishes is only that the JAX path is not a drop-in
replacement for the native one when a fit needs magnifications alone.

*Compilation is excluded, and that favours JAX.* This one is worth stating
separately because it is easy to file with the others and it cuts the other way:
the per-epoch timings are warm, so none of the 23–32× is compilation. Adding it
back makes JAX slower, not faster.

The first-call column is the second finding and the more practical one: just
under a second of compilation per distinct `(s, q, rho)`. Negligible for a light
curve of thousands of epochs, dominant for a sampler that moves the geometry
every step. Combined with the **+546 address-space mappings per compilation**
documented in `README.md` Stage 4, the binding limit is not speed but how many
distinct geometries one process may ever see before XLA aborts it against
`vm.max_map_count`.

### 8.1 The kernel-level measurement, and why the 23–32× is not about JAX

The number above is a property of the pipeline wrapper, not of the JAX or FFI
inverse-ray kernel. Three measurements separate them. All are single-threaded
(`OMP_NUM_THREADS=1`, one pinned core), which is the campaign's own condition —
see §11.

**First: what is actually being benchmarked.** `Options(jax=True)` with
`nbin="auto"` reaches `binary_magnification_native_pipeline_trajectory`, which
calls `binary_inverse_ray_cartesian_batch_ffi` — the *same compiled C++
Cartesian kernel the native path uses*. Nothing in the 23–32× is XLA-versus-C++;
both sides run identical machine code for the quadrature. Confirmed directly:
on three geometries × two tolerances both backends selected the *same method on
every epoch* and agreed to 2.6e-14.

**Second: the kernel ratio.** One FFI call over a 24-epoch block against native
forced Cartesian on identical source positions (`coordinates="center_of_mass"`,
`s=1.2, q=0.1, rho=0.02`), same nbin, routing disabled on both sides:

| nbin | native ms/ep | FFI ms/ep | FFI / native | 14-bucket ladder ms/ep | ladder / one call |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.230 | 0.726 | 3.15× | 1.648 | 2.27× |
| 32 | 0.320 | 1.146 | 3.59× | 3.447 | 3.01× |
| 64 | 0.602 | 2.191 | 3.64× | 6.676 | 3.05× |
| 128 | 1.599 | 5.020 | 3.14× | 15.429 | 3.07× |

Linear limb darkening gives 3.03–3.92×. **The FFI kernel costs 3.1–3.9× the
native kernel.** That is exactly the figure
`docs/jax-cpu-inverse-ray-mvp.md:284` recorded at design time — "native
`lcbinint` remains 3–4x faster than the JAX inverse-ray forward path". It never
regressed. The design-time nuance is intact; it was simply never the quantity
the 43–62× headline measured.

**Third: where the rest of the factor lives.** The gap is a per-*call* constant,
not a per-epoch cost. Same geometry, `nbin="auto"`, warm:

| epochs | native ms/call | JAX ms/call | JAX ms/epoch | ratio |
|---:|---:|---:|---:|---:|
| 1 | 1.37 | 585 | 585 | 428× |
| 24 | 2.11 | 638 | 26.6 | 302× |
| 96 | 4.31 | 653 | 6.81 | 152× |
| 384 | 13.03 | 675 | 1.76 | 51.8× |
| 1536 | 47.80 | 787 | 0.512 | 16.5× |

The JAX call costs ~590 ms **before it looks at a single epoch**. The marginal
per-epoch cost is 0.13 ms against native's 0.030 ms — a ratio of 4.3×,
consistent with the kernel table. Everything above that is the constant divided
by the block length.

### 8.2 The constant is a missing `jax.jit`

The constant is not compilation (first call is 2.0–2.3 s and *every* warm call
pays the 590 ms), not the caustic scan (flat from `caustic_bins` 100 to 4000
while the native cost moves 0.18 → 3.59 ms), and not the bucket ladder
(truncating it from fourteen buckets to one via `max_source_bins` leaves it at
585 ms). The decisive line is that fixing `nbin` — which leaves the calibrated
pipeline for the plain fused trajectory — drops the same call from **590 ms to
0.68 ms**, faster than native's 1.38 ms.

`binary_magnification_native_pipeline_trajectory` carries no `jax.jit`, and
`jax_backend.py:799` calls it directly. Every call re-traces the whole program —
fourteen Cartesian FFI calls, two `lax.map` bodies, a fourteen-way `lax.switch`
— and dispatches it primitive by primitive. Tracing cost scales with program
size, not with data, which is precisely the observed shape: flat in epochs, flat
in buckets, flat in `caustic_bins`.

Wrapping the existing function in `jax.jit`, changing nothing else:

| epochs | eager ms | jitted ms | speed-up | compile s | value change |
|---:|---:|---:|---:|---:|---:|
| 1 | 590.12 | 1.530 | 386× | 1.12 | none (0.0) |
| 24 | 644.75 | 3.450 | 187× | 1.37 | none (0.0) |
| 96 | 655.25 | 9.429 | 69× | 1.49 | none (0.0) |
| 384 | 678.67 | 32.919 | 21× | 1.39 | none (0.0) |

Bit-identical results. Against native at the same epoch counts the jitted
pipeline is 1.12× / 1.63× / 2.19× / 2.53× — converging on the kernel ratio.

### 8.3 Is the fourteen-bucket expansion needed?

Partly, and it is not the problem. The masked-out calls are free: eleven of the
fourteen never run, and truncating the ladder does not move the cost. When the
grid *is* the selected route the ladder costs 3.0× one call, because it arms the
selected bucket and **both** neighbours. Only the lower neighbour is normally
required — the upper is a fallback for when the lower is unsupported or switches
method — so roughly a third of that 3.0× is recoverable, worth ~1.5×. That is a
small optimisation next to the 21–386× above, and it should not be attempted
first.

### 8.4 What this section establishes

> The FFI Cartesian kernel costs 3.1–3.9× the native kernel per epoch, matching
> the design-time record. The public `Options(jax=True, nbin="auto")` pipeline
> costs an additional ~590 ms per call because it is not jitted; over the
> campaign's 24-epoch blocks that constant is what produces the 23–32×. Jitting
> it recovers 21–386× with bit-identical results.

Consequences for how the campaign's JAX numbers may be quoted:

* **23–32× is a measurement of the current shipped wrapper, not of the method.**
  It is honest as a statement about what a user pays today with
  `Options(jax=True)`. It must not be presented as JAX or XLA being slow at
  inverse-ray integration.
* **The block length is a first-order choice.** At 24 epochs the constant
  dominates; at 1536 the same code reports 16.5×. Any JAX ratio in this campaign
  is only meaningful with the block length stated.
* **The fix is one decorator**, and it should be measured again after landing —
  the whole JAX axis of the campaign is worth re-running at that point.

Still not measured: API-level *cold* cost with compilation included at a
realistic geometry-change rate, which is what a sampler actually pays, and
anything on an accelerator.

### 8.5 Fairness of the design-era comparison

`docs/jax-cpu-inverse-ray-mvp.md:669–677` reports the opposite sign — 9.48 ms
for a 64-epoch JAX trajectory against 135.20 ms native. Two harness choices in
`tests/diagnostics/jax_ir/benchmark_trajectory.py:309–332` account for it, both
favouring JAX:

* The native side is a Python loop over the **scalar** `binary_ray_shooting`,
  which rebuilds the lens geometry on every epoch. Measured here, that costs
  1.9–6.5× against the cached `LightCurve` block path — 1.497 ms/ep versus
  0.230 ms/ep at nbin=16.
* `OMP_NUM_THREADS` is unset, so the FFI batch takes up to 32 threads while
  native takes one — `batch_thread_count` in `src/lcbinint/lc/light_curve.cpp:22`
  returns 1 unconditionally, so *no* native path threads across epochs.

The native side there was also a forced `nbin=64` Cartesian grid on all 64
epochs while the JAX side routed 45 of them to hexadecapole. The document
already flags a related problem in the other direction — "the first-passing
native setting remains an optimistic latency comparison" (line 297). The
recal2026 campaign is clean on both threading (`sweep_ext.py:304,455` pins
`OMP_NUM_THREADS=1` and one worker per core) and options; its own unfairnesses
are the oracle denominator corrected above and the 24-epoch block.

## 9. Triple lens

Reduced scope by design: 32 cases, 48 epochs per block spanning 0.6 source
radii, both profiles, three tolerances, timed three times and the median taken.
`triple_compare/`, harness `tests/diagnostics/recal2026/triple_compare.py`.

**VBM exposes three multi-lens methods and all three are measured.** This
matters more than it sounds: `Nopoly` is what you get when `SetMethod` is never
called, and it is the slowest of the three by 2.5–3×. An earlier internal note
claiming lcbinint was ~2× faster than VBM on triple lenses was measured against
that default and is **wrong**; it is corrected here.

| engine | tolerance | profile | ms/epoch | err median | err p90 | err max | blocks |
|---|---|---|---|---|---|---|---|
| `lcbinint_auto` | 1e-2 | uniform | 6.161 | 1.13e-05 | 4.76e-05 | 2.70e-04 | 32 |
| `vbm_singlepoly` | 1e-2 | uniform | **0.086** | 3.77e-04 | 1.35e-03 | 1.74e-03 | 32 |
| `vbm_multipoly` | 1e-2 | uniform | 0.124 | 3.77e-04 | 1.37e-03 | 1.74e-03 | 30 |
| `vbm_nopoly` | 1e-2 | uniform | 0.256 | 3.29e-04 | 1.37e-03 | 1.74e-03 | 19 |
| `lcbinint_auto` | 1e-2 | linear | 7.284 | 6.92e-06 | 2.92e-05 | 1.80e-04 | 32 |
| `vbm_singlepoly` | 1e-2 | linear | **0.101** | 1.25e-03 | 4.16e-03 | 6.08e-03 | 32 |
| `lcbinint_auto` | 1e-3 | uniform | 6.163 | 1.11e-05 | 4.11e-05 | 8.16e-05 | 32 |
| `vbm_singlepoly` | 1e-3 | uniform | **0.163** | 2.94e-05 | 8.45e-05 | 1.17e-04 | 32 |
| `lcbinint_auto` | 1e-3 | linear | 7.284 | 6.87e-06 | 2.54e-05 | 4.89e-05 | 32 |
| `vbm_singlepoly` | 1e-3 | linear | **0.311** | 3.43e-04 | 5.16e-04 | 5.43e-04 | 32 |
| `lcbinint_auto` | 1e-4 | uniform | 12.072 | 5.09e-06 | 8.34e-06 | 8.91e-06 | 32 |
| `vbm_singlepoly` | 1e-4 | uniform | **0.305** | 6.09e-06 | 9.38e-06 | 1.25e-05 | 32 |
| `lcbinint_auto` | 1e-4 | linear | 21.368 | 4.59e-06 | 7.10e-06 | 8.62e-06 | 32 |
| `vbm_singlepoly` | 1e-4 | linear | **0.766** | 4.47e-05 | 5.34e-05 | 5.97e-05 | 32 |

Cost ratio, `lcbinint_auto` against the fastest VBM method on the same block:

| tolerance | profile | median | p10 | max |
|---|---|---|---|---|
| 1e-2 | uniform | 64.9× | 2.5× | 629.9× |
| 1e-2 | linear | 70.5× | 2.5× | 513.9× |
| 1e-3 | uniform | 38.1× | 1.4× | 237.4× |
| 1e-3 | linear | 29.1× | 1.3× | 177.3× |
| 1e-4 | uniform | 43.0× | 0.8× | 304.9× |
| 1e-4 | linear | **20.6×** | **0.8×** | 131.1× |

Four things this says.

**VBM is decisively faster on triple lenses** — 20–70× at matched knob, and
never worse than 0.8× even at the 10th percentile. The binary crossover does not
reproduce here at this corpus size.

**The limb-darkening mechanism reproduces on the triple lens anyway.** VBM's
accuracy degrades ~10× with a linear profile at fixed tolerance (2.94e-5 →
3.43e-4 at 1e-3; 6.09e-6 → 4.47e-5 at 1e-4) and its cost rises 1.7–2.6×;
lcbinint's error *improves* slightly (1.11e-5 → 6.87e-6) at +18–77% time. The
1e-4 linear cell has the smallest gap (20.6×) for exactly this reason. The
effect is visible; the corpus is simply not close enough for it to change the
verdict.

**lcbinint's triple `reltol` knob is inert between 1e-2 and 1e-3.** Bit-identical
error on 58 of 60 case-profiles, time within 5% on 57 of 60, identical route
distributions at all three tolerances. lcbinint over-delivers by roughly 1000×
on a 1e-2 request. This is a concrete defect in the triple accuracy control and
is the single clearest improvement available: honouring a loose request on the
triple path would move the 1e-2 rows by an order of magnitude at no accuracy
cost.

**VBM's triple methods are not equally robust.** Over the 32-case corpus:
`Singlepoly` 0 failures; `Multipoly` 2 real failures out of 32 (one SIGSEGV, one
SIGABRT, the latter reproducible only at `c = 0.5`, `RelTol = 1e-9`); `Nopoly` —
*the default when `SetMethod` is never called* — 13 of 32 case-profile-tolerance
sets lost to SIGSEGV, which is why its rows above judge 19 blocks rather than 32.
The reference contours were built with `Multipoly` at `RelTol = 1e-9`, and the
limb-darkened case is expensive there: 301.99 s for one block at `c = 0.5`
against 3.6 s for the same block uniform, and 0.18 s at `RelTol = 1e-4`.

## 10. Where lcbinint's own time goes

Measured with `probe_diagnostics` (counters and an ablation policy, both off
unless `LCBININT_PROBE_STATS` / `LCBININT_PROBE_POLICY` are set), over a
1040-row caustic-grazing corpus. Committed as `891ca5e`.

The heuristic probe rings that seed the image search cost **11.4× the certified
probes' root solves** (median 480 against 30) and about **85% of seeding time**.
Seeding as a share of the whole call is 22.9% on average before the change —
but that share is a strong function of resolution: 39% at nbin 8, 15% at nbin
50, 1.9% at nbin 200. It is large only where the grid is cheap.

Of the five ring stages, **only the interior rings were dead**, and they are
deleted: ablating them reproduced every magnification bit for bit over 911 grid
rows × nbin {16, 50, 128}, because the radii they sample are always reached by
the boundary ring or the certified ladder first. Mean seeding share falls from
22.9% to 9.8%.

The other rings are **not** redundant, and the reason is a defect rather than a
feature — see `REPORT_tangency_defects.md` §4. Deleting all of them would have
been worth 1.13× at nbin 4 and 1.00× at nbin 200 in any case, and the timing
blocks are far denser in grid epochs than a real light curve is.

## 11. Threats to validity

* **The grid columns are oracles.** §1. `lcbinint_auto` (§6) is the runtime
  number, and it carries a routing effect that must be separated before it is
  compared to a quadrature.
* **The references are not above suspicion.** `REPORT_tangency_defects.md`
  documents two defects in the neighbourhood of `d/rho ≈ 1`, one of which
  implicates the reference construction itself. Aggregates over ~1300 blocks are
  not moved by ~140 disputed positions, but **no individual block in the
  `d/rho ∈ [0.85, 1.05]` band should be quoted from this report** without
  checking that file.
* **The 1e-4 corpus is ~8% smaller** than the looser ones (1317–1338 of 1440
  blocks) because of the reference floor.
* **Cross-run clock.** Stage 4 (microlux, JAX) could not run at Stage 2's
  concurrency. Both runs timed the same two native Cartesian buckets on the same
  geometries; the ratio is 1.312, flat across ten deciles of block cost, and is
  applied per row from that row's own control rather than as a scalar.
* **Single machine, single build.** 64 physical cores, one worker per core,
  workers pinned. Absolute milliseconds are not portable; the ratios are the
  result.
* **`d/rho` is a corpus label.** Rules stated in terms of it describe where an
  effect lives. §4 gives the user-facing restatement.

## 12. Reproduction

```bash
# the binary speed corpus (long)
taskset -c 40-63 python -m tests.diagnostics.recal2026.sweep_speed \
    --output tests/diagnostics/results/recal2026/speed_discovery --workers 24 --repeat 3

# microlux + JAX, 11 passes; do not run concurrently with sweep_speed
python -m tests.diagnostics.recal2026.sweep_ext \
    --blocks tests/diagnostics/results/recal2026/speed_discovery \
    --output tests/diagnostics/results/recal2026/ext_discovery \
    --workers 24 --cores 40-63 --repeat 3 --blocks-per-worker 2

# triple lens, all three VBM methods, per-method subprocess isolation
python -m tests.diagnostics.recal2026.triple_compare \
    --output tests/diagnostics/results/recal2026/triple_compare --timeout 1800

# probe cost, ablations, and timing (probe_sweep.sh drives all three in turn)
bash tests/diagnostics/recal2026/probe_sweep.sh
python -m tests.diagnostics.recal2026.probe_analysis \
    tests/diagnostics/results/recal2026/probe

# the JAX kernel/wrapper decomposition of section 8; minutes, one core
OMP_NUM_THREADS=1 taskset -c 57 python -m tests.diagnostics.recal2026.jax_kernel_audit \
    --repeat 3 --block-lengths 1,24,384 \
    --out tests/diagnostics/results/recal2026/jax_kernel_audit_uniform.json
OMP_NUM_THREADS=1 taskset -c 57 python -m tests.diagnostics.recal2026.jax_kernel_audit \
    --limb-c 0.4 --repeat 3 --block-lengths 1,24,384 \
    --out tests/diagnostics/results/recal2026/jax_kernel_audit_linear.json
```

`jax_kernel_audit.py` must stay single-threaded: the FFI batch is OpenMP-parallel
over epochs and no native path threads across the epochs of one trajectory, so a
threaded run measures the harness.

Analysis of the tables above:
`speed_analysis.py`, `grid_switch.py`, `route_audit.py`, `ext_analysis.py`.
