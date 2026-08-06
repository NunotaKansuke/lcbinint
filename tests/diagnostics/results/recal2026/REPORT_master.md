# The certified-algorithm recalibration campaign — consolidated report

August 2026. lcbinint binary and triple finite-source magnification, re-measured
end to end against the certified component algorithm, and compared with
VBMicrolensing, microlux, and lcbinint's own JAX backend.

This file is the single place to start. It consolidates `README.md` (corpus,
resolution and routing rules), `REPORT_speed.md` (the external comparison),
`REPORT_tangency_defects.md` and `REPORT_tangency_fixes.md` (two correctness
defects found while checking the references the speed numbers rest on) into one
narrative, and adds §7, which states which claims the data supports and with
exactly what qualification. Those files remain the primary record for their own
sections and carry detail this one omits; where they disagree with this file,
this file is the later reading.

**Nothing in this campaign has been applied to the runtime.** Every rule below
is a measurement of what the corpus supports, not a description of what
lcbinint currently does. `docs/finite-source-auto-calibration.md` remains the
specification of the shipping rule and carries a status note to that effect.

---

## Contents

| § | |
|---|---|
| 1 | [What the campaign established](#1-what-the-campaign-established) |
| 2 | [Corpus, conventions, and what a number here means](#2-corpus-conventions-and-what-a-number-here-means) |
| 3 | [Accuracy control — certificate, indicator, error estimator](#3-accuracy-control--certificate-indicator-error-estimator) |
| 4 | [The internal rules — nbin, grid, route](#4-the-internal-rules--nbin-grid-route) |
| 5 | [The external comparison — VBM, microlux, JAX, triple](#5-the-external-comparison) |
| 6 | [Correctness at `d/rho ≈ 1`](#6-correctness-at-drho--1) |
| 7 | [Writing the paper: claims, support, qualifications](#7-writing-the-paper-claims-support-qualifications) |
| 8 | [Threats to validity](#8-threats-to-validity) |
| 9 | [Reproduction and file index](#9-reproduction-and-file-index) |

---

## 1. What the campaign established

Six results, in the order they matter.

**(1) The shipping resolution rule is over-resolved by one to two orders of
magnitude.** The certificate proves disk support and topology, so correctness no
longer rides on the bin count, and the bin count can be chosen for accuracy
alone. Measured against what the corpus actually needs, the shipping rule spends
25–278× the required work. A constant per tolerance covers 99.3–100% of an
independent holdout at 16/50/128 bins (Cartesian) and 24/100/320 (polar). §4.1.

**(2) Against VBMicrolensing on a uniform source, VBM wins essentially always.**
Median factor 17–31×, lcbinint under 1% of blocks at every tolerance. This is
not close and should be stated plainly. §5.1.

**(3) Limb darkening reverses it, through one mechanism.** Going from a uniform
to a linear profile multiplies VBM's cost by 1.02× at 1e-2, 3.15× at 1e-3 and
**11.66× at 1e-4**, while lcbinint's cost multiplies by **1.00×** at every
tolerance. A contour method stacks annuli; an inverse-ray grid already visits
the whole disk and a radial weight is one multiply per ray. The crossing happens
wherever the limb-darkened contour cost has grown enough — which is a
limb-darkened source, at 1e-4, overlapping the caustic. §5.2.

**(4) The winning regime, stated as tightly as the corpus supports:**

> **limb-darkened ∧ tolerance ≤ 1e-4 ∧ `d/rho` < 0.95 ∧ `A` > 3**
> → lcbinint wins **74.9%** of blocks, median **2.12×**, p90 **6.4×** (n=211).

Relaxing the distance cut to 1.5 makes it a coin flip; removing it drops to 38%.
§5.3.

**(5) Reliability runs opposite to speed, and it reproduces across two
independent contour implementations.** At 1e-4, VBM fails to reach the requested
accuracy on 20–24 blocks where lcbinint succeeds, against 2–4 the other way.
microlux fails on **320 of 909** judgeable limb-darkened blocks at 1e-4 — 35% —
and the blocks it drops are the hard ones, so its surviving median is
conditioned on its own success. §5.1, §5.4.

**(6) The JAX backend's headline cost was a missing `jax.jit`, and fixing it
changed the answer by a factor of nine.** The public
`Options(jax=True, nbin="auto")` pipeline carried a ~590 ms per-call constant
because `binary_magnification_native_pipeline_trajectory` was never jitted.
Before the fix the backend cost 43–62× native and won 0.0–2.2% of blocks; after,
it costs **4.8–6.2×** and wins **19–22%**. The residual sits just above the
independently measured FFI-kernel ratio of 3.1–3.9×, which is where it should
be. §5.5.

Two findings that are not speed results but bound how the speed results may be
quoted:

**Two correctness defects live at `d/rho ≈ 1`** — one in the Cartesian row scan,
one in the shared grazing quadrature — and the second implicated the references
this campaign's accuracy columns are read against. Both are now fixed and
regression-pinned. Aggregates over ~1300 blocks are unmoved; **individual blocks
in `d/rho ∈ [0.85, 1.05]` must not be quoted without arbitration.** §6.

**The flood fill is not seed-order independent.** Fixed for the two mechanisms
found (seed-union retention, canonical seed ordering), but this is a structural
property worth stating as a limitation rather than claiming closed. §6.3.

### One measurement is outstanding

The defect-B fix (§6.2) raised the grazing quadrature's cost from ~0.021 s to
~0.400 s per grid, and that route is selected by 73–93% of rows at
`d/rho ∈ [0.95, 2.0]`. **Roughly 37% of the corpus therefore carries lcbinint
timings that predate the fix**, and those rows are optimistic for lcbinint.

Result (4) — the headline — is **not** affected: it lives at `d/rho < 0.95`,
where the route is used 0–4.4% of the time. What is affected is the
`0.95–1.5` and `1.5–4` rows of §5.3's distance table, which show lcbinint
losing and would show it losing by more. Regenerating the speed corpus in that
band is the campaign's one known open item. §8.

---

## 2. Corpus, conventions, and what a number here means

### The corpus

160 binary lens cases, seed 20260803, sampled over `s ∈ [0.6, 3.4]`,
`q ∈ [1e-5, 0.3]`, `rho ∈ [3e-5, 0.5]`. Each case is visited at a ladder of
caustic distances expressed in source radii —
`d/rho ∈ {0, 0.25, 0.5, 0.8, 0.95, 1.0, 1.1, 1.35, 1.7, 2, 3, 5, 10, 30}` plus an
unlabelled far-field group — and run under both a uniform and a linearly
limb-darkened profile (`c = 0.5`): **2880 rows**, 1440 blocks per profile.

The ladder is deliberately dense between 0.8 and 2: that is where the source limb
grazes a fold, and it is both the regime the speed result turns on (§5.3) and the
one that produced the correctness defects of §6.

Triple lens is reduced scope by design: 32 cases, 48 epochs per block spanning
0.6 source radii, both profiles, three tolerances.

Limb darkening uses the convention shared by lcbinint, VBM's `a1`, and
microlux's `LinearLimbDarkening.a`: `I(mu)/I0 = 1 - c(1-mu)`, normalised by
`1 - c/3`. Verified equivalent across all three engines before any timing was
taken.

Frame conventions, which are a common source of silent disagreement: VBM is
called as `BinaryMag2(s, q, -x, y, rho)`; microlux uses the MulensModel
centre-of-mass frame with `MICROLUX_X_SIGN = -1.0`. VBM's `Tol` is **absolute**
and its `RelTol` is relative; both are swept.

### Three conventions govern every number

**Timing is per block, not per call.** A block is 24 epochs spanning 0.4 source
radii (48 over 0.6 for triple), timed three times, median reported.
Single-epoch timings on this problem are dominated by cache state. *The block
length is a first-order choice for the JAX axis specifically* — see §5.5.

**Cost is read at delivered accuracy, never at a requested tolerance.** For each
block and each engine, the cheapest setting that *actually met* the accuracy is
used. A knob is a request; the error column is what happened.

> This makes the forced-grid curves an **oracle lower bound**: they pick the
> cheapest of nine bin counts after seeing which met the target, which no caller
> can do at runtime. `lcbinint_auto` is the only column that reflects one
> decision made in advance. Any comparison between a forced grid and a routed
> pipeline has to carry this.

**Blocks whose reference floor is coarser than the accuracy under discussion are
excluded and counted.** Asking whether an engine reached 1e-4 on a block whose
converged reference is good only to 1e-3 is a question about the reference. This
is why 1e-4 columns judge 797–909 blocks rather than 1440.

### A comparison that must not be made

`lcbinint_auto` "wins" 50–81% with a median of 12–14× in the `A ∈ [1, 1.13]`
bucket, and its p90 sits pinned at ~14.1 in every cell of §5's routed table.
Those are the blocks the pipeline answered with the **point-source formula**.
That is a routing decision measured against a contour integral, and it says
nothing about either engine's finite-source algorithm. It is reported separately
throughout and must never be placed on the same axis as a quadrature timing.

---

## 3. Accuracy control — certificate, indicator, error estimator

This section is the "why now" of the whole campaign, and it is the part most
easily lost when only the tables are read.

### 3.1 What the certificate does and does not prove

The component certificate proves **disk support and topology** — that the image
components found are the components that exist, and that the source disk is
covered. It does **not** prove that the quadrature error meets the requested
`reltol`. Those are different claims and conflating them is the mistake the old
calibration made structurally.

The consequence is the campaign's premise. Under the old algorithm the bin count
was doing two jobs: resolving the integrand *and* being the mechanism by which
the image search was trusted to have found everything. Correctness therefore
depended on nbin, so nbin had to be set defensively. With the certificate
carrying the support/topology claim, the bin count is answerable to accuracy
alone — which is why §4.1 finds one to two orders of magnitude of slack, and why
that slack is real rather than an argument for cutting safety margin.

### 3.2 The boundary-area indicator has a first-order ceiling

The convergence gate was the boundary-area indicator: the count of cells the
source limb and the caustic cut, as `perimeter × cell_width / area`. It decays
like `1/source_bins` **however fast the integrated area actually converges**.

That is a first-order ceiling, and refinement cannot fix it. On the binary
tangency it is 6× pessimistic by 4096 bins; on the triple cusp over **1e6×**. At
the 4096-bin ceiling it cannot certify better than 2e-4 absolute, so a request of
`reltol = 1e-5` fails closed on an answer that is right to fourteen digits.

Two acceptance tests had been failing since the day they were written for
precisely this reason — the triple one from `060d4be`, the binary `[1e-05]` case
from `0242bb2` — and neither failure was ever about the value. The values were
right to 1.1e-6 and 1e-14 respectively. What refused them was the gate.

### 3.3 The error estimator: measure the error instead of bounding it

Commit `5e018b9` replaces the bound with a measurement, in two pieces.

**`grid_pair_error_estimate` — the Richardson correction.** The difference
between two grids measures the *coarser* one. For a first-order scheme
`|A_fine − A_coarse| ≈ (r − 1)` times the fine-grid error, where `r` is the
ratio of bin counts. Halving needs no correction, which is why the plain
difference was correct for the `/2` comparison that originally used it — but the
automatic retry ladder jumps from 400 straight to 4096, and charging
`|A(4096) − A(400)| = 1.21e-4` to a value whose own error is 4.3e-6 made the
estimate *worse the more the loop refined*. Dividing by `(r − 1)` fixes the
sign of that behaviour.

**`reconcile_with_half_resolution` — spend one evaluation at the end.** When the
retry ladder is exhausted, one half-resolution evaluation is spent and the
measured pair is allowed to rule **in both directions**: a row the indicator
passed still has to survive it (the pre-existing anti-aliasing check, unchanged),
and a row the indicator rejected is admitted only if the measurement is inside
the budget. The triple Cartesian route has no ladder at all, so it reconciles
directly.

**What still fails closed, and deliberately.** `support_proven` still forces an
infinite error, and the underresolved guard still vetoes. Neither is a statement
about *grid* error, so neither is something a second grid may overrule. The
binary geometry at `reltol = 1e-6` still returns NaN. This is the fail-closed
contract, and it is pinned by test (`c90d1c1`).

Regression status: `test_binary_cusp_component`, `test_triple_cusp_component`,
`test_component_refinement` — 29 passed; `ctest unit_core` passed.

### 3.4 The consequence for the paper

The honest framing of the accuracy story is:

> lcbinint's error control is a **measured pair estimate with a proven-support
> precondition**, not a proven error bound. Support and topology are certified;
> quadrature error is measured by Richardson-corrected grid pairing and reconciled
> against a half-resolution evaluation, and the routine fails closed when support
> is not proven.

That is a weaker claim than "certified accuracy" and a stronger one than "an
indicator that happens to work". §5's reliability columns — where lcbinint
misses the requested accuracy on 2–4 blocks against VBM's 20–24 and microlux's
320 — are the empirical evidence that it holds in practice.

---

## 4. The internal rules — nbin, grid, route

### 4.1 The nbin rule

Bins the corpus actually requires, against what the shipping rule spends:

| grid | tolerance | required median | required p99 | shipping median | work vs required |
|---|---|---|---|---|---|
| Cartesian | 1e-2 | 4 | 14 | 64 | 178× |
| Cartesian | 1e-3 | 12 | 50 | 64 | 39× |
| Cartesian | 1e-4 | 32 | 128 | 400 | 278× |
| polar | 1e-2 | 6 | 24 | 64 | 156× |
| polar | 1e-3 | 16 | 100 | 64 | 25× |
| polar | 1e-4 | 32 | 320 | 400 | 156× |

A constant bin count per tolerance, validated on an independent holdout against
a 99% coverage target:

| grid | tolerance | constant bins | holdout coverage |
|---|---|---|---|
| Cartesian | 1e-2 | 16 | 99.58% |
| Cartesian | 1e-3 | 50 | 99.81% |
| Cartesian | 1e-4 | 128 | 99.26% |
| polar | 1e-2 | 24 | 100.00% |
| polar | 1e-3 | 100 | 99.77% |
| polar | 1e-4 | 320 | 99.94% |

Every cell clears 99% except Cartesian at 1e-4, the one place a constant is not
enough on its own; the `linear` rule column in `nbin_rule.json` gives a
rho-dependent alternative reaching 99.83% there at 160 median bins.

Two things to state plainly. **Cartesian needs fewer bins than polar at every
tolerance** — but bins are not seconds, and §4.2 decides the grid on measured
time, not on this table. And **the required counts are heavily right-skewed**:
p99 is three to ten times the median, so a rule tuned to the median fails a
percent of the corpus badly. This is why coverage, not mean error, is the
selection criterion.

### 4.2 Which grid — polar or Cartesian

The median polar/Cartesian ratio sits on 1.00 in every cell and answers nothing.
The question is a corpus one: total time against an oracle that picks the better
grid per block. Always-Cartesian costs 1.29–1.45× the oracle; always-polar
1.51–2.72×.

Splits on rho, `d/rho`, and mass ratio produce no ordered structure. The split on
**magnification** lands immediately — the entire cost of always-Cartesian sits in
the top magnification quartile, the other three being within 4% of the oracle.
Re-derived on the *point-source* magnification, which the multipole stage has
already computed when the decision is made:

> **Rule: `A_point > 200` → polar, else Cartesian.**

| | always-Cartesian | `A_point > 200` | share sent polar |
|---|---|---|---|
| uniform 1e-2 | 1.290× | **1.096×** | 10.4% |
| uniform 1e-3 | 1.318× | **1.125×** | 10.1% |
| uniform 1e-4 | 1.380× | **1.256×** | 14.6% |
| linear 1e-2 | 1.303× | **1.068×** | 10.2% |
| linear 1e-3 | 1.369× | **1.171×** | 10.0% |
| linear 1e-4 | 1.445× | **1.223×** | 12.8% |

200 is the joint optimum in all six cells, and the optimum is **flat from 100 to
500** (`figures/grid-switch.pdf`) — the corpus supports the decade, not two
significant figures. Adding a rho condition does not improve it. Using the true
finite-source magnification instead of the point-source one reaches only
1.05–1.19×, so most of the remaining gap to the oracle is block-level noise
rather than predictable structure.

**A second clause is required for correctness, not speed.** §6 found that the
Cartesian row scan could not represent two intervals in one row, which is exactly
what the tangency band needs. The rule as it should be *implemented* therefore
carries `d/rho ∈ [0.88, 1.02] → polar` as well. That clause's cost is not
measured here, and the defect it worked around has since been fixed directly
(§6.1) — so it is a belt-and-braces recommendation, not a load-bearing one.

> **Do not carry that clause into the JAX backend.** §5.5.2 measures the JAX
> polar grid returning a silently wrong value — `support_valid=True`, identical
> from 16 to 400 bins, up to 20% low — in exactly the band this clause would
> route *into*, `d/rho` from ~0 to ~1.1 at `A_point > 300`. The two sections are
> about different implementations of the same choice and they point opposite
> ways; the JAX one is a measured defect, this one is a precaution against a
> defect since fixed elsewhere. Reconciling them requires refitting this rule,
> which has not been done.

### 4.3 Which route

`lcbinint_auto` records the methods it used, which turns the routing thresholds
into a measurement rather than an assumption. Delivered error grouped by route
(`route_audit.json`):

* **`point_source`: 0% miss at every accuracy and both profiles** (worst error
  0.00e+00 at 1e-4). The boundary is sound.
* **Every pure inverse-ray route: 0% miss.**
* **`hexadecapole` at 1e-4: 7.6–8.8% miss**, worst 1.53e-4 (1.5× target).
* **`inverse_ray_polar + source_plane_quadrature` at 1e-4: 8.0–11.1% miss**,
  worst 2.62e-4 (2.6× target).

The two failing routes fail at **opposite ends of the magnification axis**, which
is why a single predicate shape misreports one of them. Taking instead the tight
bounding box of the misses in `(rho, A)` — both available before the route is
chosen:

| route | box | misses caught | legitimate hits also rejected |
|---|---|---|---|
| hexadecapole @1e-4 | `rho ∈ [0.015, 0.272]`, `A ∈ [1.01, 2.36]` | 16/16 | 63 (31.5%) |
| quadrature @1e-4 | `rho ∈ [3.8e-5, 0.020]`, `A ∈ [39.8, 8770]` | 6/6 | 157 (22.3%) |

**Neither is worth imposing.** The hexadecapole box costs 63 blocks that would go
from 0.14 ms/epoch to roughly 80 ms/epoch, to remove errors exceeding the target
by at most 1.5×. The recommendation is to report the trade rather than tighten
the cut; the real fix for the hexadecapole is to **scale its acceptance test with
rho**, which is a code change needing its own study.

The quadrature misses are more interesting than their count. All six sit at
`A ≥ 39.8` with `d/rho ∈ [0.95, 1.7]` — the tangency regime. §6 reclassifies
them: they are not a route boundary set too loose but a genuine defect in the
grazing quadrature, since fixed. The route audit's verdict on every other route
is unaffected.

---

## 5. The external comparison

### 5.1 VBMicrolensing, binary — the headline

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

Read the **p90** column. On uniform sources it never reaches 0.32 — there is no
tail in which lcbinint competes at all. On limb-darkened sources it crosses 1.0
at 1e-3 and reaches 2.7 at 1e-4. The distribution does not shift uniformly; it
**grows a tail**, and §5.3 is about which blocks are in it.

The `missed` columns are the reliability result: at 1e-4 VBM misses 20–24 blocks
against lcbinint's 2–4. **The engine that is faster on the median is the engine
that more often fails to deliver the requested accuracy.**

Median and p90 cost per epoch, for absolute scale:

| profile | accuracy | lcbinint Cartesian | VBM |
|---|---|---|---|
| uniform | 1e-2 | 1.22 ms | **0.038 ms** |
| uniform | 1e-3 | 1.25 ms | **0.041 ms** |
| uniform | 1e-4 | 2.44 ms | **0.062 ms** |
| linear | 1e-2 | 1.24 ms | **0.041 ms** |
| linear | 1e-3 | 1.26 ms | **0.168 ms** |
| linear | 1e-4 | 2.28 ms | **0.863 ms** |

32× on uniform at 1e-2 collapsing to 2.6× on limb-darkened at 1e-4. At linear
1e-4 the p90 is 10.3 ms for VBM against 13.3 ms for lcbinint — essentially even.

### 5.2 The mechanism

Same block, same geometry, same accuracy, uniform versus linear:

| accuracy | engine | blocks paired | median linear/uniform cost | p90 |
|---|---|---|---|---|
| 1e-2 | VBM | 1348 | 1.02 | 6.80 |
| 1e-2 | lcbinint grid | 1351 | 1.01 | 1.07 |
| 1e-3 | VBM | 1344 | **3.15** | 16.87 |
| 1e-3 | lcbinint grid | 1347 | **1.00** | 1.08 |
| 1e-4 | VBM | 1317 | **11.66** | 52.49 |
| 1e-4 | lcbinint grid | 1338 | **1.00** | 1.19 |

**This is the whole result in one table**, and it is a structural difference
rather than a tuning difference. A contour method integrates the limb-darkened
profile by stacking annuli, so its work is multiplied by the number of annuli the
tolerance demands — and that number grows as the tolerance tightens, which is why
the multiplier runs 1.02 → 3.15 → 11.66. An inverse-ray grid already visits the
whole disk; applying a radial weight to rays it has shot anyway is one multiply
per ray. lcbinint's cost is **flat in the profile to within 1%** at every
accuracy, p90 1.19 at worst.

Everything in §5.3 follows from this table. **lcbinint does not get faster on
limb-darkened sources; VBM gets slower, by a factor the tolerance controls.**

### 5.3 Where lcbinint wins

By distance to the caustic:

| d/rho | accuracy | profile | blocks | win rate | median VBM/ours | p90 |
|---|---|---|---|---|---|---|
| < 0.95 (source over the caustic) | 1e-2 | linear | 301 | 5.0% | 0.266 | 0.729 |
| < 0.95 | 1e-3 | linear | 300 | **46.3%** | 0.953 | 2.346 |
| < 0.95 | 1e-4 | linear | 300 | **69.3%** | **1.815** | 6.129 |
| 0.95 – 1.5 | 1e-4 | linear | 319 | 17.2% | 0.321 | 2.025 |
| 1.5 – 4 | 1e-4 | linear | 260 | 16.9% | 0.357 | 1.515 |
| > 4 (far field) | 1e-4 | linear | 287 | 1.7% | 0.095 | 0.589 |
| < 0.95 | any | uniform | 302 | 0.0% | 0.041–0.059 | ≤ 0.130 |

The gradient is monotone and steep. The winning regime is where the source disk
**overlaps** the caustic, and it is gone by `d/rho > 1.5`.

By magnification (linear profile; cells are `win rate / median VBM-over-ours`):

| A | 1e-2 | 1e-3 | 1e-4 |
|---|---|---|---|
| < 1.2 | 0.0% / 0.112 | 0.8% / 0.115 | 3.7% / 0.300 |
| 1.2 – 3 | 0.7% / 0.099 | 8.0% / 0.104 | 21.5% / 0.328 |
| 3 – 10 | 2.4% / 0.088 | 21.3% / 0.156 | **34.4%** / 0.515 |
| 10 – 100 | 2.0% / 0.075 | **27.0%** / 0.255 | **40.4%** / 0.647 |
| > 100 | 3.1% / 0.068 | 12.6% / 0.179 | **33.3%** / 0.493 |

Magnification helps to `A ≈ 100` and then turns over — that turnover is the
`A_point > 200` grid-switch boundary of §4.2 appearing on the speed axis.

By rho and mass ratio (linear, 1e-4) both axes are nearly flat: a factor of 1.7
in win rate across four decades of rho (21.9 / 23.8 / **31.2** / 18.1% for
`<1e-3` / `1e-3–1e-2` / `1e-2–1e-1` / `>1e-1`), against a factor of 40 across
`d/rho`. **rho is not the variable that decides this**, which is worth saying
because the natural expectation is that a big source favours a grid. It does
not, at fixed `d/rho`: what favours the grid is the source *covering* the
caustic, and `d/rho` measures that directly while rho does not.

The joint condition:

| condition (linear profile) | blocks | win rate | median VBM/ours | p90 |
|---|---|---|---|---|
| 1e-3, `d/rho < 0.95` | 300 | 46.3% | 0.953 | 2.346 |
| 1e-3, `d/rho < 0.95`, `A > 3` | 211 | **54.5%** | **1.099** | 2.519 |
| 1e-4, `d/rho < 0.95` | 300 | 69.3% | 1.815 | 6.129 |
| 1e-4, `d/rho < 0.95`, `A > 3` | 211 | **74.9%** | **2.120** | 6.397 |
| 1e-4, `d/rho < 0.95`, `A > 10` | 130 | 70.8% | 2.055 | 7.101 |
| 1e-4, `d/rho < 1.5`, `A > 10` | 247 | 49.4% | 0.946 | 5.086 |
| 1e-4, `A > 10` (no distance cut) | 395 | 37.7% | 0.576 | 3.700 |

The regime is 211 of 1346 limb-darkened blocks at 1e-4 — about **16% of the
corpus**. But the corpus samples `d/rho` uniformly in a log-spaced ladder out to
30, which is not how a real event distributes its epochs. **In an event with a
caustic crossing, the epochs that matter for the model are exactly the ones
inside the winning regime**, and that is the sentence the paper should carry.

Restated without `d/rho`, which is a corpus label rather than something a user
chooses:

> lcbinint is the faster engine for **limb-darkened sources during a caustic
> crossing at tolerances of 1e-4 or tighter**. Outside a crossing, or on a
> uniform source, or at 1e-2, VBM is faster by one to two orders of magnitude.

### 5.4 microlux

An independent contour implementation, used to test whether §5.2's mechanism is a
property of contour integration or a property of VBM.

All ratios below are **load-calibrated** — §8 explains why and by how much.

| profile | accuracy | judgeable | reached | missed | win rate | median microlux/ours | p10 – p90 |
|---|---|---|---|---|---|---|---|
| uniform | 1e-2 | 1350 | 1334 | 16 (1.2%) | 63.1% | 0.749 | 0.105 – 1.802 |
| uniform | 1e-3 | 1349 | 1332 | 17 (1.3%) | 64.5% | 0.706 | 0.102 – 1.662 |
| uniform | 1e-4 | 797 | 769 | 28 (3.5%) | 79.2% | 0.456 | 0.084 – 1.178 |
| linear | 1e-2 | 1347 | 1319 | 28 (2.1%) | 28.4% | **3.076** | 0.573 – 12.856 |
| linear | 1e-3 | 1346 | 1247 | 99 (7.4%) | 28.9% | **3.108** | 0.569 – 12.696 |
| linear | 1e-4 | 909 | 589 | **320 (35.2%)** | 37.8% | **1.159** | 0.565 – 5.807 |

**The mechanism reproduces, and more strongly than against VBM.** On uniform
sources microlux is ahead by 1.3–2.2× on the median and its margin *widens* as
the accuracy tightens. On limb-darkened sources the sign flips at every accuracy:
lcbinint ahead by a median 3.1× at 1e-2 and 1e-3, and nearly 13× at the p90.

The 1e-4 limb-darkened cell is the weakest of the three (37.8%, median 1.159) and
must be read with the miss column, which **understates lcbinint rather than
overstating it**. microlux fails to reach 1e-4 on 320 of 909 judgeable blocks;
the ratio is computed over the 589 where it succeeded. Of the 320: **114**
exhausted the adaptive sampler's budget (`No enough space to insert new
samplings`), **89** were censored for already exceeding 0.25 s/epoch at a looser
setting, and the remainder simply delivered an error above target at every one of
the seven settings. The blocks microlux drops are the hard ones, so the surviving
median is conditioned on its own success.

Two separate effects shrink that row and must not be conflated: the judgeable
count falls 1346 → 909 because of the **reference floor** (a question about the
reference, applied identically to every engine), and 909 → 589 because of
**microlux**. Only the first applies to the uniform rows, where the miss rate
stays at 1.2–3.5%.

One methodological note worth carrying: microlux's accuracy knob is
**non-monotone**, because `default_strategy` caps the adaptive budget. A tighter
request can deliver a worse answer. Settings that hit the cap are excluded from
the accuracy curve rather than counted as points on it — hitting a ceiling is not
a measurement of a method's cost at that accuracy.

### 5.5 The JAX backend

**This axis was re-measured after five fixes and the earlier numbers are
superseded. Do not quote the pre-fix table.** The `jax.jit` omission below is
the largest single one and the easiest to describe, but `ext_capfix` was
measured with four further changes already in the tree, and the corpus figure
should not be attributed to one line — see "Four more changes are in the same
measurement".

#### What was wrong

`binary_magnification_native_pipeline_trajectory` carried **no `jax.jit`**, and
`jax_backend.py:799` called it directly. Every call re-traced the whole program —
fourteen Cartesian FFI calls, two `lax.map` bodies, a fourteen-way `lax.switch` —
and dispatched it primitive by primitive. Tracing cost scales with program size,
not with data, and the observed shape matched exactly: flat in epochs, flat in
buckets, flat in `caustic_bins`.

The constant was **~590 ms per call, before the pipeline looked at a single
epoch**:

| epochs | native ms/call | JAX ms/call | JAX ms/epoch | ratio |
|---:|---:|---:|---:|---:|
| 1 | 1.37 | 585 | 585 | 428× |
| 24 | 2.11 | 638 | 26.6 | 302× |
| 96 | 4.31 | 653 | 6.81 | 152× |
| 384 | 13.03 | 675 | 1.76 | 51.8× |
| 1536 | 47.80 | 787 | 0.512 | 16.5× |

Ruled out as causes: compilation (first call is 2.0–2.3 s and *every* warm call
paid the 590 ms), the caustic scan (flat from `caustic_bins` 100 to 4000 while
native moved 0.18 → 3.59 ms), and the bucket ladder (truncating fourteen buckets
to one via `max_source_bins` left it at 585 ms). The decisive line: fixing `nbin`
— which leaves the calibrated pipeline for the plain fused trajectory — dropped
the same call from **590 ms to 0.68 ms**, faster than native's 1.38 ms.

Wrapping the existing function in `jax.jit`, changing nothing else:

| epochs | eager ms | jitted ms | speed-up | compile s | value change |
|---:|---:|---:|---:|---:|---:|
| 1 | 590.12 | 1.530 | 386× | 1.12 | none (0.0) |
| 24 | 644.75 | 3.450 | 187× | 1.37 | none (0.0) |
| 96 | 655.25 | 9.429 | 69× | 1.49 | none (0.0) |
| 384 | 678.67 | 32.919 | 21× | 1.39 | none (0.0) |

Bit-identical results.

#### Four more changes are in the same measurement

`ext_capfix` is not a jit-only sweep. `ext_jitfix/` is — it was run with the
`jax.jit` fix and nothing else — and it still records **9.3% of rows failing
closed at reltol 1e-2**, the identical rate as the pre-fix `ext_discovery`. The
jit fix removed a constant; it did not touch the reliability defect, and it was
not the last thing to land. Four further changes ship in the measured tree:

1. **The tile capacity was sized from the physics.** Every rung of the
   calibrated ladder paired a resolution with `tile_capacity = resolution^2`. A
   tile is 16×16 cells carrying ~150 filled cells and a cell is
   `source_radius/resolution` on a side, so that capacity stopped the
   claimed-tile discovery at a **total image area of ~47 source areas** — a hard
   magnification ceiling independent of the tolerance asked for. Caustic-adjacent
   epochs overflowed every rung at once, the certificate had no supported pair,
   and the pipeline returned NaN where native converged cheaply. Inverting the
   relation, a capacity of `k·resolution²` admits `A ~ 48k` regardless of
   resolution; `_tile_capacity` in `lcbinint_jax/trajectory.py` now ships
   `min(4194304, next_pow2(512·resolution²))`.
2. **The Cartesian ladder went from three rungs to a two-phase two-rung ladder.**
3. **`moment_mode` is picked from the concrete limb coefficients**, so a uniform
   source no longer integrates two limb coefficients it does not have.
4. **The dead `base` pipeline was removed** — a second complete magnification
   solve, run on every epoch, backing a fallback set that was always empty.

Together those took the hot caustic-adjacent Cartesian epoch from **255 to 31.9
ms/epoch after jit was already applied**, so the corpus ratio is the product of
all five changes, not of the one line.

The capacity fix is also what moved the reliability number. Rows where
`lcbinint_jax` reported no usable achieved error, counting only genuine engine
failures (rows whose own `reference_floor` is non-finite, and rows the pass never
attempted, are excluded — they say nothing about this engine), out of 2812:

| accuracy | `ext_discovery` (pre) | `ext_capfix` (post-capacity) | after the route fix |
|---|---|---|---|
| 1e-2 | 262 (9.3%) | 74 (2.6%) | **14 (0.50%)** |
| 1e-3 | 125 (5.1% of attempted) | 75 (2.8%) | **30 (1.1%)** |
| 1e-4 | 58 (2.5% of attempted) | 75 (2.8%) | **28 (1.0%)** |

The capacity defect was **tolerance-dependent** — worst where the ladder started
coarsest — and the residual was flat at ~2.7% across all three tolerances, i.e.
a different defect. That second defect has since been found and fixed. The last
column re-scores every previously fail-closed row against the same stored
per-epoch references (`speed_discovery/block-*.json`, joined on
`(case_id, profile, x)`); it is a re-score of those rows only, not a fresh
sweep, so it moves the reliability number and leaves the timing table below
untouched.

Of the rows that stopped failing closed, most now certify correctly — the worst
achieved error among them is 6.5e-4 at 1e-3 — but not all: one row at 1e-2 and
three at 1e-4 come back *over* budget rather than right. That is §5.5.2, and it
is the reason the two columns must be read together. A fail-closed rate is only
half of a reliability statement.

#### 5.5.1 The residual was a route with no way back

`prefer_polar` (`resolution.py:97`) is a *prediction* from the resolution
regression, and the pipeline treated it as a commitment: `cartesian_candidate`
excluded every `prefer_polar` epoch, so an epoch the prediction sent to the
polar grid had no second route. Where the prediction is wrong the polar grid
finds no valid support and returns roughly half the true magnification with
`support_valid` clear — and the epoch fell closed with the Cartesian ladder
sitting unarmed beside it, certifying that same epoch normally at the same
resolution.

The fix arms the Cartesian ladder for any epoch whose polar rung failed to
certify, in three places:

* `trajectory.py` — the pipeline the corpus sweep runs
  (`cartesian_candidate |= polar_candidate & ~polar_converged`);
* `api.py:binary_inverse_ray_auto` — the public dispatcher, which had the
  Cartesian → polar overflow fallback but not its mirror image;
* `calibrated.py:grid_path` — an independent bottom-rung bug: at
  `bucket_index == 0` the comparison had nothing below it to compare against
  and reported a fabricated error of exactly 0. It now reaches upward, as the
  trajectory ladder already did. At 16 bins the same geometry now reports its
  real error, 7.4e-3.

`tests/jax_ir` (240 tests) passes with all three applied.

The 14 rows still failing closed at 1e-2 are a *third*, separate defect, and it
is the one the capacity fix did not finish: they are discovery overflow at
`_MAXIMUM_TILE_CAPACITY = 4194304`. Tile demand at the geometries involved runs
437k (resolution 24), 765k (40), 1.33M (64), 2.21M (100), 3.06M (128) and
saturates from 160 up, so the ladder runs out of ceiling before it runs out of
rungs. They cluster at **large** ρ (0.27, 0.44, 0.83, 1.0) — the opposite end
from the capacity defect. Raising the ceiling is not free: it costs memory on
every epoch, and it would invalidate the `ext_capfix` timing measurement that
the table below reports. Not done.

#### 5.5.2 What the NaN was hiding: the polar route is wrong where the certificate cannot see it

One re-scored row came back **9.7% outside its budget** rather than NaN
(case 118 uniform, `x = +0.006307579`, epoch 0: 1711.83 against a reference of
1895.5463 at reltol 1e-2). The route fix cannot have caused it — the fix only
arms epochs whose polar rung failed to certify, and this epoch reports
`used_polar=True, support_valid=True, value_converged=True`. Its code path is
byte-identical before and after. The error is pre-existing; the block's NaN was
masking it.

Rung by rung (`polar_rungs.py`), the polar route returns 1710–1714 with
`support_valid=True` at **every** resolution from 16 to 400 bins — a relative
error of 9.5e-2 that does not move — while the Cartesian route at the same
rungs returns 1895.55, converging to 1.3e-6. **The certificate cannot detect a
route that is wrong at every resolution.** Two adjacent rungs agreeing is
evidence of convergence, not of correctness, and this is the standing
counterexample.

**This is not one epoch, and it is already in the `ext_capfix` data.** The
sweep recorded it faithfully; no one had read that column. Rows with a finite
reference floor, scored against their own stored references, where
`lcbinint_jax` delivered *worse* than the accuracy it was asked for:

| profile | accuracy | scored | missed | rate | worst |
|---|---|---|---|---|---|
| uniform | 1e-2 | 1317 | 16 | 1.21% | 2.455e-1 |
| uniform | 1e-3 | 1288 | 18 | 1.40% | 2.455e-1 |
| uniform | 1e-4 | 1283 | 37 | 2.88% | 2.447e-1 |
| linear | 1e-2 | 1316 | 17 | 1.29% | 2.443e-1 |
| linear | 1e-3 | 1264 | 53 | 4.19% | 2.443e-1 |
| linear | 1e-4 | 1266 | 50 | 3.95% | 2.435e-1 |

The worst error is **24.5% at every tolerance** — the signature of a route
error that is fixed in size while the budget shrinks around it, which is also
why the rate roughly triples from 1e-2 to 1e-4. Read this next to claim 9 and
headline (5): the engine this report presents as the accurate one has a miss
rate of its own, smaller than VBM's and microlux's but not zero, and it is not a
resolution question. **It does not retract headline (5)** — that rests on
`lcbinint_auto`, the native path, whose 2–4 misses at 1e-4 are measured
separately. The table above is the JAX backend only.

Attributing the misses: re-running the shipped pipeline over 720 sampled blocks
(every fourth block, reltol 1e-2, `failopen_scan.py`) reproduces 12 blocks with
an over-budget epoch out of 680 scored, and **18 of the 19 offending epochs are
`used_polar=True, method=2`** — `d/rho` from 0.002 to 0.95, errors 1.2e-2 to
1.7e-1. The nineteenth is a Cartesian epoch at `A = 1.73` missing 1e-2 by half a
digit. Joining those 12 back to `ext_capfix`: 6 are rows the sweep counted (its
recorded `worst` matches this re-run to every digit), 5 have a non-finite
reference floor and so are excluded from the accounting, and 1 is case 118 —
excluded because it *used* to fail closed. Removing the NaNs did not create the
fail-open; it moved one row from the invisible column to the visible one.

Mechanism, measured at a fixed rung (resolution 100) over 126 cases / 756
sampled epochs, polar against Cartesian (`polar_vs_d.py`): 435 epochs disagree
by more than 1e-2, of which **92 have polar `support_valid=True`** — wrong
without saying so. Most of those 92 are unreachable in the shipped pipeline (low
`A`, `prefer_polar=False`; the hexadecapole route accepts them long before polar
is consulted). The reachable set is the **9 epochs with `prefer_polar=True`**:

| `d/rho` | `A_point` | polar | Cartesian | gap |
|---|---|---|---|---|
| 0.186 | 1353.5 | 775.07 | 784.28 | 1.2e-2 |
| 0.681 | 579.1 | 651.71 | 673.04 | 3.2e-2 |
| 0.722 | 638.3 | 738.34 | 751.36 | 1.7e-2 |
| 0.746 | 604.2 | 713.23 | 729.44 | 2.2e-2 |
| 0.924 | 717.4 | 917.51 | 961.82 | 4.6e-2 |
| 0.974 | 680.7 | 716.97 | 894.87 | **2.0e-1** |
| 1.026 | 646.2 | 685.26 | 791.06 | 1.3e-1 |
| 1.080 | 613.9 | 698.30 | 722.96 | 3.4e-2 |

plus the case-118 group at `d/rho ≤ 0.058` with `A_point` 924–18214 and gaps
2.1e-2 – 9.7e-2. The region is therefore **`d/rho` from ~0 to ~1.1** — the
caustic inside or at the edge of the source disk — and the errors reach 20%.

**Not fixed, deliberately.** The obvious guard is to stop `prefer_polar`
steering into that region, and its second clause does exactly that on purpose
(`(A >= 100) & (distance_ratio < 0.3)`). But a `distance_ratio >= 1` guard is
**not sufficient**: four of the eight rows above sit at `d/rho` between 0.92
and 1.08 and are selected by the *first* clause, `A >= 300`. Repairing this
means refitting the polar/Cartesian switching rule — claim 3 of this report —
not adding a condition to it, and any refit changes which route the corpus
takes and so requires §5.5's timing table to be re-measured. That is a
research decision about a published result, and it is left to the reader of
this report rather than taken here.

Coverage moved with it: rows that never completed inside the per-block seconds
cap fell from 15.8%/13.4% to 4.9%/4.8% at 1e-4/1e-3. That is why the corpus
counts in the table below are larger than the pre-fix ones — about three times
as many rows became measurable in the same time budget — and it means the
post-fix medians are taken over a *harder* row set, not an easier one.

#### The corpus measurement, after the fix

Full corpus, load-calibrated, cheapest setting reaching the accuracy, against
both candidate denominators:

| profile | accuracy | n | vs oracle grid | win | vs `lcbinint_auto` | win | first call |
|---|---|---|---|---|---|---|---|
| uniform | 1e-2 | 1297 | **6.21×** | 21.3% | 5.82× | 9.2% | 0.38 s |
| uniform | 1e-3 | 1291 | **5.87×** | 22.2% | 5.80× | 9.2% | 0.38 s |
| uniform | 1e-4 | 744 | **4.83×** | 18.8% | 5.52× | 16.7% | 0.56 s |
| linear | 1e-2 | 1296 | **5.92×** | 21.8% | 5.59× | 9.3% | 0.38 s |
| linear | 1e-3 | 1284 | **6.10×** | 22.4% | 5.84× | 9.1% | 0.44 s |
| linear | 1e-4 | 849 | **5.20×** | 19.2% | 5.45× | 13.6% | 0.56 s |

Every row above is the complete corpus against a complete control pass
(160/160 cases in each of the four `ext_capfix` passes, 2880 ext rows joined to
2880 speed rows, per-row control scale median 1.32). An earlier draft of this
table quoted 6.4×/24% from a control pass that was then about half finished; the
full pass moves the medians down by ~2–5% and the win rates down by ~2 points.
Nothing in the conclusion depends on that difference.

For contrast, the same measurement **before** the fix: 43–62× against the oracle
grid, win rate 0.0–2.2%, first call 0.88–1.21 s. The fix is worth about **nine
times** and moves the win rate from ~0% to ~20%.

The two denominators now agree to within 15%, which is itself a result: before
the fix they differed by ~1.8× because the 590 ms constant swamped whatever the
native side was doing.

#### Why 5–6× and not 1×

Both backends run **the same compiled C++ Cartesian kernel** —
`binary_inverse_ray_cartesian_batch_ffi`. Nothing here is XLA-versus-C++.
Confirmed directly: on three geometries × two tolerances both backends selected
the *same method on every epoch* and agreed to 2.6e-14.

Measured kernel against kernel, one FFI call over a 24-epoch block against native
forced Cartesian on identical source positions, same nbin, routing disabled:

| nbin | native ms/ep | FFI ms/ep | FFI / native |
|---:|---:|---:|---:|
| 16 | 0.230 | 0.726 | 3.15× |
| 32 | 0.320 | 1.146 | 3.59× |
| 64 | 0.602 | 2.191 | 3.64× |
| 128 | 1.599 | 5.020 | 3.14× |

Linear limb darkening gives 3.03–3.92×. **The FFI kernel costs 3.1–3.9× the
native kernel** — exactly the figure `docs/jax-cpu-inverse-ray-mvp.md:284`
recorded at design time. It never regressed; it was simply never the quantity the
old headline measured.

So the post-fix 4.8–6.2× decomposes as ~3.1–3.9× kernel plus ~1.6× residual
wrapper overhead at 24-epoch blocks, and the residual shrinks with block length.
That closes the story: **the JAX backend's cost is now the kernel ratio plus a
block-length-dependent remainder, and it is no longer dominated by an
implementation accident.**

#### What remains, and what this does not say

*Two known inefficiencies, both measured, neither fixed.*

**Double work in the source-plane band.** In
`binary_magnification_native_pipeline_trajectory` the fused trajectory is
evaluated unconditionally for every epoch, and the source-plane chord quadrature
then *replaces* the result on epochs the band test selects
(`python/lcbinint_jax/trajectory.py:506`). Those epochs are by construction the
ones the hexadecapole did not accept, so they pay the Cartesian grid and the
chord quadrature both. Point-routed epochs are cheap for the opposite reason —
being point-safe they are hexadecapole-accepted inside the fused call and the
grid skips them — so the double work is confined to the band. That band is not
rare: across 8640 `lcbinint_auto` entries in the speed corpus,
`source_plane_quadrature` appears in **32.9%** (against 55.0% Cartesian inverse
ray, 14.4% point source, 11.8% polar, 8.8% hexadecapole; these are per-block
method sets, so it is the share of blocks using the route on at least one of
their 24 epochs). Masking the fused call the way the Cartesian FFI is already
masked would recover roughly a factor of two on that third.

**The bucket ladder arms three buckets.** Eleven of fourteen never run and are
free. When the grid *is* the selected route the ladder costs 3.0× one call,
because it arms the selected bucket and **both** neighbours; only the lower is
normally required (the upper is a fallback for when the lower is unsupported or
switches method), so about a third of that is recoverable — worth ~1.5×.

*Three things this section does not say.*

The timings are **warm**. First call is measured separately and excluded, so none
of the ratio is compilation — and adding compilation back makes JAX slower, not
faster.

The comparison is **CPU-only and single-trajectory**, which is the regime the JAX
backend is least suited to. Its reasons to exist are differentiability and
batching across many trajectories on an accelerator, and neither appears on this
axis. What the table establishes is only that the JAX path is not a drop-in
replacement for the native one when a fit needs magnifications alone.

The ratio depends on the **24-epoch block length**, because part of the remaining
cost is still per-call. Any JAX ratio from this campaign is meaningful only with
the block length stated.

*A practical limit that is not speed.* The JAX backend takes grid resolution as
`static_argnames`, so every new `(s, q, rho)` is a fresh compilation: **+546
address-space mappings, without bound**, plus 0.38–0.56 s. One 18-row block
across two profiles and three tolerances reaches ~59,000 mappings against this
machine's `vm.max_map_count` of 65,530, at which point XLA aborts the worker with
`Unable to allocate section memory`. **It presents as memory exhaustion and is
not** — it reproduced at 24 workers, at 16, and at one worker on one block, with
170 GB free. (microlux is the opposite: ~3,900 mappings once per distinct
sampling strategy, then exactly zero per row.) For a sampler that moves the
geometry every step, the binding constraint is therefore how many distinct
geometries one process may ever see, not throughput.

#### Fairness of the design-era comparison

`docs/jax-cpu-inverse-ray-mvp.md:669–677` reports the opposite sign — 9.48 ms for
a 64-epoch JAX trajectory against 135.20 ms native. Two harness choices in
`tests/diagnostics/jax_ir/benchmark_trajectory.py:309–332` account for it, both
favouring JAX: the native side was a Python loop over the **scalar**
`binary_ray_shooting`, rebuilding the lens geometry every epoch (measured here at
1.9–6.5× against the cached `LightCurve` block path — 1.497 vs 0.230 ms/ep at
nbin=16); and `OMP_NUM_THREADS` was unset, so the FFI batch took up to 32 threads
while native took one (`batch_thread_count` in `src/lcbinint/lc/light_curve.cpp:22`
returns 1 unconditionally, so *no* native path threads across epochs). The native
side there was also a forced `nbin=64` grid on all 64 epochs while the JAX side
routed 45 of them to hexadecapole. This campaign is clean on both threading
(`sweep_ext.py:304,455` pins `OMP_NUM_THREADS=1`, one worker per core) and
options; its own unfairnesses are the oracle denominator and the block length,
both stated above.

### 5.6 Triple lens

Reduced scope by design. `triple_compare/`, harness
`tests/diagnostics/recal2026/triple_compare.py`.

**VBM exposes three multi-lens methods and all three are measured.** This matters
more than it sounds: `Nopoly` is what you get when `SetMethod` is never called,
and it is the slowest of the three by 2.5–3×. **An earlier internal note claiming
lcbinint was ~2× faster than VBM on triple lenses was measured against that
default and is wrong**; it is corrected here.

| engine | tol | profile | ms/epoch | err median | err p90 | blocks |
|---|---|---|---|---|---|---|
| `lcbinint_auto` | 1e-2 | uniform | 6.161 | 1.13e-05 | 4.76e-05 | 32 |
| `vbm_singlepoly` | 1e-2 | uniform | **0.086** | 3.77e-04 | 1.35e-03 | 32 |
| `vbm_multipoly` | 1e-2 | uniform | 0.124 | 3.77e-04 | 1.37e-03 | 30 |
| `vbm_nopoly` | 1e-2 | uniform | 0.256 | 3.29e-04 | 1.37e-03 | 19 |
| `lcbinint_auto` | 1e-2 | linear | 7.284 | 6.92e-06 | 2.92e-05 | 32 |
| `vbm_singlepoly` | 1e-2 | linear | **0.101** | 1.25e-03 | 4.16e-03 | 32 |
| `lcbinint_auto` | 1e-3 | uniform | 6.163 | 1.11e-05 | 4.11e-05 | 32 |
| `vbm_singlepoly` | 1e-3 | uniform | **0.163** | 2.94e-05 | 8.45e-05 | 32 |
| `lcbinint_auto` | 1e-3 | linear | 7.284 | 6.87e-06 | 2.54e-05 | 32 |
| `vbm_singlepoly` | 1e-3 | linear | **0.311** | 3.43e-04 | 5.16e-04 | 32 |
| `lcbinint_auto` | 1e-4 | uniform | 12.072 | 5.09e-06 | 8.34e-06 | 32 |
| `vbm_singlepoly` | 1e-4 | uniform | **0.305** | 6.09e-06 | 9.38e-06 | 32 |
| `lcbinint_auto` | 1e-4 | linear | 21.368 | 4.59e-06 | 7.10e-06 | 32 |
| `vbm_singlepoly` | 1e-4 | linear | **0.766** | 4.47e-05 | 5.34e-05 | 32 |

Cost ratio, `lcbinint_auto` against the fastest VBM method on the same block:

| tolerance | profile | median | p10 | max |
|---|---|---|---|---|
| 1e-2 | uniform | 64.9× | 2.5× | 629.9× |
| 1e-2 | linear | 70.5× | 2.5× | 513.9× |
| 1e-3 | uniform | 38.1× | 1.4× | 237.4× |
| 1e-3 | linear | 29.1× | 1.3× | 177.3× |
| 1e-4 | uniform | 43.0× | 0.8× | 304.9× |
| 1e-4 | linear | **20.6×** | **0.8×** | 131.1× |

**VBM is decisively faster on triple lenses** — 20–70× at matched knob, never
worse than 0.8× even at p10. The binary crossover does not reproduce at this
corpus size.

**The limb-darkening mechanism reproduces anyway.** VBM's accuracy degrades ~10×
with a linear profile at fixed tolerance (2.94e-5 → 3.43e-4 at 1e-3; 6.09e-6 →
4.47e-5 at 1e-4) and its cost rises 1.7–2.6×; lcbinint's error *improves*
slightly (1.11e-5 → 6.87e-6) at +18–77% time. The 1e-4 linear cell has the
smallest gap (20.6×) for exactly this reason. The effect is visible; the corpus
is simply not close enough for it to change the verdict.

**lcbinint's triple `reltol` knob is inert between 1e-2 and 1e-3.**
Bit-identical error on 58 of 60 case-profiles, time within 5% on 57 of 60,
identical route distributions at all three tolerances. lcbinint over-delivers by
roughly 1000× on a 1e-2 request. **This is a concrete defect in the triple
accuracy control and is the single clearest improvement available**: honouring a
loose request on the triple path would move the 1e-2 rows by an order of
magnitude at no accuracy cost.

**VBM's triple methods are not equally robust**, and this is worth reporting
because the default is the fragile one. Over the 32-case corpus: `Singlepoly` 0
failures; `Multipoly` 2 real failures out of 32 (one SIGSEGV, one SIGABRT, the
latter reproducible only at `c = 0.5`, `RelTol = 1e-9`); `Nopoly` — *the default
when `SetMethod` is never called* — **13 of 32** case-profile-tolerance sets lost
to SIGSEGV, which is why its rows judge 19 blocks rather than 32. The reference
contours were built with `Multipoly` at `RelTol = 1e-9`, and the limb-darkened
case is expensive there: 301.99 s for one block at `c = 0.5` against 3.6 s for
the same block uniform, and 0.18 s at `RelTol = 1e-4`.

### 5.7 Where lcbinint's own time goes

Measured with `probe_diagnostics` (counters and an ablation policy, both off
unless `LCBININT_PROBE_STATS` / `LCBININT_PROBE_POLICY` are set) over a 1040-row
caustic-grazing corpus. Committed as `891ca5e`.

The heuristic probe rings that seed the image search cost **11.4× the certified
probes' root solves** (median 480 against 30) and about **85% of seeding time**.
Seeding as a share of the whole call is 22.9% on average before the change — but
that share is a strong function of resolution: 39% at nbin 8, 15% at nbin 50,
1.9% at nbin 200. **It is large only where the grid is cheap.**

Of the five ring stages, **only the interior rings were dead**, and they are
deleted: ablating them reproduced every magnification bit for bit over 911 grid
rows × nbin {16, 50, 128}, because the radii they sample are always reached by
the boundary ring or the certified ladder first. Mean seeding share falls from
22.9% to 9.8%.

The other rings are **not** redundant, and the reason is a defect rather than a
feature — §6.2. Deleting all of them would have been worth 1.13× at nbin 4 and
1.00× at nbin 200 in any case, and the timing blocks here are far denser in grid
epochs than a real light curve is. **This is the reason the ring deletion did not
require the speed campaign to be re-run.**

---

## 6. Correctness at `d/rho ≈ 1`

Found while checking the references the speed numbers rest on. Full account in
`REPORT_tangency_defects.md` (diagnosis) and `REPORT_tangency_fixes.md` (fixes).
Method: a three-witness scan (Cartesian grid, polar grid, VBM) over the tangency
band, with an independent high-order arbiter called on every position where the
three did not agree.

### 6.1 Defect A — the Cartesian row scan assumed one interval per row

The scanline flood fill stored one interval per lattice row, so a **pinched**
component — one that a row crosses twice — lost coverage. Eleven cases, all
`inverse_ray_cartesian`, all **certified**: the certificate proved support and
topology correctly and the defect was downstream of it.

Fixed by a second scanline flood fill storing a private set of merged runs per
lattice row (`ClaimedCellRuns`), discovering every vertically connected run. Both
candidates start from identical claimed-cell state and the larger inside-support
footprint is committed — which preserves the long, sub-cell fold images for which
the legacy continuation is better resolved, while letting a pinched row
contribute multiple disjoint runs.

Relative gap on the eleven-case corpus: **3.75e-3–1.62e-2 → 1.35e-5–2.19e-4.**
The former 0.37%–1.62% Cartesian deficit is gone.

The investigation also corrected its own first account: retaining the full seed
union alone repairs all eleven stored cases even through the legacy walk. The
multi-run representation is kept because it removes the structural
one-interval limitation rather than relying on redundant seeds to partition a
pinched component.

### 6.2 Defect B — both grids wrong together, and why

Nine cases where Cartesian and polar agreed **bit-identically** and both differed
from the arbiter. The shared cause was found only after arbitrating the positions
where VBM stood alone — the step that overturned the original scan verdict.

In the current tree those nine cases **do not reach either image-plane grid**.
Both explicit grid requests are intercepted by the shared grazing
`source_plane_quadrature` route, which is what made their answers bit-identical.
The equal-area midpoint rings treated a caustic just outside the limb as smooth,
and its off-centre fold spike was not resolved. **This was not a flood-fill or
certificate-boundary failure**, which is what the first three days of the
investigation assumed.

Fixed: both the chord and grazing branches now use composite order-eight
Gauss–Legendre panels at 48/96, escalating to 192 when their absolute difference
exceeds the requested budget — the arbiter's discretisation, implemented in the
native batch point-source path.

Relative gap: **7.69e-4–2.27e-3 → 7.98e-8–7.78e-5**, and `converged` went from
4/9 to 9/9.

**The cost.** Median per-grid runtime on the nine stored cases rose from about
**0.021 s to 0.400 s** — the deliberate price of removing a silent 1e-3 error.
See §8 for what this does to the speed tables.

This also explains §5.7's finding that the probe rings other than the interior
ones are not redundant: they were compensating for this.

### 6.3 Defect C — the flood fill is not seed-order independent

Two distinct mechanisms contributed:

1. Probe-image proximity deduplication replaced an earlier representative with a
   later one. At finite lattice spacing the replacement need not reach the same
   connected run, so a deeper certificate could **remove** coverage. Probe seeds
   are now retained as a **union** and deduplicated only after lattice snapping.
2. The legacy single-run walk let the first seed determine ownership at seams.
   Production seed cells are now put in a canonical order; the multi-run
   candidate follows seed parity independently of existing ownership.

The recorded seed-addition case went from a non-monotonic
`823.163656699 → 822.016016337` to identical values
`823.163656750 → 823.163656750`. On the explicit order case the legacy walk gives
`112.918051928` versus `113.624718163` when sorted by increasing versus
decreasing `|J|`; the fixed path gives `113.624712640` in both orders.

**This should still be stated as a paper limitation.** Two mechanisms were found
and fixed; seed-order independence is a structural property that has been
empirically restored on the recorded cases, not proven.

### 6.4 Status

Independent regression corpus, `tests/regression/test_tangency_correctness.py`,
reading `tangency_correctness_before.json` / `_after.json`:

| corpus | before rel. gap | after rel. gap | certified | converged |
|---|---:|---:|---:|---:|
| A, Cartesian, 11/11 | 3.75e-3–1.62e-2 | 1.35e-5–2.19e-4 | 11/11 → 11/11 | 11/11 → 11/11 |
| B, both grids, 9/9 | 7.69e-4–2.27e-3 | 7.98e-8–7.78e-5 | 9/9 → 9/9 | 4/9 → 9/9 |

VBM is accepted as the closest party only in the nine verdicts where the
independent arbiter has a decade of decision margin. All 12 rows in
`tangency_arbitration_vbm.json` were rerun; the three where the arbiter had
already found VBM pathological remain consistent with the native grids (native
gaps 2.86e-5–1.46e-4) and are deliberately excluded from the nine-row
VBM-right assertion.

Native CTest 1/1 passed; build-extension Python regressions 224 passed, 3
skipped.

**The scan itself is 115/120, not a full post-fix scan.** The five missing cases
were retried with eight workers; after more than five minutes all five remained
in computation and no complete case file was produced, so the retry was stopped.
It must not be described as complete.

### 6.5 Independent verification of the fixes

The fixes were made in one session and checked by that session's own harness, so
they were re-verified separately, with a negative control. The check
(`tangency_correctness.py --check`) recomputes all 20 stored cases against the
arbiter values and asserts the gaps; it is not a comparison of stored numbers.

Two extension builds were run through it in-process, without touching the shared
editable install: the current tree's build, and a build predating the A and B
fixes.

| | defect A, 11 cases | defect B, 18 grid results | production route |
|---|---|---|---|
| pre-fix build | 3.75e-03 – 1.62e-02 | 7.69e-04 – 2.27e-03 | unchanged |
| current build | **1.35e-05 – 2.19e-04** | **7.98e-08 – 7.78e-05** | unchanged |

Both reproduce §6.4's recorded ranges exactly. The pre-fix build **fails** the
check at the defect-A assertion, which establishes that the test has teeth
rather than passing vacuously. Routes are unchanged in both directions —
`inverse_ray_cartesian` for A, `source_plane_quadrature` for B — so the
improvement is in the quadrature, not in the case having been routed elsewhere.

For defect C both builds already report `delta = 0` on the seed-addition and
seed-order cases, so the available pre-fix build predates only the A and B work.
C's negative control is the one built into the harness: the legacy walk is
reproduced deliberately and still shows a **0.707** order dependence, against 0
on the production path.

**One caveat found during this verification and worth carrying.** The extension
served by the editable install (`_lcbinint_editable.py`, installed with
`rebuild=False`) is a **copy in site-packages**, not the tree's `build/`
directory. Editing C++ and re-running `pytest` therefore tests the *previously
installed* extension unless the build is explicitly reinstalled, and `PYTHONPATH`
does not override it — the finder is on `sys.meta_path` and wins. At the time of
this verification the installed copy was a day older than `build/`. Both pass, so
no conclusion here is affected, but any future "the fix is in and the tests pass"
should confirm which binary was loaded.

---

## 7. Writing the paper: claims, support, qualifications

What follows is the campaign's deliverable for a manuscript: each claim, the
evidence, and the qualification it may not be quoted without.

| # | Claim | Support | Must carry |
|---|---|---|---|
| 1 | The certificate decouples correctness from resolution, exposing 25–278× of over-resolution | §4.1, `nbin_rule.json` | It is a *support/topology* certificate, not an error bound (§3.1) |
| 2 | A constant nbin per tolerance covers ≥99% of a holdout | §4.1 | Except Cartesian@1e-4 (99.26%), which needs the rho-dependent rule |
| 3 | Grid choice is decided by magnification, not by rho or `d/rho`; `A_point > 200` → polar | §4.2 | The optimum is flat 100–500; the corpus supports a decade, not two digits |
| 4 | Routing boundaries are sound except hexadecapole and grazing quadrature at 1e-4 | §4.3, `route_audit.json` | The quadrature misses were a defect, since fixed (§6.2), not a loose boundary |
| 5 | VBM is faster on uniform sources by 17–31× at every tolerance | §5.1 | Forced-grid columns are oracle lower bounds (§2) |
| 6 | Limb darkening multiplies VBM's cost by 11.66× at 1e-4 and lcbinint's by 1.00× | §5.2 | This is *the* mechanism claim; everything else follows from it |
| 7 | lcbinint wins 74.9% at median 2.12× under limb-darkened ∧ 1e-4 ∧ `d/rho`<0.95 ∧ `A`>3 | §5.3 | 211 blocks, 16% of a corpus that samples `d/rho` uniformly to 30 |
| 8 | The mechanism reproduces against an independent contour code | §5.4 | microlux's knob is non-monotone; capped settings excluded |
| 9 | The engine faster on the median is the one that more often misses the requested accuracy | §5.1, §5.4 | VBM 20–24 vs 2–4; microlux 320/909 at linear 1e-4 |
| 10 | The JAX backend costs 4.8–6.2× the oracle grid (5.5–5.8× `lcbinint_auto`) and wins ~20% of blocks | §5.5 | Post-`jax.jit` **and four further fixes**; warm; CPU-only; 24-epoch blocks; ~1.6× residual wrapper cost remains |
| 10b | The JAX backend's fail-closed rate fell 9.3% → 2.6% → **0.50%** at reltol 1e-2 | §5.5, §5.5.1 | Three defects, fixed in that order: a tile capacity capping A ~ 47, then a polar route with no way back. The 0.50% column is a re-score of the failing rows, not a fresh sweep |
| 10c | The JAX backend misses its requested accuracy on 1.2–4.2% of scored blocks, worst 24.5% | §5.5.2 | Already in `ext_capfix`; 18 of 19 attributed epochs are the polar route. The certificate cannot see it — the polar grid returns the same wrong value, `support_valid=True`, from 16 to 400 bins. **Open** — fixing it means refitting claim 3 |
| 11 | The FFI kernel costs 3.1–3.9× native, matching the design-time record | §5.5 | Kernel-to-kernel, `OMP_NUM_THREADS=1`, routing disabled both sides |
| 12 | VBM is 20–70× faster on triple lenses | §5.6 | Reduced scope, 32 cases; corrects an earlier internal note measured against `Nopoly` |
| 13 | lcbinint's triple `reltol` is inert between 1e-2 and 1e-3 | §5.6 | A defect, not a result; the clearest available improvement |
| 14 | Two correctness defects at `d/rho ≈ 1`, both fixed and regression-pinned | §6 | Scan is 115/120; no individual block in `d/rho ∈ [0.85,1.05]` quotable without arbitration |
| 15 | The fixes are independently verified against a failing pre-fix build | §6.5 | Defect C's negative control is the in-harness legacy walk, not a separate build |

**Claims the data does *not* support**, listed because they are the natural
over-reaches:

* *"lcbinint is competitive with VBM."* It is not, except inside the §5.3 regime.
  On a uniform source there is no tail in which it competes (p90 ≤ 0.32).
* *"lcbinint is faster on large sources."* rho is nearly flat (§5.3). What matters
  is the source *covering* the caustic.
* *"JAX/XLA is slow at inverse-ray integration."* Both backends run identical
  machine code for the quadrature (§5.5).
* *"lcbinint's accuracy is certified."* Support and topology are certified;
  quadrature error is measured (§3).
* *"The flood fill is seed-order independent."* Two mechanisms were found and
  fixed; the property is not proven (§6.3).
* *"lcbinint beats VBM on triples by 2×."* That note was measured against
  `Nopoly` and is withdrawn (§5.6).

**The single most publishable finding**, if only one is carried: claim 6. It is a
structural difference between contour and grid methods, it is measured on two
independent contour implementations, its size is controlled by a knob the user
sets, and it predicts the regime in claim 7 rather than being fitted to it.

---

## 8. Threats to validity

* **The grid columns are oracles.** They select the cheapest of nine bin counts
  after seeing which met the target. `lcbinint_auto` is the only runtime number,
  and it carries a routing effect that must be separated before it is compared to
  a quadrature (§2).
* **The speed tables are stale in the grazing band.** The §6.2 fix raised
  per-grid cost on the affected route from ~0.021 s to ~0.400 s, and the route
  is selected by 72.9–92.8% of rows at `d/rho ∈ [0.95, 2.0]` (against 0–4.4% at
  `d/rho ≤ 0.8` and 0–12.6% at `d/rho ≥ 3`). That is roughly **37% of the corpus,
  ~1060 rows, whose lcbinint timings predate the fix.** The headline result of
  §5.3 is *not* affected — it lives at `d/rho < 0.95`, where the route is used
  0–4.4% of the time — but the `0.95–1.5` and `1.5–4` rows of §5.3's first table
  are optimistic for lcbinint and should be regenerated before publication.
  **This is the campaign's one known outstanding measurement.**
* **The references are not above suspicion.** §6 documents two defects near
  `d/rho ≈ 1`, one implicating the reference construction itself. Aggregates over
  ~1300 blocks are not moved by ~140 disputed positions, but no individual block
  in `d/rho ∈ [0.85, 1.05]` should be quoted without checking
  `REPORT_tangency_defects.md`.
* **The 1e-4 corpus is smaller.** 797–909 of 1440 blocks have references fine
  enough to judge 1e-4 (1317–1338 for the VBM axis, which uses a different
  reference path). Conclusions at that accuracy rest on 60–93% of the corpus
  depending on the axis.
* **Cross-run clock.** The external sweep (microlux, JAX) could not run at the
  speed sweep's concurrency, because each JAX worker holds its compiled
  executables. Both runs timed the same two native Cartesian buckets on the same
  geometries. The factor is **1.312**, identical at bucket 24 and bucket 50 over
  5504 paired measurements, and flat across ten deciles of block cost
  (1.28–1.34 spanning 0.06 ms to 40 ms/epoch). It is applied **per row from that
  row's own control measurement**, not as a scalar, and both raw and calibrated
  numbers are reported. Uncorrected, every external ratio would flatter the
  external engine by 31%. Verified not to be contaminated by the §6.2 fix: the
  factor is 1.265–1.336 at every `d/rho` including the tangency band, with 0 of
  1359 rows below 0.5.
* **Single machine, single build.** 64 physical cores (2 sockets × 32), 251 GB
  RAM, one worker per core, workers pinned, `OMP_NUM_THREADS=1`. Absolute
  milliseconds are not portable; **the ratios are the result.**
* **`d/rho` is a corpus label**, not a runtime quantity. It is computed from a
  caustic-distance query the runtime has already made when it picks a route, so
  rules stated in terms of it are implementable *inside* lcbinint — but they are
  not something a user chooses. §5.3 gives the user-facing restatement.
* **JAX ratios depend on block length.** Part of the remaining cost is per-call,
  so a longer block reports a smaller ratio for the same code. 24 epochs here.

---

## 9. Reproduction and file index

The source-controlled handoff includes the reports, compact aggregate and
arbitration files, figures, and reproduction scripts. The large per-block
working directories named below are generated data and are intentionally not
committed; the commands reproduce them from the corpus and rules.

### Commands

```bash
# Stage 1-2: resolution and speed corpora (long).  These two take their cores
# from the affinity mask they inherit, not from a flag, and pin one worker per
# core from the top of that mask -- so `taskset` is how they are confined.
taskset -c 40-63 python -m tests.diagnostics.recal2026.sweep_resolution \
    --output <dir> --workers 24
taskset -c 40-63 python -m tests.diagnostics.recal2026.sweep_speed \
    --output <dir> --workers 24 --repeat 3

# Stage 3: rules and audits
python -m tests.diagnostics.recal2026.speed_analysis <speed_dir> --output speed_rule.json
python -m tests.diagnostics.recal2026.grid_switch    <speed_dir> --output grid_switch_rule.json
python -m tests.diagnostics.recal2026.route_audit    <speed_dir> --output route_audit.json

# Stage 4: microlux and JAX.  Omit --engine to drive all 11 passes in turn.
# One process pass per engine setting -- see the mapping-count note in 5.5;
# a single process cannot hold them all.
python -m tests.diagnostics.recal2026.sweep_ext \
    --blocks <speed_dir> --output <ext_dir> \
    --workers 24 --cores 40-63 --repeat 3 --blocks-per-worker 2
python -m tests.diagnostics.recal2026.ext_analysis <ext_dir> <speed_dir>

# Stage 5: figures.  `--ext` is repeatable and each directory is calibrated by
# its own control pass; the first directory to supply an engine wins, so the
# newest measurement goes first.  That is what puts the post-`jax.jit` JAX
# timings and the original microlux timings in one plot:
python -m tests.diagnostics.recal2026.figures \
    --blocks speed_discovery --ext ext_capfix --ext ext_discovery \
    --grid-switch-rule grid_switch_rule.json --output figures

# triple lens, all three VBM methods, per-method subprocess isolation
python -m tests.diagnostics.recal2026.triple_compare --output triple_compare

# probe cost, ablations, and timing
bash tests/diagnostics/recal2026/probe_sweep.sh
python -m tests.diagnostics.recal2026.probe_analysis <probe_dir>

# the JAX kernel/wrapper decomposition of 5.5; minutes, one core
OMP_NUM_THREADS=1 taskset -c 57 python -m tests.diagnostics.recal2026.jax_kernel_audit \
    --repeat 3 --block-lengths 1,24,384 --out jax_kernel_audit_uniform.json

# the tangency study: three-witness scan, then arbitrate where lcbinint stood
# alone, then -- the step that overturned the scan -- where VBM stood alone.
# The arbiter is slow by design: 9.4M samples per position, 30-770 s each.
taskset -c 12-39 python -m tests.diagnostics.recal2026.tangency_scan \
    --output tangency_scan --cases 60 --workers 28
python -m tests.diagnostics.recal2026.tangency_arbiter --from-scan tangency_scan \
    --party cartesian --threshold 1e-3 --limit 11 --out tangency_arbitration.json
python -m tests.diagnostics.recal2026.tangency_arbiter --from-scan tangency_scan \
    --party contour   --threshold 1e-4 --limit 12 --out tangency_arbitration_vbm.json

# the post-fix regression corpus behind 6.4
python -m tests.diagnostics.recal2026.tangency_correctness \
    --out tangency_correctness_after.json
```

`jax_kernel_audit.py` must stay single-threaded: the FFI batch is OpenMP-parallel
over epochs and no native path threads across the epochs of one trajectory, so a
threaded run measures the harness.

**Run `sweep_ext` only when `sweep_speed` is not running.** Both are timing
measurements and would otherwise contend for memory bandwidth and shared cache.

### Files

| file | what it holds |
|---|---|
| `REPORT_master.md` | this file |
| `README.md` | corpus construction, stage-by-stage narrative, reproduction |
| `REPORT_speed.md` | the external comparison in full, with per-section detail |
| `REPORT_tangency_defects.md` | diagnosis of defects A, B, C |
| `REPORT_tangency_fixes.md` | the fixes and their regression corpus |
| `nbin_rule.json` | §4.1 — required bins, constant and rho-dependent rules |
| `speed_rule.json` | §5.1–5.3 — per-block VBM comparison |
| `grid_switch_rule.json` | §4.2 — corpus cost against threshold |
| `route_audit.json` | §4.3 — delivered error by route |
| `ext_rule.json` | §5.4 — microlux head-to-head (`ext_discovery`; its JAX rows are pre-fix) |
| `ext_rule_capfix.json` | §5.5 — the JAX head-to-head quoted in this report (`ext_capfix`, full control) |
| `jax_kernel_audit_{uniform,linear}.json` | §5.5 — kernel/wrapper decomposition |
| `tangency_scan/` | §6 — three-witness scan, 115/120 |
| `tangency_arbitration.json` | §6 — positions where lcbinint stood alone |
| `tangency_arbitration_vbm.json` | §6 — positions where VBM stood alone |
| `tangency_correctness_{before,after}.json` | §6.4 — the regression corpus |
| `triple_compare/` | §5.6 |
| `speed_discovery/` | the Stage 2 speed corpus |
| `ext_capfix/` | the post-`jax.jit` external sweep used in §5.5 |
| `ext_discovery/` | the original external sweep; JAX rows superseded |
| `figures/` | `magnification-vs-speed`, `rho-vs-speed`, `grid-switch` (pdf+png) |

`ext_jitfix/` is an abandoned partial attempt and should be ignored.

### Related documents outside this directory

| file | relation |
|---|---|
| `docs/finite-source-auto-calibration.md` | specification of the **shipping** rule; carries a status note pointing here |
| `docs/jax-cpu-inverse-ray-mvp.md` | design-time JAX record; §5.5 explains why its line 669 comparison reports the opposite sign |
| `docs/diagnostics/certified-component-resolution-20260801.md` | the certificate and the error estimator (§3) |
| `tests/regression/test_tangency_correctness.py` | pins §6.4 |
