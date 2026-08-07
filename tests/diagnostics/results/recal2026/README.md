# Recalibration campaign, August 2026

Re-derivation of lcbinint's binary-lens empirical rules against the certified
algorithm, together with a speed comparison against VBMicrolensing, microlux,
and lcbinint's own JAX backend.

> **For the current Nbin calibration, start with
> [`REPORT_empirical_resolution_law.md`](REPORT_empirical_resolution_law.md).**
> It is the paper-facing result for the current certified Cartesian/polar
> algorithms, including the absolute branch and the validated mixed
> `min(N_abs, N_rel)` rule. For the wider campaign record, start with
> [`REPORT_master.md`](REPORT_master.md). It consolidates this
> file, `REPORT_speed.md`, and the two tangency reports into one narrative, and
> adds a section stating which claims the data supports and with what
> qualification. The files below remain the primary record for their own
> sections; where the master report differs, it is the later reading — in
> particular its JAX numbers supersede Stage 4's here, which were taken before
> the missing `jax.jit` was found.

## The files in this directory

| file | what it holds |
|---|---|
| **`REPORT_empirical_resolution_law.md`** | **the current paper-facing Nbin result**: common max-budget semantics, relative and absolute Cartesian/polar laws, 81-cell mixed holdout validation, figures, and limitations. |
| **`REPORT_master.md`** | **the consolidated report — read this first.** Everything below, in one narrative, plus §7: claims, evidence, and the qualification each may not be quoted without. |
| **`README.md`** (this file) | the resolution rules, the grid and route switching rules, the corpus construction, and the reproduction commands. Stages 1–5. |
| **`REPORT_speed.md`** | the speed comparison, written up: **when lcbinint is faster than VBM and what limb darkening does to the answer**, plus microlux, the JAX backend, the triple lens, and where lcbinint's own time goes. |
| **`REPORT_tangency_defects.md`** | **two new correctness defects** near `d/rho ≈ 1`, found while auditing the references this campaign rests on, and one known one restated. Read this before quoting any single-block number from either of the other two files. |

Data: `discovery/` and `holdout/` (resolution ladder), `speed_discovery/`
(binary timings), `ext_discovery/` (microlux and JAX), `triple_compare/`,
`probe/`, `tangency_scan/`, `figures/`. Fitted rules:
`nbin_rule.json`, `grid_switch_rule.json`, `speed_rule.json`, `route_audit.json`,
`ext_rule.json`, `tangency_arbitration*.json`. The JAX kernel/wrapper
decomposition of `REPORT_speed.md` §8.1–8.4 is in
`jax_kernel_audit_{uniform,linear}.json`.

The per-block sweep directories are working data and are intentionally not
part of the source-controlled handoff. The committed evidence is the reports,
the compact fitted and arbitration files, the figures, and the scripts needed
to reproduce the measurements. The abandoned `ext_jitfix/` attempt is not part
of the campaign record.

The reason for redoing work that was already done once is the component
certificate. It proves disk support and topology, which is what the old
resolution rule was implicitly buying with bins. It does not prove that the
quadrature error meets `reltol` — that remains the grid's job — so the question
this campaign asks is how much of the old bin count was paying for the part the
certificate now covers.

The answer, stated once here and supported below, is: most of it. The historical
shipping rule spends between 25 and 280 times the work the corpus actually
requires, and a constant bin count per tolerance covers 99.3–100% of the
corpus.

**Historical campaign status.** Stages 1–5 are complete for the binary lens: all eleven Stage 4
passes have run over the full corpus, and the figures are in `figures/`. The
reduced triple-lens scope has since been run as well — 32 cases, all three VBM
multi-lens methods, both profiles — and is reported in `REPORT_speed.md` §9.
The tables below are the historical offline ladder/oracle measurements; they
are not, by themselves, the current automatic settings.

**Current native runtime policy (`final-testing`).** The selector now uses the
measured point-source magnification only for the grid switch, with
`A_point >= 200` selecting polar and lower values selecting Cartesian. For the
relative targets used by this campaign, the one-shot buckets are:

| target | Cartesian | polar |
|---|---:|---:|
| `>= 1e-2` | 16 | 50 |
| default or `1e-3 <= reltol < 1e-2` | 50 | 100 |
| `0 < reltol < 1e-3` | 200 | 200 |

The automatic path performs one selected grid evaluation. Cartesian uses its
cheap area indicator as a fail-closed retry trigger; explicit fixed-grid calls
retain the half-resolution consistency check. The tangency fixes route the
grazing source-plane cases through chord quadrature before this inverse-ray
 grid switch, so the old report's proposed second polar clause is not part of
the current selector. This policy has been checked on the fresh 120-row
seed/certificate probe and the native regression suite; the historical 2880-row
campaign files have not been overwritten by this branch.

## Corpus and method

160 lens cases, seed 20260803, sampled over `s ∈ [0.6, 3.4]`, `q ∈ [1e-5, 0.3]`,
`rho ∈ [3e-5, 0.5]`, each visited at 8 caustic distances expressed in source
radii (`d/rho` from 0.8 to 30, plus far-field). Every case is run under both a
uniform and a linearly limb-darkened profile (`c = 0.5`), giving 2880 rows.

Three conventions govern every number below.

**Timing is per block, not per epoch call.** A block is 24 epochs spanning 0.4
source radii, timed three times, and the median is reported. Single-epoch
timings on this problem are dominated by cache state.

**Cost is read at delivered accuracy, never at a requested tolerance.** For each
block and engine, the cheapest setting that actually met the accuracy is used. A
knob is a request; the error column is what happened. This makes the grid curves
an *oracle lower bound* — they select the cheapest of nine bin counts after
seeing which met the target, which no caller can do at runtime — and that has to
be remembered whenever a forced grid is compared against a routed pipeline.

**Blocks whose reference floor is coarser than the accuracy under discussion are
excluded and counted.** Asking whether an engine reached 1e-4 on a block whose
converged reference is only good to 1e-3 is a question about the reference. This
is why the 1e-4 columns judge 797–909 blocks rather than all 1440.

## Stage 1 — the nbin rule

Bins actually required for the corpus, against what the shipping rule spends:

| grid | tolerance | required median | required p99 | shipping median | work vs required |
|---|---|---|---|---|---|
| Cartesian | 1e-2 | 4 | 14 | 64 | 178× |
| Cartesian | 1e-3 | 12 | 50 | 64 | 39× |
| Cartesian | 1e-4 | 32 | 128 | 400 | 278× |
| polar | 1e-2 | 6 | 24 | 64 | 156× |
| polar | 1e-3 | 16 | 100 | 64 | 25× |
| polar | 1e-4 | 32 | 320 | 400 | 156× |

A constant bin count per tolerance, validated on the historical independent holdout:

| grid | tolerance | constant bins | holdout coverage |
|---|---|---|---|
| Cartesian | 1e-2 | 16 | 99.58% |
| Cartesian | 1e-3 | 50 | 99.81% |
| Cartesian | 1e-4 | 128 | 99.26% |
| polar | 1e-2 | 24 | 100.00% |
| polar | 1e-3 | 100 | 99.77% |
| polar | 1e-4 | 320 | 99.94% |

Coverage is against a 99% target, so every cell clears it except Cartesian at
1e-4, which is the one place a constant is not enough on its own. The
`linear` rule column in `nbin_rule.json` gives a rho-dependent alternative that
reaches 99.83% there at 160 median bins.

Two things are worth stating plainly. Cartesian needs fewer bins than polar at
every tolerance — but bins are not seconds, and Stage 3 decides the grid on
measured time, not on this table. And the required counts are heavily
right-skewed: the p99 is three to ten times the median, so a rule tuned to the
median fails a percent of the corpus badly, which is why coverage is the
selection criterion rather than mean error.

## Stage 2 — against VBMicrolensing

Median and p90 cost per epoch, over blocks that can judge the accuracy:

| profile | accuracy | lcbinint Cartesian | VBM | VBM missed | ours missed |
|---|---|---|---|---|---|
| uniform | 1e-2 | 1.22 ms | **0.038 ms** | 3 | 0 |
| uniform | 1e-3 | 1.25 ms | **0.041 ms** | 3 | 4 |
| uniform | 1e-4 | 2.44 ms | **0.062 ms** | 23 | 11 |
| linear | 1e-2 | 1.24 ms | **0.041 ms** | 3 | 0 |
| linear | 1e-3 | 1.26 ms | **0.168 ms** | 3 | 4 |
| linear | 1e-4 | 2.28 ms | **0.863 ms** | 21 | 13 |

On the median, VBM is faster everywhere. The interesting structure is in how
that margin collapses: 32× on uniform sources at 1e-2, and 2.6× on limb-darkened
sources at 1e-4. The p90 tells the same story more sharply — at linear 1e-4 it is
10.3 ms for VBM against 13.3 ms for us, essentially even.

Resolved per block rather than per median, lcbinint wins where the source is
limb-darkened, the accuracy is tight, and the source is near or on the caustic:

| condition | win rate | median VBM/ours |
|---|---|---|
| linear, 1e-4, `d/rho < 0.95` | **63.5%** | **1.71** |
| linear, 1e-4, overall | 22.6% | — |
| linear, 1e-3, `d/rho < 0.8` | 46.3% | 0.95 |
| uniform, any accuracy | 0.0–0.5% | 0.04–0.06 |

Reliability runs the other way from speed. At 1e-4, VBM fails to reach the target
on 21–23 blocks where lcbinint succeeds; the reverse happens on 0–2.

**A comparison that must not be made.** `lcbinint_auto` "wins" 50–81% with a
median of 12–14× in the `A ∈ [1, 1.13]` bucket. That is the point-source route
answering, not a grid. It is a pipeline-level result and is reported separately
from the grid-level one; putting it on the same axis as an inverse-ray timing
would be comparing a routing decision against a quadrature.

## Stage 3 — grid switch and route boundaries

### Which grid

The median polar/Cartesian ratio sits on 1.00 in every cell and answers nothing.
The question is a corpus one: total time against an oracle that picks the better
grid per block. Always-Cartesian costs 1.29–1.45× the oracle; always-polar costs
1.51–2.72×.

Splits on rho, `d/rho`, and mass ratio produce no ordered structure. The split on
magnification lands immediately — the entire cost of always-Cartesian sits in the
top magnification quartile, with the other three within 4% of the oracle.
Re-derived on the *point-source* magnification, which the multipole stage has
already computed when the decision is made:

**Rule: `A_point >= 200` → polar, else Cartesian.**

This rule was derived on time alone. The later tangency fixes now handle the
grazing source-plane cases before inverse-ray grid selection, so this branch
does not add a second `d/rho` clause to the polar switch.

| | always-Cartesian | `A_point >= 200` | share sent polar |
|---|---|---|---|
| uniform 1e-2 | 1.290× | **1.096×** | 10.4% |
| uniform 1e-3 | 1.318× | **1.125×** | 10.1% |
| uniform 1e-4 | 1.380× | **1.256×** | 14.6% |
| linear 1e-2 | 1.303× | **1.068×** | 10.2% |
| linear 1e-3 | 1.369× | **1.171×** | 10.0% |
| linear 1e-4 | 1.445× | **1.223×** | 12.8% |

200 is the joint optimum in all six cells, and the optimum is flat from 100 to
500 (see `figures/grid-switch.pdf`), so the corpus supports the decade and not
two significant figures. Adding a rho condition does not improve it. Using the
true finite-source magnification instead of the point-source one only reaches
1.05–1.19×, so most of the remaining gap to the oracle is block-level noise
rather than predictable structure.

### Which route

`lcbinint_auto` records the methods it used, which turns the routing thresholds
into a measurement. Grouping delivered error by route (`route_audit.json`):

* **`point_source`: 0% miss at every accuracy and both profiles** (worst error
  0.00e+00 at 1e-4). The boundary is sound.
* **Every pure inverse-ray route: 0% miss.**
* **`hexadecapole` at 1e-4: 7.6–8.8% miss**, worst 1.53e-4 (1.5× target).
* **`inverse_ray_polar + source_plane_quadrature` at 1e-4: 8.0–11.1% miss**,
  worst 2.62e-4 (2.6× target).

The two failing routes fail at opposite ends of the magnification axis, which is
why a single predicate shape misreports one of them. Taking instead the tight
bounding box of the misses in `(rho, A)` — both quantities available before the
route is chosen:

| route | box | misses caught | legitimate hits also rejected |
|---|---|---|---|
| hexadecapole @1e-4 | `rho ∈ [0.015, 0.272]`, `A ∈ [1.01, 2.36]` | 16/16 | 63 (31.5%) |
| quadrature @1e-4 | `rho ∈ [3.8e-5, 0.020]`, `A ∈ [39.8, 8770]` | 6/6 | 157 (22.3%) |

Neither is worth imposing. The hexadecapole box costs 63 blocks that would go
from 0.14 ms/epoch to roughly 80 ms/epoch, to remove errors that exceed the
target by at most 1.5×. The recommendation is to report the trade rather than
tighten the cut: the real fix for the hexadecapole is to scale its acceptance
test with rho, which is a code change needing its own study.

The quadrature misses are more interesting than their count. All six sit at
`A ≥ 39.8` with `d/rho ∈ [0.95, 1.7]` — the tangency regime, where the source
limb grazes a fold. This corroborates the previously flagged `d/rho = 1.001`
disagreement with VBM and is the subject of a separate focused study.

**That study is `REPORT_tangency_defects.md`, and it reclassifies these six.**
They are not a route boundary set too loose: in that band the Cartesian grid and
the reference built from it are both unreliable, and so — for a different reason
— is the polar grid. The route audit's verdict on the other routes is unaffected.

## Stage 4 — microlux and the JAX backend

All eleven passes have run over the full corpus.

Getting this stage to run at all required a structural change, and the cause is
worth recording because it presents as something it is not.

The lcbinint JAX backend takes its grid resolution as `static_argnames` to the
`jax.jit` in `python/lcbinint_jax/trajectory.py`, and `nbin="auto"` derives those
from the geometry. Every new `(s, q, rho)` is therefore a fresh compilation:
**+546 address-space mappings per row, without bound.** One 18-row block across
two profiles and three tolerances reaches roughly 59,000 mappings against this
machine's `vm.max_map_count` of 65,530, at which point XLA aborts the worker with
`Unable to allocate section memory`. It looks like memory exhaustion and is not —
it reproduced at 24 workers, at 16, and at one worker on one block, with 170 GB
free. microlux is the opposite: ~3,900 mappings once per distinct sampling
strategy, then exactly zero per row.

The sweep is therefore split into one process pass per engine setting — seven
microlux settings, three JAX tolerances, and one control pass, eleven in all —
which holds a worker under 10,000 mappings. Censoring, which previously
compared against the looser setting earlier in the same row, now reads the
previous pass's output (`--censor-from`).

### The two runs did not share a clock

Splitting the sweep meant it could not run at the Stage 2 sweep's concurrency, so
its seconds are not the stored sweep's seconds. Both runs therefore timed the
same two native Cartesian buckets on the same geometries, and the ratio between
them is the correction.

It is **1.312** — the Stage 2 sweep's stored native seconds are 1.31× what the
same work costs in this run — and it is the same 1.312 at both bucket 24 and
bucket 50, over 5504 paired measurements. It is also flat in block cost:
1.28–1.34 across ten deciles spanning 0.06 ms to 40 ms per epoch. A single scalar
is therefore well supported, but the correction is applied per row from that
row's own control measurement rather than as a scalar, and both the raw and
calibrated numbers are reported so the size of the correction stays visible.

Every ratio below is the calibrated one. Uncorrected, they would flatter both
external engines by 31%.

### microlux

Cost is the cheapest setting that reached the accuracy, against the cheaper of
lcbinint's two grids on the same block and the same accuracy:

| profile | accuracy | judgeable | reached | missed | win rate | median microlux/ours | p10 – p90 |
|---|---|---|---|---|---|---|---|
| uniform | 1e-2 | 1350 | 1334 | 16 (1.2%) | 63.1% | 0.749 | 0.105 – 1.802 |
| uniform | 1e-3 | 1349 | 1332 | 17 (1.3%) | 64.5% | 0.706 | 0.102 – 1.662 |
| uniform | 1e-4 | 797 | 769 | 28 (3.5%) | 79.2% | 0.456 | 0.084 – 1.178 |
| linear | 1e-2 | 1347 | 1319 | 28 (2.1%) | 28.4% | **3.076** | 0.573 – 12.856 |
| linear | 1e-3 | 1346 | 1247 | 99 (7.4%) | 28.9% | **3.108** | 0.569 – 12.696 |
| linear | 1e-4 | 909 | 589 | **320 (35.2%)** | 37.8% | **1.159** | 0.565 – 5.807 |

("reached" counts blocks where microlux met the accuracy; the ratio columns are
over the subset where lcbinint also had a qualifying grid, which is one block
fewer in the two 1e-4 rows and identical elsewhere.)

The limb-darkening asymmetry seen against VBM reproduces through an independent
contour implementation, and more strongly. On uniform sources microlux is ahead
by 1.3–2.2× on the median and its margin *widens* as the accuracy tightens. On
limb-darkened sources the sign flips at every accuracy: lcbinint is ahead by a
median factor of 3.1 at 1e-2 and 1e-3, and the p90 says nearly 13× on the tail.

The 1e-4 limb-darkened cell is the weakest of the three — 37.8% and a median of
1.159 — and read with the miss column that understates lcbinint rather than
overstating it. microlux fails to reach 1e-4 on **320 of the 909 judgeable
blocks**, and the ratio is computed over the 589 where it succeeded. Of the 320:
114 exhausted the adaptive sampler's budget (`No enough space to insert new
samplings`), 89 were censored for already exceeding 0.25 s/epoch at a looser
setting, and the remainder simply delivered an error above target at every one of
the seven settings. The blocks microlux drops
are the hard ones, so the surviving median is conditioned on its own success and
understates the gap rather than measuring it.

Two separate effects shrink that row and they should not be conflated: the
judgeable count falls from 1346 to 909 because of the reference floor (a question
about the reference, applied identically to every engine), and 909 falls to 589
because of microlux. Only the first applies to the uniform rows, where the miss
rate stays at 1.2–3.5%.

The reliability story therefore runs the same direction as it did against VBM.
The engine that is faster on the median is the engine that more often fails to
deliver the requested accuracy, and on limb-darkened sources at 1e-4 that failure
rate is a third of the corpus.

### The JAX backend

The JAX backend carries the same routing *criteria* as the native path — the
fused trajectory computes the hexadecapole first and passes an `active` mask into
the Cartesian FFI, so hexadecapole-accepted epochs genuinely skip the grid, and
the batched entry point adds native point-source and caustic-band decisions on
top. The criteria are not the problem.

The cost model is. In `binary_magnification_native_pipeline_trajectory`, the fused
trajectory is evaluated unconditionally for every epoch, and the source-plane
chord quadrature then *replaces* the result on epochs the band test selects
(`python/lcbinint_jax/trajectory.py:506`). Those epochs are by construction the
ones the hexadecapole did not accept, so they pay the Cartesian grid and the
chord quadrature both. Point-routed epochs are cheap for the opposite reason —
being point-safe, they are hexadecapole-accepted inside the fused call and the
grid skips them — so the double work is confined to the source-plane band.

That band is not rare. Across the 8640 `lcbinint_auto` entries in the speed
corpus, `source_plane_quadrature` appears in 32.9% (against 55.0% for Cartesian
inverse ray, 14.4% point source, 11.8% polar, 8.8% hexadecapole); these are
per-block method sets, so the figure is the share of blocks that use the route on
at least one of their 24 epochs, not the share of epochs.

The JAX timings are therefore an upper bound on that third of the corpus,
and the JAX-versus-native comparison is a measurement of the current
implementation rather than of the algorithm. Restructuring the batched entry
point to mask the fused call the way the Cartesian FFI is already masked is a
code change, out of scope here, and it would invalidate this measurement — so it
is recorded rather than made.

With that stated, the result is not close enough for the caveat to decide it:

| profile | accuracy | blocks | JAX win rate | median JAX/native | p10 – p90 | median first call |
|---|---|---|---|---|---|---|
| uniform | 1e-2 | 1224 | 0.1% | 61.9× | 9.7 – 237 | 0.96 s |
| uniform | 1e-3 | 1220 | 0.2% | 55.5× | 9.0 – 235 | 0.96 s |
| uniform | 1e-4 | 690 | 2.2% | 44.9× | 8.2 – 249 | 1.21 s |
| linear | 1e-2 | 1220 | 0.0% | 59.2× | 9.8 – 240 | 0.88 s |
| linear | 1e-3 | 1211 | 0.0% | 57.0× | 9.9 – 240 | 0.92 s |
| linear | 1e-4 | 790 | 1.6% | 43.1× | 8.9 – 251 | 1.14 s |

The JAX backend on CPU costs 43–62× the native path per epoch and wins on 0.0–2.2%
of blocks. The double work described above is worth a factor of two on a third of
the corpus and cannot account for a factor of fifty.

**This table has since been superseded.** The missing `jax.jit` identified below
was applied, along with four further fixes, and the full corpus re-measured
(`ext_capfix/`): the backend costs **4.8–6.2×** the oracle grid and **5.5–5.8×**
`lcbinint_auto`, and wins **~20%** of blocks. Its fail-closed rate also fell from
9.3% to 2.6% at reltol 1e-2, because one of those four fixes removed a tile
capacity that had capped the reachable magnification near A ~ 47 regardless of
the tolerance requested. A later fix — the polar route had no way back to the
Cartesian ladder — took that rate to **0.50%**; see `REPORT_master.md` §5.5.1,
and §5.5.2 for the fail-*open* that the removed NaNs exposed, which is
unresolved. The two corrections that
follow are the diagnosis that produced the jit fix, and the first of them applies
just as much to the post-fix numbers: quote 5.5–5.8×, not 4.8–6.2×.

**Two corrections apply to this table, both in `REPORT_speed.md` §8. Read them
before quoting it.**

*The denominator is the wrong one.* This table divides by the oracle over 18
forced-grid settings, while the thing being timed is a routed pipeline whose
native counterpart is `lcbinint_auto`. Against `lcbinint_auto` the median is
**23–32×**, not 43–62×. The sign and the win rates are unchanged; the factor is
inflated by about 1.8.

*It is not a statement about JAX.* Both backends run the same compiled C++
Cartesian kernel, and both select the same method on every epoch. Measured
kernel against kernel that kernel costs **3.1–3.9×** native — the figure
`docs/jax-cpu-inverse-ray-mvp.md:284` recorded at design time. The remainder is a
**~590 ms per-call constant** in
`binary_magnification_native_pipeline_trajectory`, which carries no `jax.jit`
and is therefore re-traced and dispatched primitive by primitive on every call.
Wrapping it in `jax.jit` is 21–386× faster with bit-identical results. Over the
24-epoch blocks used here that constant *is* the 23–32×; at 1536 epochs the same
code reports 16.5×. Reproduce with
`tests/diagnostics/recal2026/jax_kernel_audit.py`.

Two things this does *not* say. The timings are warm — the first call is measured
separately and excluded (the `first call` column), so none of the ratio is
compilation, and adding compilation back makes JAX slower rather than faster. And the comparison is CPU-only and single-trajectory, which is the
regime the JAX backend is least suited to: its reasons to exist are
differentiability and batching across many trajectories on an accelerator, and
neither appears on this axis. What the table establishes is that the JAX path is
not a drop-in replacement for the native one when a fit only needs magnifications.

The first-call column is the second finding. Just under a second of compilation
per distinct `(s, q, rho)` is negligible for a light curve of thousands of epochs
and dominant for a sampler that moves the geometry every step — which is the usual
case in a fit. Combined with the +546 mappings per compilation, the practical
limit is not speed but how many distinct geometries one process may ever see.

## Reproduction

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
python -m tests.diagnostics.recal2026.sweep_ext \
    --blocks <speed_dir> --output ext_discovery \
    --workers 24 --cores 40-63 --repeat 3 --blocks-per-worker 2
python -m tests.diagnostics.recal2026.ext_analysis ext_discovery <speed_dir>

# Stage 4, post-fix JAX re-measurement (ext_capfix): the three JAX tolerances,
# each censoring the next, plus the control pass that calibrates cross-run load.
# microlux is deliberately not re-run -- none of the five fixes touch it, and it
# anchors through ext_discovery against the same speed_discovery Pareto plane.
previous=""
for setting in 0 1 2; do
  censor=(); [ -n "$previous" ] && censor=(--censor-from "$previous")
  PYTHONPATH=. OMP_NUM_THREADS=1 python -m tests.diagnostics.recal2026.sweep_ext \
      --blocks <speed_dir> --output ext_capfix/lcbinint_jax-$setting \
      --engine lcbinint_jax --setting $setting "${censor[@]}" \
      --workers 8 --cores 0-23 --repeat 3 --blocks-per-worker 2 --seconds-cap 0.25
  previous=ext_capfix/lcbinint_jax-$setting
done
PYTHONPATH=. OMP_NUM_THREADS=1 python -m tests.diagnostics.recal2026.sweep_ext \
    --blocks <speed_dir> --output ext_capfix/control-0 \
    --engine control --setting 0 \
    --workers 8 --cores 0-23 --repeat 3 --blocks-per-worker 2 --seconds-cap 0.25

# Stage 5: figures
python -m tests.diagnostics.recal2026.figures \
    --blocks <speed_dir> --ext ext_discovery \
    --grid-switch-rule grid_switch_rule.json --output figures
```

Run `sweep_ext` only when `sweep_speed` is not running. Both are timing
measurements and would otherwise contend for memory bandwidth and shared cache.

## Known limitations

* **The flood fill is not seed-order independent.** The claimed-cell registry
  depends on seed order and seed count. Probe rings currently mask a latent fill
  defect. This is a correctness item, not a performance one, and is a candidate
  paper limitation. `REPORT_tangency_defects.md` §5.
* **Two further correctness defects live at `d/rho ≈ 1`**, and one of them
  implicates the references this campaign's accuracy columns are read against.
  Aggregates over ~1300 blocks are unaffected; individual blocks in
  `d/rho ∈ [0.85, 1.05]` should not be quoted without arbitration.
  `REPORT_tangency_defects.md` §3 and §4.
* **Cross-run timing.** Stage 4 cannot run at Stage 2's concurrency, because each
  JAX worker holds its compiled executables. Both runs time the same two native
  Cartesian buckets so the scale factor between them is measured rather than
  assumed; `ext_analysis` reports it and does not silently apply it.
* **The 1e-4 corpus is smaller.** 797–909 of 1440 blocks have references fine
  enough to judge 1e-4. Conclusions at that accuracy rest on roughly 60% of the
  corpus.
* **`d/rho` is a corpus label, not a runtime quantity.** Rules stated in terms of
  it describe where an effect lives; they are not directly implementable as
  written.
* **The JAX ratios measure a fixable wrapper, not the method**, and they depend
  on the 24-epoch block length: the pipeline's dominant cost is a per-call
  constant, so a longer block reports a smaller ratio for the same code. That
  re-run has since happened (`ext_capfix/`, `REPORT_master.md` §5.5) and moved
  the axis from 43–62× to 4.8–6.2×; the block-length dependence survives the fix,
  because ~1.6× of the residual is still per-call. `REPORT_speed.md` §8.1–8.4.
* **This is a shared, multi-user machine, and load has to be measured, not
  assumed.** The `ext_capfix` passes ran on cores 0-23 while other users' jobs
  held affinity `0-63`; a contention logger recorded a mean of 3.83 competing
  processes on those cores, maximum 6. The control pass is what bounds the
  damage: its per-row scale moved from 1.312 to 1.323 between `ext_discovery` and
  `ext_capfix`, i.e. **0.8%**, against effects of 3–30×. Any future timing pass
  here should log affinity and load *before* it starts, not diagnose them
  afterwards — a loaded control pass biases ratios in the JAX backend's favour.
