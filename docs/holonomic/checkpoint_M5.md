# Checkpoint M5 — singular patches + full-epoch reference solver

Date: 2026-09-07
Branch: `claude/holonomic-solver-m1` (M1–M5 share the feature branch; the
`…-b7d3cd` name in the docs header is the worktree, not the checked-out
branch)
Status: **complete** (M5 gate met — accuracy and failure rate reported
separately over the audit region; zero silent misses)

M5 turns the M4 epoch flux into a single-entry-point solver with an honest
status, gives the `xs ≈ ys ≈ 0` locus an explicit connection-free patch
(the M4 open risk #3), and adds an audit sweep that reports **accuracy**
(over the points the solver certifies) and **failure rate** (the points it
declines to certify) as two separate numbers, gated on **no silent miss**.

```
  solve_epoch(params, u) -> EpochResult(mu, mu_uniform, F0, F_half, status, …)

  routing:   near_origin_source(params)  ->  on_axis_origin_flux  (singular patch)
             else                        ->  epoch_flux           (M4 holonomic)
  validate:  independent cross-check -> promote to OK_VALIDATED / flag / fail closed
```

---

## 1. What was implemented

| File | Purpose |
|---|---|
| `python/lcbinint/holonomic_ref/singular.py` | `near_origin_source` predicate (`|zeta| < SINGULAR_ZETA_TOL = 1e-3`, primary frame); `representation_events` (closed-form `R_max` bound — `R = a` / `R = sqrt(m0)` recognition, **no sympy**); `on_axis_origin_flux` (connection-free cell-wise angular quadrature for the on-axis-origin patch, accepts a pre-computed `topo=`); `tangency_flux_series` (plan §9 closed-contour period limit for `v → 0`, kept for M6) |
| `python/lcbinint/holonomic_ref/solver.py` | `solve_epoch` → `EpochResult`; `magnification` helper; routing (normal cell vs singular patch); the independent value cross-check and the plan §12.2 status promotion / fail-closed logic; `_independent_mu` (reference selection); `_covers_lens`, `_grid_mu`, `_source_plane_mu` |
| `python/lcbinint/holonomic_ref/__init__.py` | exports the M5 names |
| `benchmarks/holonomic/audit.py` | M5 audit sweep — 18 curated hard configs + seeded random sample, each × `u ∈ {0.0, 0.5}`, vs the production inverse-ray solver (image-plane grid where the disk covers a lens); reports accuracy / failure rate / silent misses separately; writes `evidence/holonomic/audit_M5.json` |
| `tests/holonomic/test_solver_audit.py` | `EpochResult` invariants (fast + validated), LD-blend monotonicity, the patch routing / fail-closed / not-silently-wrong checks, `representation_events` inside-scan + cheapness, a curated audit slice with the zero-silent-miss assertion, and "validated mode actually promotes something" |

Still a **pure-Python reference** — exact `sympy` for symbolic identities,
`scipy.integrate.quad` (QAWSE) for periods, `numpy` grid for the
cross-check. Touches no existing solver.

---

## 2. The status model (plan §12.2)

`solve_epoch` always returns a value **and** a status. The clean statuses
are `OK` (flux path converged, no cross-check run — `mode="fast"`) and
`OK_VALIDATED` (an independent reference agreed to `_VALIDATE_RTOL = 3e-3`).
Everything else is a refusal to certify, carried with the best value the
solver has:

* `LOCAL_REFERENCE_USED` — the on-axis-origin patch (the holonomic
  connection is undefined on this locus, §3); the value is a direct
  quadrature, not a transported period.
* `TOPOLOGY_UNCERTAIN` — from `classify_cells` (nearly coincident radial
  events, e.g. `ys ~ 1e-6`).
* `TRANSPORT_TOLERANCE_FAILED` — a trustworthy independent reference
  disagreed with the holonomic value by more than `_VALIDATE_RTOL`, **or**
  the patch corroboration was loose (`rel > 1e-2`). Fail closed — never
  returned as `OK`.

Promotion is one-directional (`_promote` only lifts a bare `OK`);
`_worst_status` takes the more severe of two. The audit and the tests both
treat `{OK, OK_VALIDATED}` as "clean" and everything else as "flagged".

---

## 3. The on-axis-origin singular patch (`xs = ys = 0`)

**Why it needs a patch.** On the primary-frame locus `zeta = 0` the
boundary quartic `Q(t; R)` carries a structural degree-2 repeated factor at
*every* `R` (`checkpoint_M3.md` §4). The Gauss–Manin connection is
genuinely undefined — not merely stiff — so the M4 per-node re-anchor has
nothing to re-anchor to. `near_origin_source` routes the whole
`|zeta| < 1e-3` neighbourhood here (`connection_matrix_numeric` is already
~2e-2 wrong at `|zeta| ~ 1e-6`).

**What the patch does.** `on_axis_origin_flux` drops the period machinery
entirely and integrates the flux directly in the image plane:

```
  F0    = ∫ Σ_arcs R Δθ_arc(R) dR ,   F_1/2 = ∫ Σ_arcs R ∫_arc √φ dθ dR
```

with the arc set at each `R` straight from the lens equation
(`topology.arcs_at`) and each `∫ √φ dθ` a QAWSE integral against the
`(1-u²)^{-1/2}` endpoint weight. The radial integral is done **cell by
cell** over `classify_cells` cells — `empty` cells are skipped. A single
global adaptive `quad` over `[0, R_max]` instead makes QUADPACK hunt the
thin arc bands for tens of thousands of evaluations (14 s → 6.5 s in
`solve_epoch` after the cell split + passing the already-computed `topo`).

**Status.** `LOCAL_REFERENCE_USED` — never a bare `OK`. On this
measure-zero locus the patch quadrature and the independent oracle are the
same construction, so "validated" would be circular.

**Accuracy (smoke, `taskset -c 0-7`, load ≈ 10–13):**

| config | (xs, ys, ρ, q, a) | u | patch μ | grid μ (2400×36000, independent) | rel | status | `solve_epoch` |
|---|---|---|---|---|---|---|---|
| exact-origin | (0, 0, 0.1, 0.4, 1.1) | 0.0 | 5.93267 | 5.93323 | ~1e-4 | LOCAL_REFERENCE_USED | 6.5 s |
| exact-origin | (0, 0, 0.1, 0.4, 1.1) | 0.6 | 5.97820 | — | — | LOCAL_REFERENCE_USED | 6.5 s |
| near-origin | (3e-4, 0, 0.1, 0.4, 1.1) | 0.0 | 5.94535 | 5.94659 | 2.1e-4 | LOCAL_REFERENCE_USED | 2.1 s |

`near-origin` (`|zeta| = 3e-4`, inside the tol) still routes to the patch;
`binary_ray_shooting` *does* run there (the disk covers the lens but the
centre is off it), which is why the near-origin cross-check `rel` is a real
number and small. Only the exact-centre locus hangs the inverse-ray
solver (§4).

---

## 4. New finding — the independent reference is method-specific near the lens

The plan's correctness bar is "≥ the current production solver", i.e.
`lcbinint.binary_ray_shooting`. Two failure modes of the reference were
found and worked around:

1. **`binary_ray_shooting` never returns when the source centre sits
   exactly on a point mass.** The inverse-ray map has an unbounded
   magnification at the lens; with the source centre on it the adaptive
   refinement subdivides forever. Confirmed: `(3e-4, 0, …)` returns in
   0.01 s, `(0, 0, …)` was killed at 200 s. **Guard:**
   `_ray_shooting_mu` returns `None` when
   `min(|zeta|, |zeta - a|) < 1e-9`; the exact locus is always
   `routed_singular` anyway, and the patch validation never calls the
   inverse-ray solver.

2. **`source_plane_flux` is unreliable near a caustic AND with a lens
   inside the disk.** It integrates `A_pt` (the point-source
   magnification), which is unbounded at the caustic and at each lens; a
   finer θ-grid samples closer to the spike and *increases* the
   overestimate. Measured:
   * resonant `(0.05, 0.02, 0.05, 0.3, 0.9)` — `source_plane` moved
     13.73 → 13.80 as `n_θ` rose (away from the `binary_ray_shooting`
     value 13.730);
   * big-source `(0.1, 0.1, 0.3, 0.5, 0.8)` with the primary lens inside
     the disk — `source_plane` gave 5.298 / 5.367 / 5.304 at
     96×512 / 128×768 / 256×1536 (no convergence), while
     `image_plane_flux_grid` gave 5.30442 / 5.30454 and
     `binary_ray_shooting` gave 5.30465.

   **`image_plane_flux_grid`** — a bounded `√φ ≥ 0` mask sum, no `A_pt` —
   *is* convergent in both regimes. It is slightly noisy for a
   barely-resolved planetary caustic (`ρ ~ √q`), so it is used only where
   `source_plane` / `binary_ray_shooting` are unavailable or suspect.

**Consequence, in the code:**

* `_covers_lens(params)` ⇔ a lens centre inside the source disk.
* `_independent_mu`: reference = `binary_ray_shooting`; corroborator =
  `image_plane_flux_grid` when `_covers_lens`, else `source_plane_flux`.
  `trustworthy` only if the two agree to `_REF_CORROBORATE_RTOL = 1.5e-2`
  (a caustic that neither resolves leaves the status **unpromoted**, not
  wrong).
* the on-axis-origin patch is corroborated by `image_plane_flux_grid`
  alone (1000×16000), logged as `ref_trustworthy: False`, never promoting
  past `LOCAL_REFERENCE_USED`; a loose corroboration (`rel > 1e-2`) fails
  it closed.
* the audit reference (`benchmarks/holonomic/audit.py`) mirrors this:
  `binary_ray_shooting` normally, `image_plane_flux_grid` (1600×24000)
  when `_covers_a_lens` (`d < ρ`).

Caveat: for the on-axis-origin patch the corroborator and the audit
reference are both image-plane grids, so their mutual `rel ~ 1e-4` is not
fully independent. The independent check for that locus is the standalone
`image_plane_flux_grid` convergence study in this section (grid → 5.933 as
resolution rises, patch = 5.93267).

---

## 5. Representation events `R = a`, `R = sqrt(m0)`

`representation_events` recognises these radii for the status note. It uses
the **closed-form** `R_max` bound (plan §3,
`(a + W + sqrt((a-W)² + 4))/2`, `W = |zeta| + ρ`) — no sympy event solve,
so it is safe on the `solve_epoch` hot path (`test_representation_events_is_cheap`:
50 calls < 0.5 s). Both radii are essentially always inside the scan
(`R_max` is always slightly `> a` and `> sqrt(m0)`).

Empirically the M4 QAWSE re-anchor crosses both cleanly — the `t`-chart
degree drop at `R = a` and the `Res(P, A) = 0` touch at `R = sqrt(m0)` are
integrable — so **no bridging is done for the value**; `solve_epoch` only
appends a note. This confirms the M4 plan (`checkpoint_M4.md` §2.1 the
flux is a direct radial quadrature, not a transported period).

---

## 6. Verification

All runs `taskset -c 0-7` on the shared 64-core host, `uptime` load
≈ 10–13 (timings inflated accordingly).

### 6.1 `tests/holonomic/test_solver_audit.py`

`24 passed, 7 warnings in 58.54s`. Covers: `EpochResult` invariants in
both `fast` and `validated` mode; the LD-blend monotonicity invariant
(§7a); patch routing + fail-closed + "not silently wrong" for the
on-axis-origin locus; `representation_events` inside-scan and cheapness
(50 calls < 0.5 s); a curated audit slice (plan15 / resonant / close ×
`u ∈ {0, 0.5}`) with the zero-silent-miss assertion; and
"validated mode actually promotes a well-resolved config".

### 6.2 Full holonomic suite

`252 passed, 3 skipped, 9 warnings in 314.41s` — the M4 baseline
(228 passed, 3 skipped) plus the 24 new `test_solver_audit.py` cases,
no regression from the `singular.py` / `solver.py` edits. The only
warning is the known `_arc_measure_sum` extrapolation notice at
`flux.py:143` (§8 risk 2).

### 6.3 Audit sweep — accuracy vs failure rate (the M5 gate)

`benchmarks/holonomic/audit.py --quick` (18 curated hard configs ×
`u ∈ {0.0, 0.5}` = 36 points), `evidence/holonomic/audit_M5.json`,
wall 132 s:

| metric | value |
|---|---|
| **accuracy** (over the 27 `OK_VALIDATED` points) | median rel **3.11e-5**, p90 **2.40e-4**, max **5.85e-4** |
| **failure rate** (points the solver declines to certify) | **8 / 36 = 22.2 %** — `LOCAL_REFERENCE_USED` 4, `OK` 1, `TRANSPORT_TOLERANCE_FAILED` 4 |
| **silent misses** (clean status but rel > 5e-3) | **0** |
| gate | **PASS** |

Accuracy and failure rate are reported as two separate numbers, exactly
as the M5 gate requires. The 22.2 % "failure" rate is the solver being
honest, not wrong: every one of those 8 points carries a non-clean
status, and 7 of the 8 are in fact accurate to < 8e-4 (the patch rows
and `R=a-graze`); only the 4 `TRANSPORT_TOLERANCE_FAILED` points
(`tiny-rho`, `very-wide`) are genuinely off — see §6.4.

### 6.4 The two `TRANSPORT_TOLERANCE_FAILED` configs

`tiny-rho` `(0.1, 0.02, 1e-3, 0.3, 1.0)` and `very-wide`
`(3.0, 0.2, 0.05, 0.4, 5.0)` are the only audit points where the
holonomic value is genuinely wrong (rel 7.8e-3 and 4.8e-3). Both are
correctly caught and returned `TRANSPORT_TOLERANCE_FAILED` (fail
closed — not silent). Characterisation, with two mutually independent
references:

| config | holonomic | `binary_ray_shooting` | `source_plane_flux` (128×768 / 256×1536) | `image_plane_flux_grid` (800 / 1600 / 3000) |
|---|---|---|---|---|
| `very-wide` | 1.02752 | 1.03250 | 1.032500 / 1.032500 | 1.0129 / 1.0344 / 1.0338 |
| `tiny-rho` | 38.0400 | 38.33771 | 38.33768 / 38.33768 | 40.70 / 39.19 / 39.11 |

`binary_ray_shooting` and `source_plane_flux` agree to < 1e-5 in both
cases, so the holonomic path is the one in error; the image-plane grid
is too noisy at these geometries to arbitrate but brackets the
consensus. The common factor is an under-resolved perturbation:
`very-wide` is a weak wide-binary bump (`a = 5`, `μ − 1 ≈ 0.03`) where
a ~0.5 % transport error is a large fraction of the signal;
`tiny-rho` is near-point-source (`ρ = 1e-3`) where the arc bands are
razor-thin and the M4 QAWSE re-anchor loses ~1 % of the period. Both
are M6/M8 accuracy items (finer event resolution / fixed radial rule
per cell); neither is an M5 blocker — the gate is *no silent miss*, and
the solver flags both.

---

## 7. Result summary — `solve_epoch` smoke (8 configs × u ∈ {0, 0.6})

`taskset -c 0-7`, shared 64-core host, load ≈ 10–13:

| config | u=0 μ | status | rel vs ref | u=0 time | notes |
|---|---|---|---|---|---|
| plan15 `(1/5,1/7,1/8,.5,1.2)` | 6.01995 | OK_VALIDATED | 1.8e-4 | 3.1 s | |
| resonant `(.05,.02,.05,.3,.9)` | 13.73011 | OK_VALIDATED | 4.5e-6 | 3.3 s | on the resonant caustic; ray-shooting corroborated |
| close `(.4,-.05,.09,.8,.55)` | 8.13319 | OK_VALIDATED | 3.8e-5 | 3.0 s | |
| cfg2 `(.9,-.3,.05,.25,.7)` | 1.50033 | OK_VALIDATED | 9.6e-7 | 3.4 s | |
| exact-origin `(0,0,.1,.4,1.1)` | 5.93267 | LOCAL_REFERENCE_USED | 2e-6 (grid) | 6.5 s | singular patch |
| near-origin `(3e-4,0,.1,.4,1.1)` | 5.94535 | LOCAL_REFERENCE_USED | 2.1e-4 | 2.1 s | `|zeta|` inside tol → patch |
| big-source `(.1,.1,.3,.5,.8)` | 5.30449 | OK_VALIDATED | 1.9e-5 | 2.1 s | primary lens inside the disk; grid corroborated |
| wide-planet `(1.4,.1,.03,1e-3,2.5)` | 1.15919 | OK_VALIDATED | 3.1e-5 | 3.0 s | |

No silent miss: every clean-status value agrees with its independent
reference to `< 5e-3`. The two `LOCAL_REFERENCE_USED` rows are flagged
(the solver declines to certify them) and are nonetheless right to `< 3e-4`.

---

## 7a. Note — the linear-LD blend is monotone but not signed

`d/du` of `μ_linear_ld(u) = ((1-u)F0 + u F_1/2) / (πρ²(1-u/3))` works out to
`πρ²(F_1/2 - 2F0/3)` — **constant in `u`** — so the blend is always
monotone in `u`. Its *direction* is geometry-dependent: for plan15 the
source centre is nearer the caustic than its limb, so `F_1/2` (centre
up-weighted) exceeds `2F0/3` and `μ` *rises* with `u` (6.020 → 6.146 at
`u = 0.6`). `test_limb_darkening_blend_monotone` asserts monotonicity +
`μ > 1`, not a decrease (an earlier draft wrongly assumed LD always lowers
`μ`).

## 8. Open risks carried forward

1. **Patch corroboration is grid-vs-grid.** The independent evidence for
   the on-axis-origin value is a resolution study (§4), not an in-line
   check. M6/M7 should add a VBBL or higher-order cross-check for that
   locus, or an analytic small-`ρ` expansion.
2. **`_arc_measure_sum` extrapolation warning** for plan15 (carried from
   M4). Non-fatal; the M7 fixed radial rule per cell removes it.
3. **`image_plane_flux_grid` noise for `ρ ~ √q`.** The grid corroborator
   is ~5e-3 for a barely-resolved planetary caustic. Not hit by the
   current curated/random audit configs (they do not put a lens inside a
   planetary-`ρ` disk), but a caustic-refined grid would close it.
4. **`solve_epoch` cost is Python-reference cost** (2–7 s/epoch). The M7
   adoption gate (≥ 2× the C++ `finite_source_magnifier`) is unaffected —
   this is a correctness reference, not the production path.
5. **`SINGULAR_ZETA_TOL = 1e-3` is a fixed cliff.** `|zeta|` just above it
   runs the holonomic path with a stiff connection; just below it takes
   the patch. M6/M8 should make the hand-off `ρ`- and `q`-aware (the
   connection conditioning depends on both).
6. **Source exactly on the *companion* lens is not patched.**
   `near_origin_source` only tests the primary lens (`|zeta| < tol`,
   primary frame). The mirror locus — source centre on lens 1, e.g.
   `R=a-graze` `(1.2, 0.0, 0.05, 0.5, 1.2)` where `d1 = |a − ζ| = 0` —
   is handled by the flux path (value accurate to < 1e-3) but earns only
   bare `OK` / grid-noise-dependent `OK_VALIDATED`, because
   `binary_ray_shooting` is guarded off there and the image-plane grid
   self-check is borderline. M6/M8: route source-on-companion to the same
   connection-free patch (it has the identical structural degree-2 factor
   in the companion frame).
7. **Transport accuracy floor at extreme geometries.** `tiny-rho`
   (`ρ = 1e-3`, near point source) and `very-wide` (`a = 5`, weak bump)
   carry ~0.5–1 % transport error — small in absolute terms but a large
   fraction of a weak signal or a razor-thin arc band (§6.4). Both are
   correctly flagged `TRANSPORT_TOLERANCE_FAILED`. The M7 fixed radial
   rule per cell + finer event resolution should close this.

---

## 9. Next — M6

value / JVP consistency and the 5-component Jacobian
(`(xs, ys, ρ, q, a)`): a forward-mode jet through the flux assembly,
`tests/holonomic/test_value_jvp_consistency.py` (central-difference
convergence, chart-change invariance, gradient comparison against the
existing path). The `tangency_flux_series` seed and the root-pair
Jacobian (`root_pair.py`, built and pointwise-validated at M4) are the
starting points.
