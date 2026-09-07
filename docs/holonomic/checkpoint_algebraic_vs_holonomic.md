# Checkpoint — 3-way comparison: algebraic-boundary vs holonomic M7 vs inverse-ray

**Date:** 2026-09-08
**Branch:** `claude/holonomic-solver-m1` (holonomic), benchmarked against
`algebraic-boundary-ld-moment-recurrence` @ `cc5e55d` (algebraic, isolated build).
**Status:** comparison complete. Framing rule in force — the holonomic speedup is
stated **only as "vs inverse-ray baseline"**, never as a final verdict, until the
M8 deep-solve work lands.

---

## 0. What was compared, and how

| backend | value path | 5-Jacobian path | source of numbers |
|---|---|---|---|
| **(I) inverse-ray** (incumbent M0) | `binary_ray_shooting` | central FD ×11 | `evidence/holonomic/baseline_M0.json` |
| **(A) algebraic-boundary** | `experimental_algebraic_boundary_binary_mag` | `experimental_algebraic_boundary_binary_jacobian` — **analytic** forward-mode `ForwardJet<5>` | this run (isolated `cc5e55d` libs) |
| **(H) holonomic M7** | — (no separate value path) | `lcbinint::holonomic::epoch_jacobian` — **analytic**, value+Jacobian **fused** | this run (`src/lcbinint/magnification/holonomic/*.hpp`) |

* **Same 108 (config, u) points** — `/tmp/bench_cases.tsv`, 54 configs × {u=0, u=0.5}.
* **Same frame** — Mapping B (centre-of-mass frame): from a holonomic-frame
  `(xs, ys, ρ, q, a)` the algebraic / inverse-ray call uses
  `separation = a`, `mass_ratio = q`, `source = (xs − q/(1+q)·a, ys)`, `source_radius = ρ`.
  Empirically validated (plan15 u0: A 6.019954, H 6.020410, I 6.018865 — agree to ~2e-4).
* **Same linear-LD convention** — `limb_darkening_c = u`, `limb_darkening_d = 0`
  (`brightness = 1 − c(1−μ)`; holonomic `1 − u(1−√(1−(r/ρ)²))`).
* **Same trusted reference for value accuracy** — `binary_ray_shooting` μ from
  `baseline_M0.json` (the incumbent's own value output), cross-checked A↔H.
* **Same fail-closed policy expectation** — a backend must either return a
  trustworthy answer or report failure; a confidently-wrong answer or a
  non-terminating call is a policy violation.
* **Timing** — one process, `taskset -c 0-7`, best-of-150 wall-clock
  (`steady_clock`), warm cache. Machine load average 10–15 throughout
  (`uptime` logged before/after each run; box is oversubscribed, so best-of is
  the uncontended-execution estimate). Isolated algebraic libs built with the
  **same** aggressive flags as holonomic M7
  (`-O3 -march=native -funroll-loops -ffp-contract=fast -fno-math-errno`).
* The shared `site-packages/_lcbinint.so` was **not** touched — the algebraic
  side is a detached worktree at `cc5e55d` with a private static-lib build
  (`.claude/worktrees/algebraic-bench-cc5e55d/build-bench/`).

Harness: `tests/holonomic_cpp/bench_three_way.cpp` (+ `bench_algebraic_one.cpp`
for the per-case `timeout` sweep that isolates the algebraic non-termination).

---

## 1. Headline — this is not a single "faster / slower"

The local algebraic-boundary backend is **bimodal and routing-dependent**. It
keeps `binary_mag`'s shared routing and only swaps the terminal solve, so:

* **74 / 108 configs → multipole shortcut.** `binary_mag` routing decides the
  point-source / hexadecapole stencil is within tolerance and **no boundary
  integral is run** (`diagnostics.multipole_shortcut = 1`, zero root solves).
  value ≈ 0.05–0.25 ms, analytic Jacobian ≈ 0.013 ms (a closed-form multipole
  stencil, *cheaper than the value*).
* **32 / 108 configs → deep algebraic-boundary solve.** Full polar
  boundary-integral: for plan15 u0, ~2060 quartic root solves, ~10 k lens-map
  evaluations over 2 radial bands / ~48 subdivisions, (m,v)-transport 1640×
  (420 fallbacks). value p50 1.1 ms, value+analytic-Jac p50 1.3 ms
  (forward-mode adds ~2.5–3×).
* **2 / 108 → non-terminating** (see §4).

Holonomic M7 has **one** path and always runs the full finite-source radial
reconstruction (value+Jacobian fused): 3.7–8.5 ms regardless of routing.

So a raw 108-point median comparison ("algebraic 0.27 ms vs holonomic 4.78 ms")
compares *holonomic's full solve* against *algebraic's approximation on the
easy 2/3 of the set*. The meaningful comparison is the **deep-solve regime**
(§3), where both backends actually do finite-source work.

---

## 2. Latency — full 108 (best-of-150, taskset -c 0-7, load ~12)

### value + 5-component Jacobian (ms)

| slice | backend | n | p50 | p90 | p95 | p99 | max |
|---|---|--:|--:|--:|--:|--:|--:|
| ALL 108 | algebraic (analytic) | 106 | **0.163** | 3.08 | 5.75 | 9.41 | 12.47 |
| | holonomic (fused) | 108 | 4.78 | 8.12 | 8.30 | 8.44 | 8.47 |
| | inverse-ray (FD ×11) | 108 | 14.89 | 145.3 | 4392 | 4628 | 4643 |
| ordinary binary (q > 1.5e-3) | algebraic | 46 | **0.27** | 6.39 | 8.89 | 11.10 | 12.47 |
| | holonomic | 46 | 4.56 | 8.27 | 8.37 | 8.46 | 8.47 |
| | inverse-ray | 46 | 15.11 | 58.6 | 63.4 | 69.9 | 71.4 |
| planetary low-q (0 < q ≤ 1.5e-3) | algebraic | 60 | **0.05** | 1.08 | 1.25 | 2.00 | 2.27 |
| | holonomic | 62 | 4.85 | 7.17 | 8.19 | 8.33 | 8.36 |
| | inverse-ray | 62 | 13.62 | 4365 | 4419 | 4643 | 4643 |
| uniform u=0 | algebraic | 53 | 0.15 | 2.48 | 4.10 | 5.18 | 5.54 |
| | holonomic | 54 | 4.78 | 8.10 | 8.31 | 8.40 | 8.45 |
| linear LD u>0 | algebraic | 53 | 0.18 | 5.11 | 8.69 | 10.89 | 12.47 |
| | holonomic | 54 | 4.78 | 8.09 | 8.27 | 8.42 | 8.47 |

### value-only (ms)

| slice | algebraic | holonomic | inverse-ray |
|---|--:|--:|--:|
| ALL 108 p50 | 0.17 | *n/a — fused* (≈ value+Jac) | 1.91 |
| ALL 108 p95 | 1.78 | *n/a* | 399 |
| ordinary binary p50 | 0.18 | *n/a* | 1.95 |
| planetary p95 | 1.24 | *n/a* | 402 |

**Holonomic has no cheaper value-only mode** — the 5-Jacobian is accumulated in
the same per-cell Gauss–Chebyshev radial pass as F0/F_half, so its value cost
*is* its value+Jacobian cost. For algebraic and inverse-ray the Jacobian is a
multiplier over the value (analytic forward-mode ≈ 2.5–3×; central FD ×11).

### LD split, deep-solve only

Linear LD roughly **doubles** the algebraic deep-solve Jacobian cost
(plan15: u0 3.5 ms → u0.5 6.9 ms) — the `ld_moment_series` recurrence adds
~1300 evaluations. Holonomic linear-LD cost is flat (`(1−u)F0 + uF_half` reuses
the same two accumulators; +0 ms).

---

## 3. Latency — deep-solve regime (the honest comparison)

Subset where algebraic runs the real boundary integral (value > 0.4 ms), n = 32:

| backend | p50 | p90 | p95 | p99 | max |
|---|--:|--:|--:|--:|--:|
| **algebraic value-only** | 1.09 | 1.98 | 2.25 | 2.46 | 2.55 |
| **algebraic value + analytic Jac** | 1.27 | 8.32 | 9.21 | 11.53 | 12.47 |
| **holonomic fused value + Jac** | 7.14 | 8.36 | 8.41 | 8.46 | 8.47 |
| inverse-ray value + FD Jac | 57.4 | 4413 | 4521 | 4643 | 4643 |

Ordinary-binary **deep-solve** only (n = 14): algebraic value+Jac p50 **5.19 ms**
/ p95 10.50 ms; holonomic p50 **7.21 ms** / p95 **8.46 ms**.

Reading:

* **value-only:** algebraic wins clearly (≈ 1 ms vs ≈ 7 ms). Its polar
  boundary integral with (m,v) root-pair transport is a genuinely cheaper way
  to get μ than holonomic's full radial reconstruction.
* **value + Jacobian, median:** algebraic still ahead (5.2 vs 7.2 ms in
  ordinary binary) — but only ~1.4×, and see §5: on most of these cases its
  Jacobian is flagged unreliable.
* **value + Jacobian, tail:** holonomic wins — p95 8.5 ms vs 10.5 ms, max
  8.5 ms vs 12.5 ms. Holonomic's cost is bounded by a fixed node budget
  (n_r = 64, 64-node angular, 3072-node topology probes); the algebraic deep
  solve's cost scales with the caustic geometry and the LD recurrence length.
* **both** annihilate the inverse-ray baseline in the planetary regime, where
  its ray count blows up (p95 4.4 s).

---

## 4. Robustness / fail-closed

| | inverse-ray | algebraic | holonomic |
|---|--:|--:|--:|
| value: hard failure (no answer, non-terminating) | 0 | **2 / 108** | 0 |
| value: fail-closed (returns, `success=false`) | 0 | 2 / 108 | 4 / 108 |
| value: silent (`OK` but > 5 % wrong vs M0) | 0 | 0 | 0 |
| Jacobian: delivered & reliable | 108 (FD) | **62 / 108 (57 %)** | **104 / 108 (96 %)** |

* **`rand002` (u=0 and u=0.5) — algebraic non-termination.** Config
  `xs −0.410, ys −0.086, ρ 0.069, q 1.19e-4, a 7.64` (wide planet).
  `experimental_algebraic_boundary_binary_mag` does not return — memory grows
  unbounded (~45 MB/s, killed at 6 GB). Verified in isolation with three
  independent probes. **This is a fail-closed policy violation** — the backend
  neither answers nor reports failure. Holonomic returns μ = 2.5474, status OK
  on the same config (matches M0 μ = 2.5474 to 7e-6). The 3-way harness
  hard-codes a skip for `rand002` so the sweep completes.
* **`rand003` (u=0, 0.5) — algebraic fail-closed.** `unsafe_reason =
  "certified_seed_did_not_lie_on_an_active_radius"`, returns μ = 0 with
  `success = false`. Correct fail-closed behaviour. Holonomic: status OK,
  μ = 2.156 (matches M0 to 3e-5).
* **Holonomic fail-closed set (4):** `rand008` (u=0, 0.5), `rand031` (u=0, 0.5)
  → `GRADIENT_UNRELIABLE` / `TOPOLOGY_UNCERTAIN` (genuine near-tangency; the
  three `arcs_at` probe radii disagree). Same points the M6 reference flags.
* **Algebraic Jacobian fail-closed rate is regime-correlated:** 30 % of
  ordinary-binary points and **52 % of planetary points** get
  `grad_reliable = false` (caller must NaN the row). These are
  disproportionately the deep-solve / near-caustic configs — i.e. exactly the
  cases where a finite-source Jacobian actually matters. Holonomic's
  reliable-Jacobian rate is 96 % in both regimes.

No backend produced a confidently-wrong value (> 5 %) with an OK status.

---

## 5. Accuracy

### value, relative to `binary_ray_shooting` M0 μ

| slice | algebraic vs M0 | holonomic vs M0 | algebraic vs holonomic |
|---|---|---|---|
| ALL (n≈104) | med 7.9e-6, p90 9.3e-5, max 3.5e-4 | med 2.7e-5, p90 1.7e-4, max 7.7e-3 | med 2.5e-6, p90 1.0e-4, max 7.7e-3 |
| ordinary binary | med 4.1e-6, max 3.5e-4 | med 5.9e-5, max 7.7e-3 | med 3.6e-5 |
| planetary low-q | med 8.5e-6, max 2.5e-4 | med 1.6e-5, max 2.5e-4 | med 2.6e-8, p90 3.8e-5 |
| uniform u=0 | med 6.3e-6 | med 2.9e-5 | med 2.5e-6 |
| linear LD u>0 | med 8.2e-6 | med 2.4e-5 | med 2.5e-6 |

Both backends are well inside microlensing modelling tolerance. Algebraic is
~3× tighter at the median (helped by the exact multipole shortcut on easy
configs).

**Two holonomic OK-status outliers** (already recorded in
`evidence/holonomic/m7_accuracy_coverage.txt`, not new):

| config | holonomic μ | M0 μ (= algebraic μ) | rel err |
|---|--:|--:|--:|
| `tiny-rho` (ρ = 5e-3) u=0 / 0.5 | 38.044 / 38.029 | 38.338 / 38.323 | 7.7e-3 |
| `very-wide` (a = 5) u=0 / 0.5 | 1.02752 | 1.03250 | 4.8e-3 |

Status is `OK` (mildly optimistic) — a ~0.5–0.8 % bias in the deep-source
reconstruction for a very small source and for a very wide binary. Below the
5 % silent-failure gate but worth a dedicated tolerance/So-status check.
**M8 follow-up item.** Algebraic hits its exact multipole path here and matches
M0.

### Jacobian

* On configs where **both** deliver a reliable analytic Jacobian and the
  geometry is smooth, the two analytic Jacobians agree with each other and each
  with a 4th-order finite difference to **≤ 1e-3** (spot-checked: wide-planet,
  fold-tangent — d/d[x,y,ρ] components). Both analytic Jacobians are correct.
* Holonomic analytic ∇μ was validated against the M6 Python analytic Jacobian
  in M7: median 5.4e-10, p90 6.0e-7, max 2.1e-2 (caustic-cross, a topology
  edge). See `evidence/holonomic/m7_accuracy_coverage.txt`.
* Algebraic analytic Jacobian carries its own two-form reliability estimate
  (`grad_error`) and gates on it — the mechanism behind the 57 % delivery rate.
* The aggregate "holo vs algebraic analytic-Jac" cross-metric in the harness
  (`median 0.11`) is **not meaningful** — it is dominated by (a) near-zero
  components (symmetric configs, ∂μ/∂y ≈ 1e-13) blown up by the relative
  denominator floor, and (b) configs sampled next to a topology transition.
  It is reported by the harness for completeness only.

---

## 6. Wall-clock breakdown per backend

### inverse-ray (incumbent M0)

* value: `binary_ray_shooting`, p50 ≈ 1.9 ms; **planetary p95 ≈ 400 ms, max
  422 ms** — ray count scales with 1/q and with separation.
* Jacobian: central FD, **11 × value** (1 base + 10 perturbed). p50 14.9 ms;
  planetary p95 4.4 s.
* No shared geometry between the 11 solves; no analytic derivative.

### algebraic-boundary (`cc5e55d`)

* **routing / multipole shortcut** (74/108): a point-source + hexadecapole
  stencil, closed form. ~0.05–0.25 ms value, ~0.013 ms Jacobian. No boundary
  integral.
* **deep solve** (32/108), from `AlgebraicBoundaryDiagnostics` (plan15 u0):
  * quartic boundary-root solves ~2060× (0 fallbacks) — the dominant term
  * lens-map evaluations ~10 k (u=0) / ~15 k (u=0.5)
  * 2 radial bands, ~48 adaptive subdivisions
  * (m,v) root-pair transport 1640× with **420 fallbacks** (~20 % fall back to
    cold re-solve — a tail driver)
  * linear LD: `ld_moment_series` recurrence ~1340× (41 fallbacks) — the u>0
    doubling
  * value p50 ≈ 1.1 ms
* **Jacobian**: one `ForwardJet<5>` pass over the frozen seed set — ~2.5–3×
  the value (p50 1.3 ms; deep-solve p90 8.3 ms; LD max 12.5 ms).

### holonomic M7 (fresh sub-phase timing, best-of-200, taskset)

| phase | plan15 u0 | resonant u0 | caustic-cross u0 | wide-planet u0 |
|---|--:|--:|--:|--:|
| `classify_cells` (topology) | 2.92 ms (64 %) | 4.70 ms (58 %) | 6.06 ms (83 %) | 2.75 ms (75 %) |
|  ↳ of which `radial_events` (D14/p4/L build + solve) | 0.58 | 2.01 | 3.78 | 0.71 |
| per-cell radial × angular pass (+ fused 5-Jac) | 1.62 ms (36 %) | 3.44 ms (42 %) | 1.26 ms (17 %) | 0.94 ms (25 %) |
| **total `epoch_jacobian`** | **4.54 ms** | **8.14 ms** | **7.32 ms** | **3.69 ms** |

This **revises** the M7 §7 estimate (which put `classify_cells` at ~30 %). Fresh
measurement: **`classify_cells` is 58–83 % of holonomic wall-clock**, and it is
dominated by the **3 × `arcs_at(3072)` probes per cell** plus `radial_events`.
The per-cell radial × angular pass is 17–42 %. The D14 two-stage solve proper
is ~6 % (measured in M7). Chain rule + assembly < 4 %.

**⇒ the user's M8 prioritisation is confirmed by data:** cut `arcs_at(3072)` /
`classify_cells` first; the per-cell radial×angular pass second; do **not**
over-invest in the D14 solve.

---

## 7. Verdict (with the mandated framing)

1. **Holonomic vs inverse-ray baseline:** value+Jacobian median **4.78 ms vs
   14.89 ms = 3.1× (vs inverse-ray baseline)**; p95 8.30 ms vs 4392 ms; no
   planetary tail. This is stated *only* as "vs inverse-ray baseline".
2. **Holonomic vs algebraic-boundary, as of M7:**
   * value-only: **algebraic faster** (deep-solve ~1 ms vs ~7 ms).
   * value+Jacobian median, ordinary-binary deep-solve: **algebraic faster**
     (~5.2 ms vs ~7.2 ms, ≈ 1.4×).
   * value+Jacobian **tail**: **holonomic faster / bounded** (p95 8.5 vs
     10.5 ms; max 8.5 vs 12.5 ms).
   * **Jacobian availability: holonomic decisively better** — 96 % vs 57 %
     reliable delivery, and algebraic fail-closes hardest exactly on the
     near-caustic deep-solve cases.
   * **Robustness: holonomic better** — algebraic has a non-terminating config
     (`rand002`); holonomic has none.
   * **Accuracy:** both fine; algebraic ~3× tighter median; holonomic has two
     ~0.5–0.8 % wide/tiny-source biases to close.
3. **The M8 goal is well-posed and reachable:** beat the algebraic-boundary
   **ordinary-binary deep-solve median** (currently 7.2 → target < 5 ms, ideally
   < 3 ms) while keeping the p95/p99 bounded-tail and Jacobian-availability
   advantages. `classify_cells` / `arcs_at(3072)` is 58–83 % of the cost —
   a `quartic_topology` replacement + trajectory-level cache + SIMD on the
   radial×angular pass is the path.

---

## 8. Artifacts

* `tests/holonomic_cpp/bench_three_way.cpp` — the 3-way harness.
* `tests/holonomic_cpp/bench_algebraic_one.cpp` — single-case algebraic probe
  (for `timeout`-bounded sweeps around the non-termination).
* `/tmp/bench_cases_ext.tsv` — 108 cases + M0 μ/grad/timings, regenerable from
  `evidence/holonomic/baseline_M0.json`.
* Isolated algebraic build: detached worktree at `cc5e55d`,
  `.claude/worktrees/algebraic-bench-cc5e55d/build-bench/*.a` (private static
  libs; shared `.so` untouched). **Remove with `git worktree remove` once M8
  no longer needs the A/B.**
* Raw run logs: `/tmp/tw_full2.out` (full report + `#CSV` per-case dump).
