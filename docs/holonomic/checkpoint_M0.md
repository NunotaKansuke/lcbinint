# Checkpoint M0 — baseline fixed and instrumented

Date: 2026-09-07
Branch: `claude/holonomic-solver-m1` (shared feature branch; the `…-b7d3cd`
in the docs header is the worktree, not the checked-out branch)
Status: **complete** (baseline cost decomposition reproducible;
`benchmarks/holonomic/baseline_probe.py` + `evidence/holonomic/baseline_M0.json`)

M0 was deferred until after the Python reference (M1–M6) existed so the
baseline could be measured against the same audit domain the reference is
graded on. It fixes the number the M7 C++ solver has to beat and — more
importantly — it **corrects a wrong premise in the plan**: the incumbent
inverse-ray solver is *not* uniformly slow.

---

## 1. What was implemented

| File | Purpose |
|---|---|
| `benchmarks/holonomic/baseline_probe.py` | times `lcbinint.binary_ray_shooting` (uniform + linear-LD `u=0.5`) for **value-only** and **value + 5-component Jacobian** (central FD, 11 calls) over `audit.curated()` + `audit.random_sample()`; each config in a `fork`ed child with a hard timeout; reports median / p95 / coverage; optional `--ref` times the Python `holonomic_ref` oracle and grid accuracy; always writes `evidence/holonomic/baseline_M0.json` |
| `evidence/holonomic/baseline_M0.json` | raw rows from the `--n 40` run below (108 points = 27 usable curated + 40 random, × `u ∈ {0, 0.5}`) |

No production code touched. No build. `binary_ray_shooting` is the plan's
correctness bar ("現行 production solver 同等以上", `checkpoint_M5.md`).

Run: `taskset -c 0-7 python -m benchmarks.holonomic.baseline_probe --n 40`
(`uptime` load ≈ 10–16 on 64 cores during the run; recorded in the JSON
header line and stdout).

---

## 2. Baseline choice — design decision 18

The natural baseline is the **JAX differentiable backend**: one
forward/reverse pass of `binary_ray_shooting(..., jax=True)` gives value +
Jacobian directly. It is **unavailable in this build**:

```
>>> import jax; jax.config.update("jax_enable_x64", True)
>>> import lcbinint
>>> lcbinint.binary_ray_shooting(0.2, 0.1, s=1.2, q=0.5, rho=0.1, jax=True)
RuntimeError: lcbinint was built without the polar epoch FFI
#   _lcbinint._jax_ir has no attribute 'polar_epoch_directional_ffi'
```

and it cannot be rebuilt here — the FFI lives in the single shared
`site-packages/_lcbinint*.so` that every other session and the user's own
sweeps import, and a `pip install` rebuild overwrites it globally
(memory `project_editable_install_serves_stale_so.md`).

**Resolution (recorded, autonomous):** the baseline for *value + Jacobian*
is a **central finite-difference sweep of the native inverse-ray solver**
— 11 `binary_ray_shooting` calls (base + 2 per parameter, steps
`FD_REL = {xs,ys,a: 1e-4, rho: 1e-3, q: 1e-3}` with an absolute floor).
This is the route a production consumer actually has today. The M7 C++
solver is additionally reported against the value-only native cost (the
absolute floor for a Jacobian-free consumer) for honesty.

The isolated-build technique for M7 (build into `build-holonomic-m7/` in
the worktree, load via `sys.modules["lcbinint._lcbinint"]` pre-population,
never touch site-packages) is design decision 19, recorded in
`checkpoint_M7.md`.

---

## 3. Result — the incumbent cost is strongly bimodal

`--n 40`, 54 configs × `u ∈ {0, 0.5}` (identical timing for both `u`;
numbers below are `u = 0`, `u = 0.5` within noise):

| percentile | value-only | value + Jacobian (central FD, 11 calls) |
|---|---|---|
| p50  | **1.9 ms** | **15 ms** |
| p75  | 4.8 ms | 45 ms |
| p90  | 12 ms  | 0.12 s |
| p95  | **398 ms** | **4.38 s** |
| p99  | 411 ms | 4.53 s |
| max  | 421 ms | 4.64 s |
| mean | 40 ms  | 0.43 s |

Coverage 108/108 (the source-covers-a-lens configs are filtered up front —
inverse-ray never returns there; those are the holonomic singular patch and
have **no native baseline at all**).

**The slow tail is the extreme-mass-ratio / planetary regime.** The 9.3 %
of configs with value+Jacobian > 0.5 s all have `q ≲ 1.4e-3`
(`extreme-q-planet` `q=3e-5`; `rand031` `q=6e-8`; `rand003` `q=4e-7`;
`rand025` `q=1.8e-4`; `rand029` `q=1.4e-3`). Inverse-ray shooting has to
resolve a tiny planetary caustic / tiny secondary images → dense grid →
~400 ms/call. Everything else (binary-scale `q`) is 2–12 ms/call and the
FD Jacobian is 15–120 ms.

### 3.1 The Python `holonomic_ref` reference (oracle, **not** an M7 target)

`epoch_jacobian(params, u, n_r=64)` — the M6 finite-numeric jet, the exact
computation M7 ports — is **0.40–0.45 s** in pure Python, essentially
independent of `q`. `cProfile` (plan15, `n_r=64`):

| block | cumtime | why it is slow in Python | C++ replacement |
|---|---|---|---|
| `radial_events` / `_pos_roots` (`classify_cells`) | 0.23 s | `sympy.Poly.nroots` / `real_roots` (mpmath exact arithmetic) | numeric companion / quartic solve, ~µs |
| `radius_terms` ×512 | 0.20 s | `phi_grad` ×41 k (0.09 s) + mpmath root-polish (0.13 s) | all `double`, ~5 ms |
| connection / transport | — | not in the hot path at `n_r=64` (periods re-anchored per node, `checkpoint_M4.md` §4) | — |

Removing sympy+mpmath is a 50–100× constant-factor win on the two hot
blocks, so a competent C++ port of `epoch_jacobian` at `n_r=64` is
**estimated at 3–12 ms** for value + full Jacobian, *bounded* across `q`.
Estimate only — unproven until M7 exists.

---

## 4. Consequence for the M7 gate — re-framing (design decision 18, cont.)

Plan §14 gate: "linear-LD value+Jacobian の end-to-end **median >= 2x**,
**p95 悪化 <= 25%**".

Measured against the corrected baseline:

* **Median (15 ms):** target ≤ 7.5 ms. The C++ estimate (3–12 ms) makes
  this **borderline** — the median inverse-ray config is already near the
  floor (~2 ms value); "2× on the median" is a race at the bottom and not
  where a holonomic solver's value lies.
* **p95 (4.38 s) and the 9.3 % extreme-`q` tail:** the holonomic solver's
  cost is *bounded* (no caustic to resolve on a grid), so here it is
  ~300–800× faster, trivially inside "悪化 <= 25%".
* **Jacobian quality:** the baseline Jacobian is a *finite difference of a
  grid solver* — it carries the grid's ~1e-3 value noise amplified by
  `1/h`. The holonomic jet is the analytic JVP from the same numeric
  reconstruction as the value (plan §11 contract). This is a correctness
  advantage the median-time gate does not capture.

**Recommended M7 success criteria (superseding the bare "median ≥ 2×"):**

1. **p90 / p95 / p99 end-to-end** value+Jacobian cost strictly below the
   baseline (the bounded-cost win — this is the headline).
2. **Median** value+Jacobian **≤ the baseline median** (no regression;
   ≥ 2× is a bonus, not a gate).
3. **Same accuracy / coverage** as M5/M6: audit median rel ≤ 1e-3,
   zero silent miss, fail-closed on the flagged statuses.
4. Jacobian agrees with an independent central FD of the *reference* to
   the M6 bars (plan15/resonant ≤ 1e-3; small-ρ ∂μ/∂ρ, ∂μ/∂a oracle-
   limited ~1e-2).

This is a change to the *emphasis* of the gate, not its intent: the plan
wants the holonomic backend adopted only if it is a real improvement, and
the real improvement is the bounded tail + the clean Jacobian, not the
already-fast median.

---

## 5. Open items carried into M7

* The C++ 3–12 ms estimate is unverified. If the port lands above ~15 ms
  the median gate fails and adoption rests entirely on the tail + Jacobian
  quality argument — flag to the user at that point.
* The extreme-`q` tail configs need the holonomic solver's *own* accuracy
  checked there (M5 audit covered `q ≥ 1e-8` random but the
  planetary-caustic `ρ ~ √q` band is the M5 non-adoption region,
  design decision 9). M7 must either certify or fail-closed there.
* `baseline_probe.py` uses `fork` + `Queue` + `terminate()` for the
  per-config timeout; the extreme-`q` value calls are ~0.4 s so the
  900 s Jacobian budget is generous. If a future domain widens `q` down
  to 1e-12 the inverse-ray calls may hit minutes — the timeout will then
  start reporting `TIMEOUT` rows (fail-closed, counted against coverage).
