# Checkpoint — unified finite-source engine + warmup integration (design only)

Status: **design checkpoint, no code change.** Written against the tree at
`cbaa673` (holonomic worktree) and the algebraic-boundary backend at
`cc5e55d` (`algebraic-bench-cc5e55d` worktree).

**Revision 2 (2026-09-08).** Three corrections from review, before any
implementation:

1. The M1 micro-experiment is split into **three variants V0 / V1 / V2** so the
   speed-up cause is separable (§13). Kernel unification is **not** started yet.
2. §5.2's "band endpoint `|dφ/dθ|` above threshold" was **wrong** — a radial
   band endpoint *is* a tangency (`φ = 0`, `φ_θ = 0`), so `|dφ/dθ| ≈ 0` there
   by definition. Replaced with the tangency-system regularity determinant
   `det ∂(φ, φ_θ)/∂(R, θ)`.
3. The fast-planner checks are a **heuristic / local confidence screen**, not a
   completeness proof. Explicit three-tier split: fast planner → *candidate*
   bands; local checks → *confidence screen*; D14 / future root-exclusion
   certificate → *completeness authority*. The same rule applies to warmup
   reuse: `compare_warmup_geometry`'s `warn == false` is a pre-filter, **not**
   proof that a cached band topology is still complete.

Goal restated: stop optimising *algebraic-boundary* and *holonomic* as two
competing backends. Design one **fully integrated finite-source engine** that
(a) shares each side's strong parts in a single numerical kernel, and (b)
routes purely by *requested output × geometry*, doing only the minimal work
each answer needs. Do **not** assume either side "wins".

The four requested-output modes this engine must serve:

| mode | outputs | today's cheapest source |
|---|---|---|
| **M1** | uniform value only (c = d = 0) | algebraic: exact `m0` from interval lengths + multipole shortcut |
| **M2** | linear-LD value only (c > 0) | algebraic: closed-form LD moment series → GK fallback |
| **M3** | value + JVP (one direction) | (neither has a dedicated path today) |
| **M4** | value + full 5-Jacobian + ∂μ/∂u | holonomic: fused analytic pass, decision-20 met (2.14 ms median, 11.7× incumbent) |

---

## 1. Current algebraic vs holonomic — common parts and differences

### 1.1 Shared substrate (already literally the same maths)

* **Fixed image radius R.** Both integrate the lens equation on circles
  `|z| = R` and sweep R radially.
* **Binary boundary quartic `P(t; R)`, `t = tan(θ/2)`.** `holonomic::boundary_quartic`
  and `algebraic::algebraic_boundary_quartic_fixed` build the *same* degree-4
  self-inversive polynomial (confirmed by the `find_radial_bands_d14` comment
  in `algebraic_boundary.cpp:1883`).
* **D14(v = R²).** The angular discriminant of `P(t; R)` with the R⁴ chart
  factor removed, degree 14 in `v = R²`. `holonomic::radial_events` enumerates
  its positive real roots in `__float128`; the algebraic side has the same
  thing as the experimental `find_radial_bands_d14()`.
* **Real boundary roots → angular arcs → radial integrand.** Both: solve the
  quartic at each radial node, filter to physical roots, form
  `(θ_enter, θ_leave)` arcs, integrate `√φ`-weighted moments along each arc,
  accumulate radially.
* **Root continuation between neighbouring radial nodes.** Both warm-start the
  per-node root solve from the previous node.
* **Aberth–Ehrlich.** `holonomic/poly_roots.hpp` for the quartic (double) and
  D14 (`__float128`); the algebraic side additionally has a real self-inversive
  quartic path and Skowron–Gould as fallback.

### 1.2 Where they differ

| aspect | holonomic (`epoch_jacobian.hpp` / `radius_terms.hpp` / `cells.hpp`) | algebraic (`algebraic_boundary.cpp`) |
|---|---|---|
| band discovery | **always** D14 `__float128` (`radial_events`) | seed-anchored exponential radial march + 16 bisections + `solve_radial_tangency`, **~0.03 ms** (D14 only as experimental `find_radial_bands_d14`) |
| radial rule | fixed `n_r = 64` Gauss–Chebyshev per cell | adaptive Gauss–Kronrod bands, one-panel estimate first, refine only if error over budget |
| angular rule | fixed GC-1(64) `√φ` sweep | sine-mapped Gauss–Kronrod 21 + **closed-form `algebraic_moment_series`** fast path (GK fallback) |
| value-only (c = d = 0) | still runs the full fused pass (F0, F_half, dF0, dF_half) | `m0` **exact** from interval lengths, **no angular quadrature at all** |
| ordinary geometry | no shortcut — always solves the topology | **multipole/hexadecapole shortcut resolves ~74/108 configs with no boundary integral** |
| Jacobian | **fused** in the same radial pass: analytic IFT-θ endpoint derivs + `d√φ/dP` accumulation, 5 internal params → chain rule → 5 user params | **separate** `experimental_algebraic_boundary_jacobian`: re-runs one `ForwardJet<5>` forward pass over the frozen discrete plan of the primal; plus `..._multipole_jacobian` (~1% cost) for shortcut-resolved epochs |
| quartic roots | Aberth double, `QuarticWarm` per-cell warm-start | self-inversive real quartic → `(m,v)` E/O transport (`active_mv_cache`) → Skowron–Gould (`RootContinuationWorkspace`) |
| endpoint | 6-iter Newton `polish_endpoint` on `φ(R,θ)=0` | lens-map residual certification (`residual = |φ|·ρ`) |
| denormal guard | `ScopedFlushDenormals` (FTZ/DAZ MXCSR) around the whole epoch | none |
| failure signalling | `Status` enum (`OK` / `GRADIENT_UNRELIABLE` / `TOPOLOGY_UNCERTAIN`) — fail closed | `unsafe_reason` string + `grad_reliable` bool — fail closed |
| parametrisation | internal `(X, Y, ρ, m0, a_pf)` → user `(xs, ys, ρ, q, a)` | Mapping B (barycentric): `separation = a`, `mass_ratio = q`, `source = (xs − q/(1+q)·a, ys)` |
| wiring | standalone header, driven by bench harness — **not** in `FiniteSourceMagnifier` | standalone experimental functions — **not** in `FiniteSourceMagnifier` |

### 1.3 Strength summary (the parts worth keeping)

* **Algebraic:** ~0.03 ms band march; exact `m0` value-only; closed-form LD
  moment series; multipole shortcut (68% of configs skip the boundary integral
  entirely); no global topology solve for ordinary geometry.
* **Holonomic:** LD moments in the same pass; **fused value + analytic
  Jacobian** (no second solve); bounded `__float128` D14 tail; Jacobian
  reliability gating; robustness on pathological geometry; `QuarticWarm`
  warm-start; FTZ/DAZ guard.

---

## 2. Which components can be unified into one

Concrete unification candidates, in decreasing confidence:

1. **Boundary-quartic construction.** One `boundary_quartic(R, frame)` builder.
   Today two copies of the same polynomial. *No behaviour change.*
2. **Scaling / balancing + Aberth cold solve + neighbouring-R warm-start.**
   One templated `aberth<R>` (already in `poly_roots.hpp`), one warm-start
   state type. The algebraic real-quartic + `(m,v)` transport and the
   holonomic `QuarticWarm` become two *policies* over the same primitive.
3. **Root correspondence + `(m, v)` representation.** A single "ordered root
   set / quadratic-factor" representation carried across radial nodes and
   across epochs (§7). The `(m,v)` E/O system (`refine_mv_pair`) is exact for
   a quartic and is the natural transport form for both sides.
4. **Newton endpoint polish.** `polish_endpoint` (holonomic) and the lens-map
   residual gate (algebraic) are the same certification; unify as "polish then
   certify by residual", one function.
5. **FTZ/DAZ denormal guard.** Apply to the whole unified epoch (the algebraic
   side currently pays the x86 denormal penalty it doesn't know about).
6. **Radial-event / D14 kernel.** One `radial_events` (`__float128`), shared as
   the completeness oracle — *not* the normal path (§5).
7. **Conditioning / fallback policy.** One `Status` enum replacing the
   `unsafe_reason` string; one "valid → use / uncertain → recompute /
   planner-uncertain → D14 / D14-uncertain → fail closed" ladder (§8).
8. **Engine-neutral geometry.** `FiniteSourceGeometry` already exists
   (`finite_source_magnifier.hpp:28`) and is exactly the shared input.

**Not unified (kept as mode-specific policy, not backend):** the radial rule
(fixed GC for M4, adaptive GK for M1/M2), the angular rule (skip for M1,
series/GK for M2, fused-analytic for M4), and the multipole shortcut (a
front gate, not part of the deep kernel).

---

## 3. Proposed unified solver data flow

```
INPUT:  FiniteSourceGeometry g           (shared, engine-neutral, exists today)
        RequestedOutputs r = { value, +LD(c,d), +JVP(seed) | +Jacobian5 }

 1. frame = PrimaryFrame::from(g)                       [shared boundary frame]

 2. MULTIPOLE / HEXADECAPOLE GATE  (ordinary-geometry fast exit)
      compute VBM a4 correction on the 1- or 13-point stencil
      if |a4|/mu <= adaptive_hex_threshold:
          value      -> return stencil value
          +Jacobian  -> ForwardJet<5> over the SAME stencil (~1% cost) -> return
      else -> deep path

 3. BAND DISCOVERY  (fast planner, ~0.03 ms)  ->  CANDIDATE bands only
      seed-anchored exponential radial march (step doubling)
        + boolean bisection + algebraic tangency solve
      -> candidate bands + per-endpoint conditioning margins
         (tangency regularity det = phi_R*phi_thetatheta at each endpoint,
          min band width vs march resolution,
          up-march vs down-march band-count agreement)

 4. LOCAL CONFIDENCE SCREEN  (cheap, no degree-14 solve; heuristic, NOT a proof)
      (a) 3 in-band probes agree on (kind, crossing count)   [holonomic cell check, minus the 3072 grid]
      (b) every band endpoint's tangency regularity |det ∂(phi,phi_theta)/∂(R,theta)|
          = |phi_R * phi_thetatheta| above a relative threshold
      (c) up-march and down-march band counts agree
      (d) no band narrower than N x march resolution
      pass       -> adopt bands, provenance = fast_march
      any fail   -> D14 ORACLE (completeness authority):  radial_events(frame) in __float128
                      clean      -> adopt, provenance = d14_oracle
                      degenerate -> Status = TOPOLOGY_UNCERTAIN, FAIL CLOSED

 5. PER BAND, PER RADIAL NODE R_k  (rule chosen by mode, step 6)
      solve boundary quartic once  (warm-started from node k-1: unified
        (m,v) / QuarticWarm state; seeded from the prepared-geometry cache
        at k = 0 when available, see section 7)
      form arcs (theta_enter, theta_leave)
      M1 (c=d=0):  accumulate f0 = R * sum (theta_leave - theta_enter)   [NO angular pass]
      M2:          + angular moment  int sqrt(phi) dtheta   (series -> GK fallback)
      M3:          + directional derivative along `seed`     (one ForwardJet direction)
      M4:          + fused analytic dF0[5], dF_half[5]       (IFT-theta + d sqrt(phi)/dP, same loop)

 6. RADIAL ACCUMULATION
      M1 / M2:  adaptive Gauss-Kronrod, one-panel estimate first, refine to budget
      M3 / M4:  fixed n_r Gauss-Chebyshev  (the analytic IFT derivative assumes
                the value used this same fixed grid -- see section 12)

 7. ASSEMBLE
      mu(u)   = ((1-u) F0 + u F_half) / D,   D = pi rho^2 (1 - u/3)
      grad_mu = ((1-u) dF0 + u dF_half) / D   (- 2 mu / rho on the rho column)
      dmu_du  = pi rho^2 (F_half - 2 F0 / 3) / D^2
      Status  = worst over every stage (never upgraded back to OK)
```

The router is a single internal function
`finite_source_binary(FiniteSourceGeometry, RequestedOutputs) -> Result` —
**dispatch is by `RequestedOutputs` and the step-2/step-4 geometry verdict,
never by "algebraic" or "holonomic".**

---

## 4. Work performed per mode

`+` = performed, `–` = skipped, `~` = cheap variant.

| stage | M1 value | M2 value+LD | M3 value+JVP | M4 value+5-Jac |
|---|---|---|---|---|
| multipole gate | + | + | + (stencil deriv) | + (stencil deriv) |
| band discovery (fast march) | + | + | + | + |
| D14 oracle | only if cert fails | only if cert fails | only if cert fails | only if cert fails |
| boundary quartic / node | roots only | roots only | roots + 1 direction | roots + 5 directions |
| endpoint polish + certify | + | + | + | + |
| angular `√φ` pass | **–** (m0 exact) | + (series → GK) | + `d/dseed` | + fused `d/dP` (5) |
| `F_half` | – | + | + | + |
| radial rule | adaptive GK, 1-panel exit | adaptive GK | fixed GC `n_r` | fixed GC `n_r` |
| **second full solve for the Jacobian** | – | – | **– (fused)** | **– (fused)** |
| Jacobian state alloc (`dF0`,`dF_half`,`*_internal`) | **–** | **–** | 1 column | 5 columns |
| chain rule internal→user | – | – | + | + |
| denormal guard | + | + | + | + |

Key invariants the user asked for:

* **M1/M2 pay no Jacobian-state cost** — the `dF0/dF_half` accumulators and
  the internal→user chain rule are not entered.
* **M4 does not re-run a solve after the value** — value and Jacobian come out
  of the same per-node loop (this is exactly today's `flux_jacobian`).
* Marginal cost to *add* LD over M1 = one angular quadrature per node.
* Marginal cost to *add* the Jacobian over M2 = the derivative accumulation in
  the loop already visiting every node (no new geometry work).

---

## 5. Fast algebraic planner + D14 fallback design

### 5.1 Fast planner (normal path)

Port of `find_radial_bands()` (`algebraic_boundary.cpp:1660`):

* seed-anchored exponential march from the certified image seeds, `step *= 2`,
  capped at `settings.maximum_band_search_steps`;
* 16-iteration boolean bisection on each `radius_has_image` sign change;
* `solve_radial_tangency()` algebraic tangency solve (3-iteration prefilter +
  boolean-bisection fallback) for each band endpoint;
* band merge at `1e-11`.

Cost measured on the algebraic side: **~0.03 ms**. Output: bands +, per
endpoint, the tangency-solve residual and the `|dφ/dθ|` at the tangency.

### 5.2 Local confidence screen (cheap, no degree-14 solve)

**This is a heuristic screen, not a completeness proof** (see §5.5). It raises
confidence that the fast planner's candidate band set is complete and
well-conditioned; it cannot *prove* it. The band set passes the screen iff
**all** of:

1. **Uniform topology in band.** The 3 in-band probe radii (fractions
   0.18 / 0.50 / 0.82) agree on `(kind, crossing count)` via
   `quartic_topology` — the holonomic cell check from `cells.hpp:99`, but
   *without* the 3072-point grid cross-check.
2. **Tangency regularity.** A radial band endpoint is a *tangency*: at the
   endpoint `(R*, θ*)` both `φ(R*, θ*) = 0` and `φ_θ(R*, θ*) = 0`. So the
   endpoint's own `|φ_θ|` is ≈ 0 by construction and is **not** a usable
   margin. The correct local metric is the regularity (fold vs higher-order
   degeneracy) of the tangency system `G(R, θ) = (φ, φ_θ)`:

   ```
   det ∂G/∂(R, θ) = φ_R φ_θθ − φ_θ φ_Rθ
                  = φ_R φ_θθ        at the tangency (φ_θ = 0)
   ```

   This is the discriminant of the projection of the tangency curve
   `{φ = φ_θ = 0}` onto the R axis. `|det|` well above a relative threshold ⇒
   an ordinary fold (band birth/death moves regularly in R) ⇒ the fast march's
   local linear picture is valid. `|det| → 0` ⇒ either `φ_R ≈ 0` (band edge
   stationary in R) or `φ_θθ ≈ 0` (cusp-like higher-order contact) ⇒ **do not
   fast-accept, escalate to D14.** (Any mathematically equivalent regularity
   metric — e.g. the resultant-based tangency discriminant — is acceptable;
   `φ_R φ_θθ` is the cheapest given the holonomic `φ` derivative primitives.)
3. **Direction agreement.** An independent march *down* from `r_max` yields the
   same band count as the march up.
4. **Resolution margin.** No candidate band is narrower than `N ×` the local
   march step (an unresolved thin band would be silently missed → `m0` low).

The 512-point independent grid cross-check that `classify_cells` does today is
replaced, *at screen level*, by (1)+(3): two independent *root-based* verdicts
instead of one root-based + one grid-based. Grid probing stays available only
inside the D14 oracle path for the chart (`p4 ≈ 0`) radius, as
`quartic_topology` already does.

### 5.3 D14 oracle (completeness authority)

If any screen check fails → run `radial_events(frame)` in `__float128`
(the existing holonomic path: balanced double pre-search → warm `__float128`
Aberth → cold `__float128` on residual miss). This is the **completeness
certificate**: its positive real roots are every band birth/death, its complex
roots (Re v > 0) are the soft boundaries. Provenance is recorded as
`d14_oracle`.

If `d14_coeffs` is structurally degenerate (leading coeff vanishes / not even
in R) or `double_root_is_real` cannot classify a root → `Status =
TOPOLOGY_UNCERTAIN`, the epoch fails closed (caller NaNs the row / falls back
to the incumbent ray-shooting solver).

**D14 is an oracle, not a normal-path step.** Expected screen-fail rate: low
for ordinary and wide/close geometry, higher near caustic cusps and for
extreme q — to be measured (§11). The existing evidence
(`checkpoint_fullsolve_and_d14_algebraic.md`) that "the holonomic win is the
fused transport pass, not D14" and that a naive D14-into-algebraic port gave a
silent −9.6% with no root-solve win is the reason D14 must not run on every
epoch.

### 5.4 Three-tier responsibility split (epistemics)

| tier | component | guarantee it provides |
|---|---|---|
| **candidate** | fast seed-anchored radial march (§5.1) | a plausible band set, ~0.03 ms; **no** completeness guarantee |
| **confidence screen** | the 4 checks (§5.2) | *heuristic* local confidence: conditioning is OK and no obvious missed band; **not a proof** |
| **completeness authority** | D14 `radial_events` (§5.3); *future:* an inter-band root-exclusion certificate (§5.5) | every band birth/death is accounted for, to `__float128` precision |

The screen can be fooled: an up-march, a down-march and all 3 in-band probes
can in principle miss the *same* thin newborn band simultaneously (they sample,
they do not bound). So the screen governs only *whether the D14 authority is
invoked*, never *whether completeness holds*. Any epoch whose screen is not
fully confident goes to the D14 authority; any epoch the authority cannot
resolve fails closed.

**Consequence for the architecture:** as currently specified, D14 is
"fallback-only" only in the sense that it does not run when the screen is
confident — it remains the sole completeness authority and will run on a
non-trivial minority of epochs. Making D14 *genuinely* rare (a true
fallback-only architecture) requires the §5.5 certificate; that re-evaluation
is deferred until the screen-fail rate is measured (§11).

### 5.5 Future research (NOT an implementation premise now)

A **root-exclusion / root-count certificate**: a Sturm / Budan sign-count or
interval-arithmetic bound on D14 over each *inter-band gap* that *proves* "no
undiscovered real root of D14 in this interval" *without* the degree-14
`__float128` Aberth solve. Cheaper than the oracle, it would promote the fast
planner + screen to a genuine completeness authority and demote D14 to a true
fallback. Flagged as a research candidate only; the decision to pursue it
depends on the §11 screen-fail-rate measurement.

---

## 6. Existing warmup vs proposed trajectory cache — correspondence

The user's constraint: **extend the existing warmup machinery, do not add a
new public `trajectory_jacobian()` API.** The existing infrastructure
(`python/lcbinint/warmup.py`, C++ `MagnificationExecutionPlan`) already
provides everything the proposed "trajectory-level classify_cells /
radial_events cache" needs a home for.

| existing warmup | proposed prepared-geometry cache | relationship |
|---|---|---|
| `LightCurve.warmup(times, params)` | same entry point | reuse — no signature change |
| `MagnificationExecutionPlan { method, resolution }` | `+ optional prepared-geometry handle` | **additive field**, default null = today |
| `WarmupGeometry` (frozen: src x/y, sep, q, ρ, caustic_distance, topology) | the validity key for prepared geometry | reuse verbatim |
| `compare_warmup_geometry()` → `WarmupDriftReport` (warn-only) | Layer-1 candidate **pre-filter** | reuse — a `warn == false` epoch becomes a *reuse candidate*, still subject to the cheap prepared-band validation in §8; it is **not** a completeness proof |
| `_binary_topology()` → close / wide / resonant per epoch | topology-change detector | reuse; any change drops that epoch's prepared geometry from the candidate set |
| `build_warmup_report()` per-epoch route/resolution calibration | per-epoch band structure / root-ordering calibration | same shape, richer payload |
| `build_jax_warmup_report()` compiled fixed plan + certification vs native + budget | same, plus prepared geometry frozen at compile time | extend |
| `JaxWarmupReport.execution_plan` retained by the owning `LightCurve` | prepared geometry retained the same way (immutable) | reuse the ownership model |
| traced-JAX proposal → `compare_warmup_geometry` returns `available = False` | prepared geometry also unavailable under tracing | inherit |
| `LightCurve` batched-epoch `*_preplanned` execution path | the loop that consumes prepared geometry per epoch | reuse — plan vector already threaded through `light_curve.cpp` / `lens_model.cpp` |

### 6.1 Two reuse types — one mechanism

* **(A) intra-trajectory reuse.** Epoch *k*'s prepared state seeds epoch
  *k+1*'s solve. Same epoch array, adjacent times, geometry drifts smoothly
  along the trajectory.
* **(B) inter-proposal warmup reuse.** The warmup anchor for epoch *k* seeds
  the *next parameter proposal*'s epoch *k* (HMC/MCMC).

Both are keyed by the **same** `WarmupGeometry` + `compare_warmup_geometry`
drift comparison (the *pre-filter*), then **both** run the same cheap
prepared-band validation (§8) before the cached structure is trusted, and both
feed the **same** prepared-state struct (§7). They are not implemented
separately.

When *both* a same-epoch (B) anchor and a previous-epoch (A) trajectory state
pass the drift pre-filter for epoch *k*: both are cheap seed candidates. Run
the fast march (§5.1) from whichever seed set has the smaller drift norm to the
current geometry; keep the other as the fallback seed if the first march's
screen fails, before paying a cold march. Selecting the seed is O(1) (a
drift-norm comparison); it does not cost a solve. In all cases the fast march
output is re-screened (§5.2) — a reused seed never bypasses the screen.

---

## 7. Concrete reuse-state data structure

Conceptual — **not** a new public type. Lives inside the warmup report.

```cpp
// One per epoch. Immutable once built at warmup; safe to share read-only
// across HMC chains / JAX device threads.
struct PreparedEpochGeometry {
    // --- validity key / provenance ---
    WarmupGeometryRow anchor;      // src(x,y), sep, q, rho, caustic_distance, topology
    ConditioningMargins margins;   // min |tangency regularity det| (= |phi_R*phi_thetatheta|),
                                   //   min band width, up/down march agreement, min |p4|
    enum { fast_march, d14_oracle, escalated } provenance;
    Status status;                 // OK | TOPOLOGY_UNCERTAIN | GRADIENT_UNRELIABLE

    // --- prepared geometry ---
    double r_max;
    small_vector<BandRow>      bands;   // (r_lo, r_hi, kind, n_crossings)
    small_vector<RootOrderRow> roots;   // per anchor radius: ordered quartic roots
                                        //   / (m,v) quadratic-factor pairs
    std::optional<D14EventSet> d14;     // only populated if the oracle ran
};
```

Two-layer structure (the user's requested split):

* **Layer 1 — immutable baseline anchors.** `std::vector<PreparedEpochGeometry>`
  built once by `build_warmup_report` / `build_jax_warmup_report`, owned by the
  report / `LightCurve`, **never mutated after construction** (no lazy fields,
  no mutable cache members). Concurrent evaluations share it by const
  reference.
* **Layer 2 — evaluation-local scratch.** `EpochGeometryScratch` = the live
  `QuarticWarm` / `(m,v)` cache / `RootContinuationWorkspace`, seeded from
  Layer 1 at the first radial node of each cell, discarded when the call
  returns. One per evaluation (per thread, per JAX trace). **No design that
  mutates Layer 1.**

`MagnificationExecutionPlan` gains `const PreparedEpochGeometry* prepared =
nullptr;` (or an index into the report vector). Additive; the existing
`{method, resolution}` contract is untouched; `nullptr` reproduces today's
behaviour exactly.

---

## 8. Validity / certification / fallback policy

Per-epoch fail-closed ladder:

```
1. WARMUP DRIFT PRE-FILTER (Layer 1 reuse-candidate selection)
     compare_warmup_geometry -> WarmupDriftReport
     epoch k's cached PreparedEpochGeometry is a REUSE CANDIDATE only if
         warn == false  AND  topology_changed == false
     otherwise -> drop Layer 1 for this epoch, recompute from scratch
                  (fast march, cold or from the nearest still-valid neighbour)

     *** This pre-filter is NOT a completeness proof. warn == false means
         "geometry drifted little enough to be worth trying to reuse", not
         "the cached band topology is still correct". A reuse candidate
         ALWAYS proceeds to step 2 below before its bands / root ordering
         are trusted. The existing scalar method/resolution warmup keeps
         its warn-only semantics unchanged; this candidate/validate split
         is scoped to the new PreparedEpochGeometry payload. ***

2. MANDATORY CHEAP VALIDATION OF THE REUSE CANDIDATE
     re-run the section 5.2 confidence screen against the CACHED structure:
       - 3 in-band probes at the current geometry still agree on
         (kind, crossing count) for every cached band
       - every cached band endpoint re-solved (3-iter tangency prefilter)
         still lands inside its cached bracket AND its tangency-regularity
         det |phi_R*phi_thetatheta| is still above threshold
       - cached root ordering at each anchor radius still matches a fresh
         cheap quartic solve (order + count)
     all agree -> reuse the cached bands / root ordering, provenance = fast_march (reused)
     any disagreement OR uncertain -> DISCARD the candidate, per-epoch
         recompute: fast march (section 5.1) -> screen (section 5.2) -> D14 if the screen fails

3. LOCAL FAST-MARCH CONFIDENCE SCREEN  (section 5.2, the 4 checks; for a fresh march)
     pass  -> use bands directly, provenance = fast_march
     fail  -> D14 oracle

4. D14 ORACLE  (radial_events, __float128) -- completeness authority
     clean      -> use, provenance = d14_oracle
     degenerate -> Status = TOPOLOGY_UNCERTAIN, FAIL CLOSED

5. PER-RADIUS RELIABILITY  (radius_terms.reliable)
     near-tangency / degenerate quartic / full circle / near-origin source
        -> Status = GRADIENT_UNRELIABLE   (value may still be OK; Jacobian row NaN'd)

Status is the worst of every stage and is never upgraded back to OK.
Failure is a status, never a silent approximation.
```

The existing algebraic backends and the existing holonomic backend stay in the
tree as independent experimental entry points for A/B comparison at every
phase; nothing is deleted.

---

## 9. Are public API changes really necessary?

**Findings from the code review:**

* Neither the holonomic nor the algebraic backend is wired into
  `FiniteSourceMagnifier` / `FiniteSourceMethod` today — both are standalone
  experimental functions driven by bench harnesses.
* `LightCurve.warmup()` / `WarmupReport` / `JaxWarmupReport` already provide
  the epoch array, the baseline anchor, the retained compiled plan, the drift
  comparator (`compare_warmup_geometry`), and the batched-epoch `*_preplanned`
  execution path.
* `MagnificationExecutionPlan` is already threaded through every `*_preplanned`
  entry point in `light_curve.hpp/cpp` and `lens_model.hpp/cpp`.

**Conclusion: no new public API is required for either direction.**

* **Direction 1 (unified engine):** add one internal
  `finite_source_binary(geometry, requested)` router + one new
  `FiniteSourceMethod` enum value + its `binary_mag` / `binary_mag_preplanned`
  cases. All internal. The existing `experimental_algebraic_boundary_*` and
  `holonomic::epoch_jacobian` entry points remain for A/B.
* **Direction 2 (warmup):** extend `MagnificationExecutionPlan` with the
  optional `PreparedEpochGeometry` handle (additive field), populate it inside
  `build_warmup_report` / `build_jax_warmup_report`. **No** new public
  function, **no** change to `warmup()`'s signature, **no** `trajectory_jacobian()`.
* The only *optionally* new public surface is a read-only diagnostic accessor
  on the warmup report (`report.prepared_geometry_provenance` /
  hit-rate counters) — nice-to-have, not required.

---

## 10. Implementation phases

Each phase keeps the standalone experimental backends and stays A/B-comparable;
each ends with a benchmark + 108-case parity (0 status changes) + `test_holonomic_m7`.

| phase | scope | gate |
|---|---|---|
| **0 (done)** | M8 holonomic fused pass | decision-20 met (2.14 ms median, 11.7×) |
| **1** | unify the numerical kernel: one `boundary_quartic`, one `aberth`, one warm-start state type shared by both backends (header move, no behaviour change) | bit-parity both backends |
| **2** | M1 value-only fast lane: exact-`m0` + multipole shortcut + adaptive-GK radial into the holonomic entry; `u = 0` value-only skips `F_half` and all Jacobian state | uniform value-only median ≥ 2× faster than current fused value; μ parity |
| **3** | fast planner + §5.2 confidence screen in front of D14; D14 is the completeness authority, invoked on screen-fail | band-discovery cost ↓; p90/p99 non-regressing; screen-fail (D14-invocation) rate measured |
| **4** | the mode router `finite_source_binary(geometry, requested)`; wire as a `FiniteSourceMethod` | all 4 modes parity + latency |
| **5** | prepared-geometry cache: Layer-1 struct in the warmup report, Layer-2 scratch, the §8 drift pre-filter + mandatory cheap validation | trajectory + HMC-proposal reuse win; false-reuse audit passes (forced caustic crossing recomputes) |
| **6** | M3 JVP mode (single-direction ForwardJet / fused single direction) | JVP cheaper than one FD column |

---

## 11. Benchmark plan

Harness: `bench_holonomic_m7` (108 cases, best-of-200, `taskset -c 0-7`, log
`uptime` + affinity **before** timing). Isolated build (`build-holonomic-m7/`),
never touch the shared `_lcbinint.so`.

**Two distinct benchmark families — never merged into one table:**

* **(A) Algorithm micro-benchmark — multipole / hexadecapole shortcut OFF.**
  Every one of the 108 cases runs the full boundary integral. This isolates the
  *kernel* cost (band discovery + quartic + arcs + radial/angular rules +
  Jacobian state) with no router masking. All the V0/V1/V2 numbers in §13 and
  all per-mode kernel comparisons below are of this family.
* **(B) Router-inclusive production benchmark — shortcut ON.** The real
  end-to-end cost, ~74/108 cases resolved by the multipole shortcut. Reported
  as its own separate table, and only once a mode is wired through the router
  (phase 4+). Do **not** compare (A) and (B) numbers directly.

**Per mode — separate median / p90 / p95 / p99 (family A unless noted):**

* **M1** uniform value-only: vs `binary_mag` point/hex path, vs algebraic
  value-only, vs current `epoch_jacobian` value.
* **M2** value+LD: vs current holonomic `F_half`, vs algebraic moment series.
* **M3** value+JVP: vs one FD column.
* **M4** value+5-Jacobian: **decision-20 gate stays median ≤ 7.44 ms**, p90/p95/p99
  non-regressing; vs current holonomic 2.14 ms; analytic-Jacobian quality
  (grad L2 rel, dμ/du rel).

**Planner:**

* band-discovery microbench: fast march vs D14 `__float128`, per geometry class;
* fraction of the 108 cases whose §5.2 screen fails (→ D14 authority invoked);
* fraction that fail closed (`TOPOLOGY_UNCERTAIN`);
* fraction of reuse candidates rejected by the §8 step-2 validation.

**Cache:**

* N-epoch trajectory, cold vs Layer-1-warm, per-epoch amortised cost;
* HMC chain of K proposals × N epochs: per-proposal amortised cost + Layer-1
  hit rate;
* **drift-gate false-reuse audit:** force a caustic-topology crossing at one
  mid-trajectory epoch, assert that epoch recomputes and the rest reuse.

**Correctness (every phase):** 108-case parity — μ rel, grad L2 rel, dμ/du rel,
**status-change count = 0**; `test_holonomic_m7` full pass.

**Fail-closed audit:** pathological set (caustic cusp, near-origin source,
extreme q, deliberately thin band) → assert `Status ≠ OK`, never a silent
number.

---

## 12. Correctness / thread-safety / autodiff risks

### Correctness

* **M1 exact-`m0` completeness.** `m0` from interval lengths is only correct if
  the interval classification is complete. A thin band missed by the fast
  march makes `m0` silently low. Mitigation: §5.2 check (4) (resolution
  margin) + the D14 fallback.
* **Inconsistent value/Jacobian grid.** The analytic IFT-θ derivative in
  `radius_terms` assumes the value used the *same fixed* GC grid. An adaptive
  value + fixed-grid Jacobian is an inconsistent pair. Rule: **when a Jacobian
  or JVP is requested, the value uses the Jacobian's grid** (fixed GC `n_r`).
* **Moment-series vs GK disagreement.** The closed-form `algebraic_moment_series`
  fast path must fail closed to GK on disagreement, never average the two.
* **D14 float128 vs MXCSR.** `radial_events`' `__float128` Aberth is software
  (libquadmath), unaffected by FTZ/DAZ — but its *double* pre-search runs under
  the guard. Keep `ScopedFlushDenormals` scoped to the whole epoch, as today.
* **Odd boundary-root count.** Both backends already treat an odd physical
  root count as a certification failure (retry cold, then fail closed). The
  unified kernel must keep that — an odd count is never a valid arc set.

### Thread-safety

* The algebraic backend uses `static thread_local` scratch (`physical_roots`,
  `angles`, `retry_roots`) and RAII-scoped `thread_local` globals
  (`active_root_workspace`, `active_mv_cache`, `active_integrand_cache`). The
  unified engine **must** keep this discipline — no `static` non-`thread_local`
  mutable state anywhere in the kernel.
* **Layer 1 `PreparedEpochGeometry` must be genuinely immutable** after warmup
  (no lazy init, no mutable members) so concurrent HMC chains and JAX device
  threads share one report safely. All mutation is Layer-2, per-evaluation.
* `ScopedFlushDenormals` mutates the per-thread MXCSR and RAII-restores it.
  Safe across a reused thread pool as long as no path leaves the guard's scope
  with MXCSR still modified.

### Autodiff / JAX

* **Tracing.** `compare_warmup_geometry` returns `available = False` for
  non-concrete proposals; prepared geometry inherits this — under `jax.jit`
  the plan (and cache) is a compile-time constant, and Layer-2 scratch must
  never capture tracers.
* **One parameter convention.** Holonomic Jacobian is wrt `(xs, ys, ρ, q, a)`;
  the algebraic `ForwardJet` is wrt Mapping-B params. The unified engine must
  expose **one** convention (recommend the holonomic user convention) and map
  internally, or the JVP/Jacobian seed semantics diverge between modes.
* **Frozen crossing angles.** The algebraic `ForwardJet` re-pass evaluates the
  boundary at crossing angles *frozen from the primal*. If the prepared-geometry
  cache hands back stale crossing angles after a drift, the derivative is
  silently wrong. The §8 step-2 mandatory validation must re-check the cached
  **root ordering** (order + count vs a fresh cheap quartic solve), not just the
  band bounds — the drift pre-filter alone does not.
* **Per-epoch, not per-trajectory, invalidation.** An HMC proposal that crosses
  a caustic-topology boundary for even one epoch must invalidate *that epoch
  only*. The gate is per-epoch.
* **Measure-zero transitions.** `F'(θ_k) ≈ 0`, singular band-endpoint tangency
  system, two-form derivative disagreement over `grad_tol` → NaN the row. Both
  backends do this today; the unified engine keeps it as `GRADIENT_UNRELIABLE`.

---

## 13. Next experiment — the V0 / V1 / V2 three-variant micro-benchmark

**Isolated bench only. No kernel unification, no wiring, no API change, no
cache.** One bench executable builds three variants that share *everything*
except the two things under test, so the speed-up cause is separable. Kernel
unification / fast planner / warmup cache implementation order is decided by
the user *after* seeing these numbers.

### 13.1 The three variants

All three run the **same** `classify_cells` band structure, the **same** cell
list, the **same** per-cell `QuarticWarm` warm-start, the **same**
`boundary_quartic` / root path, and the **same** `polish_endpoint` on
`φ(R,θ)=0`. The multipole / hexadecapole shortcut is **OFF** for all three
(benchmark family A, §11) — every case runs the full boundary integral.

| variant | radial rule | angular / derivative work | what it is |
|---|---|---|---|
| **V0** | fixed GC64 per cell | value **+ fused 5-Jacobian + ∂μ/∂u** (`F0`, `F_half`, `dF0[5]`, `dF_half[5]`, IFT-θ endpoint derivs, `d√φ/dP`, internal→user chain rule) | **the baseline** — today's `epoch_jacobian` exactly |
| **V1** | fixed GC64 per cell (**identical to V0**) | **value only**: accumulate `F0` from arc-interval lengths (`R·(θ_leave−θ_enter)`); skip `F_half`, `dF0`, `dF_half`, the `√φ` angular loop, the chain rule, and every derivative accumulator / state field | V0's kernel, uniform value-only, fixed grid |
| **V2** | **adaptive one-panel-first Gauss–Kronrod** (algebraic-style: one-panel estimate per band, bisect only if the panel error is over budget) | same value-only kernel as V1 | V1 with only the radial rule swapped |

The GK rule and adaptive bisection are reimplemented inside the bench
(`integrate_outer_adaptive` / `gauss_kronrod21()` are in an anonymous namespace
in `algebraic_boundary.cpp` and not exported).

### 13.2 What the differences isolate

```
V0 − V1  =  marginal cost of the LD half-integral + the full Jacobian state
            (F_half, the √φ angular loop, dF0/dF_half accumulation,
             IFT-θ endpoint derivatives, d√φ/dP, internal→user chain rule)
            — everything else held identical.

V1 − V2  =  marginal cost of radial SCHEDULING
            (fixed 64-node Gauss–Chebyshev  vs  adaptive one-panel-first GK)
```

**Caveat to report alongside V1 − V2:** the adaptive GK rule does not only
change *when* nodes are placed — it also changes the *node count* and the node
*distribution*, and a non-monotone / non-Chebyshev node sequence can weaken the
`QuarticWarm` warm-start (which assumes the near-monotone GC node march). So
V1 − V2 is "scheduling **and** its knock-on effects on node count and
warm-start", not a pure scheduling delta. The per-variant metrics below
(radial-node count, quartic-solve count, warm-hit rate) are exactly what
exposes this — the experiment stays well-posed, but the report must not claim
V1 − V2 is scheduling alone.

### 13.3 Metrics — on the 108 full-solve cases

Reported for **each** of V0, V1, V2:

* wall time: **median / p90 / p95 / p99** (best-of-200 per case, `taskset -c 0-7`,
  `uptime` + affinity logged before the run);
* **radial-node count** (total quartic-solve sites): mean + distribution;
* **quartic-solve count** (cold + warm, i.e. Aberth invocations): mean;
* **warm-hit rate** (`QuarticWarm.warm_hits / (warm_hits + cold_falls)`);
* **μ parity**: max \|Δμ/μ\| of V1 and V2 vs V0 (the fused baseline value).

Plus the two derived deltas (`V0−V1`, `V1−V2`) at median / p90 / p95 / p99, and
the fraction of cases where each delta is positive (a per-case win, not just an
aggregate one).

### 13.4 Separate table — router-inclusive production benchmark

Benchmark family B (§11), shortcut **ON**, reported as its own table and
explicitly **not** compared line-to-line with 13.3. This is deferred: it only
becomes meaningful once a value-only mode is actually wired through
`finite_source_binary` (phase 4). For this experiment it is a placeholder row
noting "deferred to phase 4".

### 13.5 Why this experiment, and what it decides

It tests the central premise of Direction 1 — *"value-only must not pay
Jacobian-state cost, and that saving is large"* — while **separating** that
claim from the independent claim that adaptive radial scheduling helps. Using
only code that already exists on both sides. Outcomes:

* **V0 − V1 large, V1 − V2 small** ⇒ the Jacobian-state skip is the whole M1/M2
  win; adaptive GK is not worth the warm-start risk → keep fixed GC for every
  mode, prioritise the mode router (§10 phase 2/4) over the fast planner.
* **V0 − V1 small** ⇒ value-only is *not* materially cheaper than the fused
  pass → the §10 mode-router priority order changes; M4 stays the primary
  target and M1/M2 get de-prioritised.
* **V1 − V2 large and favourable** (with warm-hit rate holding up) ⇒ adaptive
  radial scheduling is a real win → the fast planner + adaptive radial rule
  moves up the implementation order.
