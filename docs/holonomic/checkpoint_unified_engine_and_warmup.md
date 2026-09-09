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

---

## 0. Fixed design policy (2026-09-08) — GOVERNING

The user has fixed this as the design policy for all further work. It overrides
any earlier framing that treated algebraic and holonomic as competing backends.

0.1 **One common geometry / root kernel at the centre.** Boundary-quartic
    construction, scaling/balancing, quartic root solve, neighbouring-R warm
    start, `(m,v)` continuation, root correspondence, endpoint polish / residual
    certification, conditioning detection, FTZ/DAZ, fail-closed fallback are
    **one** numerical kernel shared by every mode. "algebraic solver" and
    "holonomic solver" each running their own copy of the same quartic is
    **not** the final form.

0.2 **Work is decided by requested output, not by algorithm name.** No
    "pick algebraic / pick holonomic" dispatch. Modes: uniform value-only,
    limb-darkened value-only, value+JVP, value+full Jacobian. Answer-only is the
    shortest path; each added output costs only its own marginal work. Uniform
    value-only uses algebraic's strength (compute `F0` directly / cheaply).
    Jacobian uses holonomic's proven fused analytic derivative — **no second
    geometry solve after the value**.

0.3 **Topology discovery: the cheap algebraic method is the normal path.**
    fast algebraic discovery → cheap validation / confidence checks → use
    directly if safe → **D14 only if uncertain**. D14 is **not** deleted: it is
    the correctness oracle / fallback for difficult geometry, near-degeneracy,
    or questionable completeness. A cheaper-than-D14 root-exclusion / root-count
    certificate is permitted as *future research*, not a current premise.

0.4 **Keep both sides' good parts; force neither structure onto the other.**
    From algebraic: cheap band discovery, uniform value-only lightness, adaptive
    work allocation, moment-series fast path, `(m,v)` continuation. From
    holonomic: fused LD moments, fused analytic Jacobian / JVP, bounded
    difficult-case behaviour, strong reliability gating, quartic warm-start,
    FTZ/DAZ, D14 fallback / completeness machinery.

0.5 **warmup / trajectory reuse integrates into existing infrastructure.** No
    new independent public API as a premise. Extend `LightCurve.warmup()`, the
    per-epoch execution plan, and the geometry-drift machinery. Future: a
    per-epoch prepared geometry holding bands/cells, root ordering, quartic seed
    state, conditioning margins, optional D14 info; neighbouring-epoch reuse and
    neighbouring-HMC/MCMC-proposal reuse handled by the **same** internal
    mechanism. Cached topology is **never** blindly reused on a heuristic drift
    check alone — it must pass a cheap validation; if uncertain, recompute that
    epoch only (fail closed).

0.6 **Optimisation decisions are driven by profile and benchmark.** Measure
    latency, node / root-solve count, warm-start hit rate, D14 invocation rate,
    accuracy, Jacobian reliability, p90/p95/p99 tail, pathological geometry.
    Proceed from highest measured impact first. A new idea that beats this
    policy *with data* is acceptable. The goal is a production-quality
    finite-source engine that is fast even value-only, cheap when LD is added,
    low added cost when derivatives are requested, and does not break on
    difficult geometry — **not** "make holonomic win" or "keep algebraic".

**Implementation order, intermediate experiments, and internal data-structure
details are delegated to the implementer, decided from the current code and
profile.** §0.7 records the order chosen; §14 records the E1 measurement it
rests on.

### 0.7 Chosen implementation order (profile-driven; supersedes §10's numbering)

E1 (§14) measured where a value-only holonomic epoch spends its time:
**`radial_events` / D14 is 74% of the epoch** (median 1.02 ms of 1.60 ms), grid
probes another 10%, the radial integration pass only 14%, and the cheap
`quartic_topology` probes 2%. The fast-planner *ceiling* — D14 + grid probes —
is **84% of the epoch**; the *floor* it cannot touch is **16%** (median
0.26 ms). This dominates every other lever (the value-only Jacobian-state skip
was 12% of the *old* epoch; adaptive GK 7% with a parity regression).

| order | phase | why here | gate |
|---|---|---|---|
| **A** | **fast planner + §5.2 confidence screen in front of D14** (was §10 phase 3) | attacks 84% of the measured epoch cost; also directly policy §0.3 | band-discovery cost ↓ ≥ 3×; 108-case topology parity vs `classify_cells` (0 disagreement on the adopted band set); D14-invocation rate measured; p90/p95/p99 non-regressing; full-Jacobian decision-20 still met |
| **B** | unify the numerical kernel (was §10 phase 1): one `boundary_quartic`, one `aberth`, one warm-start state; port `solve_radial_tangency` + point-source seed solver into a shared header | now scoped by what the planner actually needs shared; no perf claim | bit-parity both standalone backends; `test_holonomic_m7` full pass |
| **C** | mode router `finite_source_binary(geometry, requested)` + `FiniteSourceMethod` wiring (was §10 phase 4) | makes M1–M4 real; unlocks benchmark family B (multipole shortcut ON) | all 4 modes parity + latency; family-B table |
| **D** | M1 / M2 value lanes (was §10 phase 2): exact-`m0`, LD moment series, adaptive-GK radial | *after* phase A the value-only skip is ~50% of the shrunken epoch, not 12% — worth more, but needs the router | uniform value-only median ≥ 2× the fused value; μ parity; moment-series fails closed to GK |
| **E** | prepared-geometry cache (was §10 phase 5): Layer-1 struct in the warmup report, Layer-2 scratch, §8 drift pre-filter + mandatory cheap validation | the planner's seeds / bands / margins are exactly the cache payload | trajectory + HMC-proposal reuse win; forced-caustic-crossing false-reuse audit passes |
| **F** | M3 JVP mode (was §10 phase 6) | smallest independent slice | JVP cheaper than one FD column |

Every phase keeps the standalone experimental `algebraic_boundary_*` and
`holonomic::epoch_jacobian` entry points for A/B; nothing is deleted; failure is
a `Status`, never a silent approximation; each phase ends with a committed
checkpoint.

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

### 13.6 Results (2026-09-08)

Bench: `tests/holonomic_cpp/bench_v0v1v2.cpp`, isolated build
(`build-holonomic-m7/bench_v0v1v2`), 108 full-solve cases, best-of-200,
`taskset -c 0-7`, load average ≈ 11 (64-core box) logged before the run.
Raw: `evidence/holonomic/v0v1v2_bench.txt`.

| variant | median | p90 | p95 | p99 | radial nodes (mean) | quartic-solve sites | warm-hit rate |
|---|---|---|---|---|---|---|---|
| **V0** (fused value + 5-Jac) | 2.144 ms | 3.822 | 3.959 | 4.081 | 239.4 | 239.4 | 0.984 |
| **V1** (V0 kernel, value-only, fixed GC64) | 1.614 ms | 3.418 | 3.468 | 3.710 | 239.4 | 239.4 | 0.984 |
| **V2** (V1 kernel, adaptive one-panel-first GK) | 1.379 ms | 3.162 | 3.357 | 3.582 | 116.7 | 116.7 | 0.933 |

Derived deltas (per-case, then percentiles):

| delta | median | p90 | p95 | p99 | cases improved |
|---|---|---|---|---|---|
| **V0 − V1** — LD half-integral + full Jacobian-state marginal cost | +0.256 ms | +0.831 | +0.971 | +1.181 | 108 / 108 |
| **V1 − V2** — radial scheduling (incl. node-count + warm-start knock-on) | +0.117 ms | +0.314 | +0.342 | +0.529 | 98 / 108 |

μ parity (uniform μ = F0 / (π ρ²), vs V0's F0):

* **V1 vs V0: max \|Δμ/μ\| = 0.0 — bit-identical.** Confirms V1 is V0's kernel
  exactly, only the derivative / F_half work removed.
* **V2 vs V0: max \|Δμ/μ\| = 1.1 × 10⁻⁴** on case `very-close` (V0 status OK),
  median 6 × 10⁻¹⁴. The algebraic one-panel-first error gate accepted a panel
  whose true error was ~100× the `|Kronrod − Gauss|` estimate — a real failure
  mode of the one-panel heuristic near close-topology geometry, above the
  `rel_tol = 1e-6` target.

Status: V0 / V1 / V2 all 104 / 108 OK (same 4 extreme-q non-OK cases);
**status-change count vs V0 = 0** for both V1 and V2.

**Reading.**

1. *"Value-only must not pay Jacobian-state cost"* is **true but the win is
   modest at the median** — 12% (0.26 ms of 2.14 ms) — and **tail-heavy**: 22%
   at p90, 29% at p99, every case improving. It is not the "large" win the
   premise assumed, because on the holonomic path `classify_cells` (always-on
   D14 `__float128` band discovery + probes) is the dominant fixed cost, and no
   variant here touches it.
2. Adaptive radial scheduling (V1 → V2) adds a further **7% at the median** but
   (a) breaks bit-exact parity (the 1.1e-4 outlier), (b) **regresses 10 / 108
   cases** (the one-panel + re-integrate double evaluation and the priority-queue
   subdivision can cost more than the fixed grid), and (c) drops the warm-hit
   rate 0.984 → 0.933 despite halving the node count — the single-slot
   `QuarticWarm` cannot follow the non-monotone GK node order. A faithful port
   would need the algebraic `RootContinuationWorkspace` nearest-parent-node
   warm start.
3. The **real latency floor is `classify_cells` / D14**, untouched by both
   deltas. If the objective is median latency, this is direct evidence to
   prioritise the **fast planner (§10 phase 3)** — replacing always-on D14
   `__float128` with the fast march + §5.2 screen — over both kernel
   unification and the value-only lane. The value-only lane (phase 2) is a
   real but second-order win; adaptive GK as specified is marginal and carries
   a parity regression, so it should not lead.

This is input to the user's decision on implementation order; it is **not**
itself an implementation step.

---

## 14. E1 — planner cost-split (2026-09-08)

Bench: `tests/holonomic_cpp/bench_planner_split.cpp`, isolated build
(`build-holonomic-m7/bench_planner_split`), 108 cases, best-of-200, `taskset -c
0-7`, load average ≈ 9.9 logged before the run. `classify_cells` re-implemented
with per-stage `steady_clock` splits (behaviour identical — same events, merge,
3-fraction probe, escalation rule). The "best-of" picks the fastest full epoch
per case and reports that epoch's stage breakdown.

Purpose: §13.6 concluded from *inference* that `classify_cells` / D14 is the
latency floor. E1 measures it directly, and separates the part a fast planner
**replaces** (D14 + the 512/3072 grid probes) from the part the §5.2 screen
**keeps** (`quartic_topology` ×3/cell) and the part no planner touches (the
radial integration pass).

| stage | median | p90 | p99 | mean | share of epoch |
|---|---|---|---|---|---|
| **(A) `radial_events` / D14** | 1.018 ms | 2.797 | 2.831 | 1.446 | **74%** |
| (B) grid probes (512 + 3072 escalations) | 0.147 ms | 0.297 | 0.580 | 0.189 | 10% |
| (C) `quartic_topology` ×3/cell (screen keeps) | 0.037 ms | 0.056 | 0.065 | 0.039 | 2% |
| event merge / bounds | 0.0005 ms | — | — | 0.0005 | 0% |
| (D) radial integration pass (V1 value-only) | 0.224 ms | 0.489 | 0.725 | 0.282 | 14% |
| `classify_cells` total | 1.227 ms | 3.004 | 3.284 | 1.674 | — |
| **full epoch** | 1.599 ms | 3.418 | 3.708 | 1.958 | — |

Fast-planner **ceiling vs floor**:

| | median | p90 | p99 | mean | share |
|---|---|---|---|---|---|
| **saveable** (A)+(B) | 1.200 ms | 2.954 | 3.228 | 1.635 | **84%** |
| **floor** (C)+(D) | 0.261 ms | 0.538 | 0.781 | 0.321 | 16% |

Realistic planner-path epoch ≈ floor + fast march (~0.03 ms, algebraic-side
measured) + the §5.2 screen overhead ≈ **~0.3–0.4 ms median**, i.e. a
**~4–5× reduction** of the value-only epoch and a comparable cut to the
full-Jacobian epoch (V0 2.14 ms → plausibly ~0.8–1.0 ms), *if* the screen-fail
(D14-invocation) rate stays low.

Structure: mean **12.7 cells/epoch**; **34/108 cases** hit ≥ 1 `arcs_at(3072)`
escalation (mean 0.43/epoch). Those 34 are the cases most likely to fail the
§5.2 screen and fall through to the D14 oracle — the D14-invocation rate is the
phase-A gate metric.

**Decision:** phase A (fast planner) leads. Confirmed by direct measurement, and
aligned with policy §0.3. Raw: `evidence/holonomic/planner_split_bench.txt`.

---

## 15. Phase A steps 1–2 — seed-anchored fast band planner (2026-09-08)

Two new headers, both **additive** (no existing solver touched; `classify_cells`
stays the oracle):

* `point_images.hpp` — binary point-source image solve (Witt & Mao degree-5
  polynomial). **Two-tier:** a `double` complex Aberth pass, its roots verified
  by residual against the *original non-holomorphic* lens equation; a clean odd
  set (3 or 5, worst residual ≤ `1e-9`) is returned directly. Otherwise the
  build + solve is redone in `__float128` (the extreme-q planet cluster
  collapses to a 2-image count in `double`). Seeds only anchor the radial
  march — band *edges* come from independent bisection — so the looser `double`
  accept is safe. Committed `5093876` was `__float128`-only; the two-tier split
  is this checkpoint.
* `fast_bands.hpp` — seed-anchored radial band planner. From each unique
  point-source image radius, a **neighbour-bounded** step-doubling march
  outward/inward on `has_image(R)` (= `arc_intervals(R)` yields a non-empty
  φ>0 arc set), then boolean bisection of the bracket. Each seed's march is
  bounded by the adjacent seed radii. Raw brackets are merged, then a
  **solidity pass** re-probes any merged band that absorbed >1 seed bracket or
  had a step-doubled march: interior probes per seed-gap at ~ρ/50 resolution,
  splitting at image-free notches the march tunnelled. `has_image` resolves the
  `t = tan(θ/2)` chart singularity (p4≈0 → `kDegenerate`) on a 2048-grid, same
  as `quartic_topology`; a genuine odd crossing count there → `UNCERTAIN` →
  fail closed.

### 15.1 `classify_cells` razor-band finding (revises the phase-A gate)

The §0.7 gate says "0 disagreement on the adopted band set". Taken literally
against `classify_cells` this is **both unachievable and undesirable**:

`classify_cells` **silently mislabels razor-thin image bands as `kEmpty`**.
When the boundary quartic resolves a real image arc of angular width below the
3072-grid step (~2e-3 rad), the grid512 cross-check overrules the correct
quartic result and the grid3072 escalation confirms the wrong `Empty`
(`cells.hpp:106–120`, `cs = Status::OK` — no `TOPOLOGY_UNCERTAIN`). Verified
these bands are real: `phi_lens` margin ~0.99 at the arc midpoint, independent
point-source-solver agreement, boundary-quartic root pair. Their contribution
to μ is ~1e-7…1e-9 relative — below M7 reference tolerance — so this is not an
accuracy bug in the incumbent, but it means `fast_bands` legitimately reports
**more** bands than the oracle on 27/82 adopted cases (56 "razor extras"
total). `fast_bands` is *strictly more correct* here.

**Reframed parity criterion (significance-aware):** a fast band is
*significant* if its widest arc over 3 interior radii is ≥ `kRazorAng = 3e-3`
rad. Parity is required only on the significant subset:
significant-band **count** match, significant ref bands **covered**
(overlap-based), and significant band **edges** agree to < 5e-3 of band width.
Razor extras are logged as a strict refinement, not a disagreement.

### 15.2 Bench result — `bench_fast_bands` (108 cases, best-of-200, load ~12)

```
fast_bands reliable          : 82 / 108   (26 rows / 13 named cases fail closed -> D14)
significant-band count match  : 82 / 82
significant ref bands covered : 82 / 82
significant band edges agree  : 82 / 82    (median 5.7e-8, p90 7.9e-7, max 9.6e-7 rel)
razor extras                  : 56 total   (sub-grid arcs classify_cells drops)
classify_cells UNCERTAIN      : 4 / 108

point_images + fast_bands : median 0.353 ms   p90 0.785   p99 2.567   mean 0.532
classify_cells (D14 path) : median 1.262 ms   p90 3.025   p99 3.280   mean 1.692
band-discovery speedup    : 3.57x median / 3.18x mean   (p90 3.85x, p99 non-regressing)
has_image calls / epoch   : median 152   p90 242   max 458
```

* **Gate "band-discovery cost ↓ ≥ 3×": MET** (3.57× median, 3.85× p90).
* **Gate "topology parity, 0 disagreement on the adopted band set": MET** under
  the §15.1 significance-aware reading (82/82 significant count + coverage +
  edges); 56 razor extras documented as refinement.
* **Gate "p90/p95/p99 non-regressing": MET** — fast path is faster than the
  D14 path at every percentile measured (p99 2.567 < 3.280).
* **D14-invocation rate: 24% (26/108)** — entirely the extreme-q cases
  (q ≲ 3e-4 with the source near the planet) where even `__float128`
  point-source gives an even image count. This *is* the §5.2 "seeds unreliable
  → D14" screen, already fail-closed. Non-extreme-q cases: 0% invocation.
* full-Jacobian decision-20: not re-run this step (planner not yet wired into
  `epoch_jacobian`); that is phase-A step 4.

Caveat: the 13 D14-bail cases pay `point_images` (`__float128`, p99 ~2.5 ms)
*before* falling through to D14, i.e. a small absolute regression on those
epochs. Phase-A step 3 adds a cheap upfront extreme-q predicate to skip the
seed solve on geometries that cannot succeed.

Raw: `evidence/holonomic/fast_bands_bench.txt`. Not yet committed to the
engine — `classify_cells` / D14 remains the only wired path.

### 15.3 Remaining phase-A work

* **step 3** — §5.2 local confidence screen + extreme-q fast-fail predicate;
  measure screen-fail rate. **DONE 2026-09-08 — see §16.** (Screen built; D14
  rate 24.1% = the extreme-q bail set, 0 new failures. Extreme-q predicate
  REJECTED as unsafe, §16.5.)
* **step 4** — wire the planner as an alternative discovery path behind a flag
  in the value-only and full-Jacobian epochs; full-epoch bench both modes;
  108-case parity (0 `Status` changes); decision-20 re-check; commit.
  **DONE 2026-09-08 — see §17.** (Flag `HOLO_FAST_PLANNER`, OFF by default;
  default build byte-identical, reference harness 0 failures. `panel_cuts`
  cheap subdivision + fail-closed per-sub-cell guard. Adoption 24.1%,
  0 `Status` changes, decision-20 PASS. Not yet a viable default — the
  `physical_complex` / fold-edge residual and the p99-from-fallback-prelude
  structural cost motivate phases D & E.)

---

## 16. Phase A step 3 — §5.2 local confidence screen (2026-09-08)

New file `src/lcbinint/magnification/holonomic/fast_bands_screen.hpp`
(header-only, `fast_bands_screen(pf, fb, seeds) -> FastBandsScreen`). New bench
target `bench_fast_bands_screen`. Nothing wired into the engine yet —
`classify_cells` / D14 is still the only path the solver calls.

### 16.1 What the screen is

A **local confidence gate**, not a completeness proof (§5.4/§5.5). It governs
*only* whether the D14 oracle is invoked for an epoch; it never decides that
completeness holds. Any epoch whose screen does not pass goes to D14; any epoch
D14 cannot resolve fails closed. It is applied on top of the §15 seed-anchored
`fast_bands` planner, over that planner's **significant** band subset (arcs wide
enough — ≥ 3e-3 rad — to move μ at the M7 reference tolerance; razor bands are
not screened, §15.1).

### 16.2 The four checks (deviations from the §5.2 sketch, all forced by the
available primitives — `phi.hpp` exposes no 2nd derivatives, `radial_events`
*is* the D14 core so it cannot be a "cheap independent check")

| # | §5.2 sketch | as built | why the deviation |
|---|-------------|----------|-------------------|
| 1 | "uniform in-band `quartic_topology` ×3" | **no interior kEmpty**: `quartic_topology` at 7 interior fractions of each significant band, each must be kFull or kArcs-even. Crossing *count* may vary. | the M7 moment integrator re-derives the arc set at every quadrature radius (`epoch_jacobian.hpp`), so a 2→4→2 arc split inside a band (source straddling a caustic fold) is harmless; only a real image-free sub-region breaks the integration partition. |
| 2 | "tangency regularity — `det ∂G/∂(R,θ) = φ_R·φ_θθ`" | **fold half-width ratio**: `hw(4δ)/hw(δ)` from `arc_intervals`, must lie in `[1.55, 2.35]` (regular fold = 2.0; ≈1.41 = cusp/higher contact; ≫2 = two bands merging). | `phi.hpp` has no φ_θθ / φ_R. The half-width growth rate is the scale-free equivalent and needs only `arc_intervals`. |
| 3 | "up/down march band-count agreement" | **seed coverage** (probe-free): every reliable point-source seed radius whose own local arc is significant must fall inside some planner band. | a blind independent radial recount cannot cheaply match the seed-anchored planner's resolution (rand000 / on-axis-off traces, §15). The necessary condition "source centre ∈ disk ⇒ every point image z_i ∈ image region ⇒ |z_i| ∈ some band" is both cheaper (O(seeds·bands), 1 `arc_intervals` call/seed) and stronger against the failure that matters (a dropped band). |
| 4 | "resolution margin" | **complement scan** (runs unconditionally): every inter-band gap wider than 2× the edge inset, plus below-innermost and above-outermost, swept linearly and must be image-free. Sub-0.3ρ gaps skipped (a newborn band cannot hide there). | originally gated on a straddle indicator; fault injection (`dbg_seedless`) showed the gate leaked adversarial seed-less-band drops, so it is now always on. |

`Status` is worst-of / fail-closed throughout.

### 16.3 Measured — `bench_fast_bands_screen` (108 epochs, best-of-200, load ~11.8)

```
planner bail (seeds/fb unreliable) : 26  (24.1%)
screen PASS                        : 82  (75.9%)
screen FAIL                        : 0
--> D14 oracle invocation rate     : 26 / 108  (24.1%)
screen FAIL by check               : all zero
worst fold ratio  : min 1.661  median 1.930  p90 1.937  max 2.100   [window 1.55..2.35]
screen probe calls: median 82  p90 115  max 129

seeds + fast_bands + screen : median 0.448  p90 0.871  p95 1.005  p99 1.081  ms
classify_cells (D14 path)   : median 1.262  p90 3.025  p95 3.068  p99 3.289  ms
BLENDED (screen + D14 on the 24.1%) : median 0.660  p90 3.159  p95 3.322  p99 3.931  ms
speedup vs D14 (median)     : screen-only 2.82x   blended 1.91x
```

Raw: `evidence/holonomic/fast_bands_screen_bench.txt`.

### 16.4 Gate assessment (§0.7 Phase A)

* **D14-invocation rate measured: 24.1%** (26/108) — *exactly* the extreme-q
  bail set from §15 (13 named cases × 2 u-values), where even `__float128`
  point-source returns an even image count. The screen adds **zero** new
  failures: all 82 planner-reliable epochs pass all four checks. Non-extreme-q
  invocation rate: 0%.
* **Band-discovery cost ↓ ≥ 3×**: the discovery *primitive* (`fast_bands`
  alone) is still 3.5–3.6× (§15.2, re-confirmed this session). The screen costs
  ~0.10 ms median (0.35 → 0.45) / ~0.28 ms p99, trading that for a 75.9%
  D14-skip. Screen-inclusive speedup is **2.82× median**. The blended path
  (screen always + D14 on the 24.1% that don't pass) is **1.91× median** but
  its p90+ tail *regresses* vs D14 alone, because the 26 bail epochs now pay
  `point_images` + a bailed `fast_bands` *before* D14. → **step-4 wiring must
  route the extreme-q set to D14 without first running the planner** (see 16.5).
* **p90/p95/p99 non-regressing** (screen-only path): MET — 0.87 / 1.00 / 1.08
  ms, well under the D14 path at every percentile.
* **108-case topology parity, 0 disagreement on the adopted band set**: MET
  under the §15.1 significance-aware reading (82/82 significant-band count +
  coverage + edges from §15.2; screen adds 0 FAIL, so the adopted band set is
  unchanged).
* **full-Jacobian decision-20**: not re-run — planner still not wired into
  `epoch_jacobian` (step 4).

### 16.5 The extreme-q fast-fail predicate — REJECTED as unsafe

The §15.3 plan called for "a cheap upfront extreme-q predicate to skip the seed
solve on geometries that cannot succeed". **This cannot be done safely on q
magnitude.** `dbg_tier` over the 108 cases: the double→`__float128` escalation
is genuinely load-bearing and q does **not** separate quad-rescuable from
quad-unrescuable geometries —

| case | q | double | `__float128` |
|------|------|--------|--------------|
| rand031 | 6.5e-8 | n=2 | **n=3, rescued** |
| rand015 | 7.0e-7 | n=2 | **n=3, rescued** |
| rand024 | 4.0e-7 | n=2 | **n=3, rescued** |
| rand016 | 2.2e-7 | n=2 | n=2, bails |
| rand011 | 2.0e-7 | n=2 | n=2, bails |

rand031 at q=6.5e-8 is rescued while rand016 at *larger* q=2.2e-7 bails. Any
q-threshold that skipped tier 2 would wrongly send the rescued cases to D14 and
lose their screen-pass. The two-tier solve's own **odd-count + residual gate**
(`point_images.hpp`) already *is* the correct predicate, and the `__float128`
Aberth is not a measurable cost problem: bail-epoch wall time is dominated by
the *subsequent* D14 call (p90 3.03 ms), not the quad solve. **No predicate
added.** The step-4 routing win is structural (don't run `fast_bands` when the
seeds are known-unreliable → straight to D14), not a q heuristic.

### 16.6 Fault-injection — the screen's teeth (`dbg_fault`, `dbg_seedless`)

| injection | caught | note |
|-----------|--------|------|
| drop the outermost significant band | 80 / 80 | check 3 (seed coverage) |
| drop a middle significant band | 20 / 20 | check 3 |
| add a spurious *empty* band in the largest gap | 0 / 80 | **harmless** — `band_significant` rejects it, and the M7 integrator skips kEmpty cells; μ unchanged |
| drop a significant band **and its nearest seed** | 68 / 182 | the §5.4 irreducible residual — a thin fully-seed-less band in a wide gap defeats the linear complement sweep |

The last row is the acknowledged limit: **the screen provably cannot certify a
thin seed-less band cold.** Physically such a band only forms where the source
disk straddles a caustic fold, and check 4 catches the resolvable cases, but a
sufficiently thin one in a wide gap is missed. → **step-4 wiring needs
D14-as-authority on the first epoch of a trajectory (warmup) and/or a low-rate
audit** (design points 3 and 5), not a screen tweak.

### 16.7 Files

* `src/lcbinint/magnification/holonomic/fast_bands_screen.hpp` — NEW, the screen.
* `tests/holonomic_cpp/bench_fast_bands_screen.cpp` + `CMakeLists.txt` target — NEW.
* `evidence/holonomic/fast_bands_screen_bench.txt` — NEW, raw bench.
* `src/lcbinint/magnification/holonomic/point_images.hpp` — `aiter` cap 200→80
  (bounds only the non-convergent extreme-q path, which escalates to D14
  regardless; `test_point_images` all-pass, `bench_fast_bands` parity 82/82
  unchanged).

### 16.8 Remaining phase-A work

* **step 4** — wire `fast_bands` + `fast_bands_screen` as an alternative
  discovery path **behind a flag** in the value-only and full-Jacobian epochs.
  Deliverables: full-epoch bench (fast path vs `classify_cells` path) both
  modes; 108-case parity (0 `Status` changes); decision-20 re-check; commit.
  **Design constraints from step 3:** (a) route known-unreliable-seed epochs
  straight to D14 without running the planner (16.4/16.5); (b) D14-as-authority
  on the first epoch of a trajectory and/or a low-rate audit, since the screen
  cannot certify thin seed-less bands cold (16.6).


---

## 17. Phase A step 4 — fast-planner topology wired into `epoch_jacobian` (flag-gated) (2026-09-08)

### 17.1 What landed

`classify_cells_fast(pf, &FastTopologyStats)` is now the alternative band-discovery
path, reachable from the fused value+5-Jacobian epoch:

* **`fast_topology.hpp`** — `classify_cells_fast` = double-tier point-source seed
  solve (`pimg_solve_verify`) → `fast_bands` → `fast_bands_screen` → build a
  `TopologyResult` from the screened bands, **or** `return classify_cells(pf)` on
  any bail. `FastTopologyStats { used_fast, fell_back, reason, n_bands,
  screen_probe_calls }`.
* **`epoch_jacobian.hpp`** — `flux_jacobian` / `epoch_jacobian` gained a 3rd/4th
  arg `bool use_fast_planner`; the 2-arg / 3-arg wrappers read
  `holo_fast_planner_enabled()` (env `HOLO_FAST_PLANNER`, parsed once as
  `static const bool`, `e && e[0]=='1'`). `classify_cells` / D14 stays the
  default and the authority.
* **`fast_bands_screen.hpp`** — `sig.empty()` (no significant band) now yields
  `pass = false` ("near-tangency, D14 authority"), not a silent empty adopt.
* **`point_images.hpp`** — informational `bool escalated` on `PointImages`
  (tier-2 `__float128` rescue marker); `aiter` cap 80.

Default build (`HOLO_FAST_PLANNER` unset) is **byte-identical** to before step 4:
`test_holonomic_m7 m7_reference.tsv` → **10397 checks, 0 failures**.

### 17.2 The key finding — `classify_cells`' cell list is a quadrature panel grid

The M7 integrator (`flux_jacobian`) runs a **fixed `n_r = 64` Gauss–Chebyshev
radial pass per cell**. `classify_cells` places a cell boundary at **every
`radial_events` radius** — folds (`physical_real`), soft boundaries
(`physical_complex`), `R = √m0`, `R = a`, `L_root`, `chart_p4`. Those boundaries
are what make the per-cell radial quadrature spectral: a fold at a panel edge is
integrable; a fold **mid-panel** is a √-type singularity that drops the pass to
~2nd-order. `fast_bands` returns only the outer image-region **envelopes** — a
correct topology, but 64 nodes then span a fold-crossing range and μ is wrong at
~1e-3. **So the fast planner cannot just hand its bands to the n_r=64 integrator;
it has to reconstruct the panel grid inside each envelope.**

### 17.3 `panel_cuts` — cheap panel-grid reconstruction

`fast_topo_detail::panel_cuts(lo, hi, pf, cuts)` subdivides one envelope using
only the **closed-form boundary quartic** (`boundary_quartic`, ~0.3 µs), no D14:

* **Folds** — sign changes of the **quartic discriminant** `D(R)` over a K=128
  scan of the envelope, refined by `bisect_sign`. Quartic disc sign encodes the
  real-root count (0↔2↔4 all flip it); a fold is exactly where it flips.
* **Chart crossings** — sign changes of `p4(R)` (the quartic's leading
  coefficient; `quartic_topology` returns `kDegenerate` at `p4=0`, a boundary
  point crossing θ=π).
* **`R = √m0` and `R = a`** — added directly when interior to the envelope.

Cost is negligible (all `boundary_quartic`, ~0.3 µs × ~260 evals per envelope).
`panel_cuts` recovers the benign / close / cusp cells to an **exact
`classify_cells` match**; plan15 μ error 6e-3 → 5e-5.

### 17.4 The `quartic_topology` φ>0 blind spot and the fail-closed guard

`quartic_topology(R, pf)` counts the boundary quartic's real-root θ's but does
**not** verify `phi_lens > 0` on the arc (confirmed with `dbg_vw.cpp` on a
very-wide razor band: `quartic_topology` reports `nx=2` uniformly across the
envelope while `arcs_at(R, pf, 512)` and `arcs_at(…, 3072)` both correctly return
`kEmpty` — φ<0 everywhere). A `panel_cuts` grid built from `quartic_topology`
alone would therefore integrate phantom arcs → **grad_rel max 6.28e-1**, a
silently wrong Jacobian.

Fix — a **fail-closed per-sub-cell guard**, mirroring `classify_cells`'
per-cell validation (`cells.hpp:83-124`): for each sub-cell probe fractions
{0.18, 0.50, 0.82} with `quartic_topology`, require kind+n_crossings uniform,
**and** cross-check the midpoint against `arcs_at(mid, pf, 512)`. On
`!uniform || !mid_ok`, **or** a `kEmpty` midpoint, the whole epoch returns
`classify_cells(pf)` (`s.fell_back = true`, reason recorded). Post-guard:
grad_rel max 1.20e-2, μ_rel max 3.03e-5; adoption 46 → 26, p99 4.06 → 4.64 ms.

### 17.5 Final numbers (`bench_fast_topology`, 108 epochs, best-of-150)

| metric | value |
|--------|-------|
| fast band set adopted (screen pass) | **26 / 108 (24.1%)** |
| fell back to `classify_cells` / D14 | 82 / 108 (75.9%) |
| `Status` changes fast vs base | **0 (PASS)** |
| μ_rel median / p90 / max | 0 / 3.21e-8 / 3.03e-5 |
| grad_rel (L2) median / p90 / max | 0 / 6.13e-5 / 1.20e-2 |
| dmu_du_rel median / p90 / max | 0 / 1.57e-5 / 2.42e-1 |
| `classify_cells` path median / p90 / p95 / p99 | 2.162 / 3.847 / 3.978 / 4.111 ms |
| fast path median / p90 / p95 / p99 | 1.800 / 3.477 / 3.874 / **4.642** ms |
| **DECISION-20** (fast path median ≤ 7.44 ms) | **PASS** (10.5× incumbent median) |

Fallback-reason tally (54 configs, u=0): 13 adopt / **30** "point-source double
tier not a clean odd set" (the §16.4 extreme-q design — bails at the cheap seed
stage, never runs `fast_bands`) / 10 "fast sub-cell topology not certified
(razor / fold)" / 1 "no significant band".

`HOLO_FAST_PLANNER=1` reference harness: **81 failures** on 6 geometries —
benign(10) / close(9) / cusp(10) grad-component-only (μ passes); plan15 μ 5e-5,
resonant μ 6e-7, tiny-rho (ρ=5e-3) μ 2.7e-3. All confined to `physical_complex`
soft-boundary panels + fold-edge ∂/∂ρ precision (see §17.6).

### 17.6 Residual gaps — findings that motivate phases D & E (NOT bugs to fix in A)

1. **`physical_complex` soft boundaries are not reconstructible from the boundary
   quartic.** They are constructed from the degree-14 resultant's **complex**
   roots (Re(v)>0), or real double roots D14 re-classifies as complex — arc
   count is *unchanged* across them, so there is no quartic-disc sign flip, and
   `dbg_dip.cpp` confirms they are **not** at arc-width / gap local minima
   either (all sub-cell width ratios 1.0–1.16, no separation). A missing soft
   panel boundary → mid-panel near-tangency → the n_r=64 pass loses digits:
   resonant μ ~6e-7, plan15 ~5e-5, tiny-rho ~2.7e-3. This **is** the "cheaper-
   than-D14 root-exclusion / root-count certificate" that design point 3
   explicitly defers to future research — not a current premise.
2. **Fold-edge precision.** `panel_cuts` lands fold radii by disc-sign bisection
   to ~1e-8; `classify_cells` gets them from float128 D14 Aberth to ~1e-12. Near
   a fold ∂μ/∂ρ ~ 1/√(R−R\*) amplifies the ~1e-8 offset to ~1e-4 grad error on
   benign / close / cusp (μ still passes).
3. **p99 +0.5 ms is structural.** With a 76% fallback rate, every fallback epoch
   pays the fast-path prelude — seed solve (~0.2 ms) + `fast_bands` (~0.3 ms) +
   screen (~0.45 ms) — *on top of* the same D14. A cold per-epoch planner
   **cannot** meet "p99 non-regressing" as a `classify_cells` replacement. The
   fix is **phase E** (prepared-geometry cache / trajectory reuse: pay D14 once
   at warmup, then the cheap planner + screen per epoch) and **phase D** (a
   panel-robust / adaptive integrator that tolerates the coarser envelope grid
   so more epochs adopt).
4. **Tiny-q razor planet bands** (rand030/037/038): grad_rel ~1e-2..6e-2 while
   μ_rel ~1e-8..1e-11 — the razor band contributes ≈0 to μ. Inherent to the
   screen's "razor bands not screened" design (`kRazorAng`).

### 17.7 Standing design constraints carried forward (from §16.4–16.6)

* **Route fragile-seed epochs straight to D14** — `classify_cells_fast` bails to
  `classify_cells` *before* `fast_bands` whenever `pimg_solve_verify` does not
  return a clean odd set (30/54 configs). No q-threshold predicate (§16.5,
  REJECTED — q does not separate quad-rescuable geometries).
* **D14-as-authority on trajectory epoch 1 + a low-rate audit** — the screen
  provably cannot certify a thin seed-less band cold (§16.6 last row). Phase E's
  warmup must run D14 on the first epoch and re-audit at a low rate, not trust
  the screen alone.

### 17.8 Verdict

Phase A step 4 lands the wiring, the cheap subdivision, and the fail-closed
guard, with the flag **OFF by default** (zero change to the shipped path). The
fast planner is **not yet a viable default**: 24.1% adoption, a real
`physical_complex` accuracy gap, and a structural p99 regression from the 76%
fallback prelude. Phases **D** (panel-robust integrator) and **E**
(prepared-geometry cache / trajectory reuse) are prerequisites before the flag
can flip. Phase A (fast planner + §5.2 screen in front of D14) is **complete**.

### 17.9 Files

* `src/lcbinint/magnification/holonomic/fast_topology.hpp` — `fast_topo_detail`
  namespace (`quartic_disc_sign`, `quartic_p4_sign`, `bisect_sign`,
  `panel_cuts`); `classify_cells_fast` cell-build loop rewritten to subdivide
  each band + the fail-closed per-sub-cell guard.
* `src/lcbinint/magnification/holonomic/epoch_jacobian.hpp` — 3-arg
  `flux_jacobian` / 4-arg `epoch_jacobian` + `holo_fast_planner_enabled()`
  wrappers; `#include "…/fast_topology.hpp"`.
* `src/lcbinint/magnification/holonomic/fast_bands_screen.hpp` — `sig.empty()`
  → `pass = false`.
* `src/lcbinint/magnification/holonomic/point_images.hpp` — `bool escalated`.
* `tests/holonomic_cpp/bench_fast_topology.cpp` + `CMakeLists.txt` target — NEW.
* `evidence/holonomic/fast_topology_bench.txt` — raw bench + reference-harness
  default/flag-on summary + fallback tally.

---

## 18. Phase B step 0 — `radial_events` / D14 cost split (2026-09-08)

**Governing task (user, 2026-09-08):** this phase does **not** remove D14. Keep
every bit of information D14 carries — the real fold events *and* the
complex-root-derived panel boundaries — and cut the **cost of solving** it.
Four candidate approaches, user ranking in parentheses: ① compensated
Ehrlich–Aberth (user's #1, "may help even the cold single shot"), ② secular
equation transformation (#2, gate on implementation cost), ③ exact on-axis
6+4 factorization `Disc_t(P)/R⁴ = 16·F6(R²)·G4(R²)²` for `ys = 0` (#3, narrow
but mathematically clean; symbolically verified — `87521a12-d14_axis_factor_check.py`),
④ all-root warm-start across trajectory epochs (#4, big for the series/light-curve
workload). Explicit judging rule: **separate coefficient-construction precision
from root-solve precision**; judge on the *whole* `radial_events` (coeff build +
solve + verify + classify + panel generation) and the final-epoch time, not the
bare root solve; hold root information, final magnification, and Jacobian
precision fixed; single-shot cold vs series warm judged separately.

### 18.1 Measurement

`tests/holonomic_cpp/bench_d14_split.cpp` — per (config, u) point, best-of-N,
times each stage of `radial_events` in isolation: `pcoef` (`p_coeffs_in_R`),
`d14c` (`d14_coeffs` — the Iq/Jq quartic-invariant assembly and
`disc = (4·Iq³ − Jq²)/27` degree-36 `__float128` polynomial arithmetic),
`dsrch` (balanced `aberth<double>` pre-search, 200-iter cap), `qpol` (seeded
`aberth<__float128>` warm polish, 24-iter cap), `drr` (`double_root_is_real`
loop), `side` (`p4(R)=0` chart + `L(v)` on-axis + misc), plus full
`radial_events` and full `epoch_jacobian(p, u, 64, false)`. `ScopedFlushDenormals`
per case. 108 (config, u) points from `/tmp/bench_cases.tsv`, reps=120,
`taskset -c 0-7`, load ~10.

```
  pcoef   median 0.0123   ( 1.2% of radial_events ,  0.6% of epoch )
  d14c    median 0.0425   ( 4.2% of radial_events ,  2.0% of epoch )   p90 0.0429  max 0.0436
  dsrch   median 0.1411   (13.8% of radial_events ,  6.5% of epoch )
  qpol    median 0.6885   (67.5% of radial_events , 31.9% of epoch )   p90 2.4621  max 2.5010
  drr     median 0.0028   ( 0.3% /  0.1% )
  side    median 0.1229   (12.1% /  5.7% )
  RE      median 1.0193   p90 2.8063   max 2.8421
  epoch   median 2.1612   p90 3.8395   max 4.1364
  cold-quad fallback alone : median 12.6600  p90 42.7050  max 47.2116 ms
  (dsrch + qpol) / radial_events : median 81.4%  p90 93.0%
  d14_coeffs empty : 0/108   warm-polish -> cold-quad fallback : 0/108
```

### 18.2 Verdict

* The **`__float128` warm polish (`qpol`) is the entire target** — 67.5% of the
  `radial_events` median, 31.9% of the full value+5-Jacobian epoch median, and
  ~all of the `radial_events` p90 tail (0.69 → 2.46 ms, a 3.6× spread: the
  24-iter cap lets easy roots break at ~5 iters while near-multiple clusters run
  all 24).
* **Coefficient construction is not the bottleneck.** `d14c` is 4.2% of
  `radial_events` and dead flat (0.0425 median / 0.0436 max — no case-to-case
  spread). `pcoef` 1.2%. Per the user's "separate coeff precision from solve
  precision" rule: the answer is that the coeff build can stay in `__float128`;
  the **solve** precision level is what to attack. ③'s cheaper on-axis coeff
  build is a minor bonus; ③'s real value is removing the structural `G4²`
  double root.
* `(dsrch + qpol)` = 81.4% / 93.0% (median / p90) of `radial_events`.
* Cold-quad fallback (400 iter, tol 1e-22) costs 12.7 ms median / 42.7 ms p90
  when it fires — 0/108 on this bench (the 24-iter polish + 1e-12 residual gate
  is well calibrated for these cases) but the latent p99 landmine (decision 21:
  un-balanced wide-binary D14 coeff spread → spurious complex pairs in the
  double pre-search → warm seed rejected → cold quad).

### 18.3 Phase B implementation order (set by this verdict)

* **B1 = ① compensated Ehrlich–Aberth.** Replace the `__float128` warm polish
  with a double-double (error-free-transformation) Aberth correction:
  TwoSum / TwoProduct / FMA, compensated Horner for D14 and D14′, compensated
  Aberth update; freeze-on-converge; escalate only the unconverged / genuinely
  near-multiple root subset to `__float128`. D14 κ ≈ 1e9 sits inside the
  double-double range; true quad only for local κ > ~1e14 clusters. Resolve
  FTZ/DAZ — the `ScopedFlushDenormals` guard flushes subnormals and destroys
  EFT error terms → nested MXCSR restore around the compensated solve. Preserve
  the decision-21 monomial balancing. Add a Gershgorin/inclusion count check so
  completeness is not silently weakened (float128 solving was never itself a
  completeness proof — inclusion verification is a separate need). Keep the
  `__float128` path as the retained A/B fallback.
* **B2 = ④ all-root warm-start** across trajectory epochs via `warmup()` /
  the execution plan — no new public API. `aberth<R>()` already has the `seed`
  and `final_step` hooks; `QuarticWarm` / `real_root_thetas_warm` (98% hit) are
  the in-repo precedents. Carry ALL roots (positive-real *and* complex) from the
  previous epoch as the next epoch's Aberth seed. Separate the warm-start
  success test from the missing-root check (root correspondence near a
  bifurcation is hard). Compounds with B1.
* **B3 = ③ exact on-axis 6+4 factorization**, guarded on `|Y| < 1e-14` (existing
  `L_root` precedent at `radial_events.hpp` line ~409). Ferrari closed-form
  sextic + quartic; `G4²` is a known a-priori double → no numerical
  multiplicity-2 cluster. Must **not** round near-axis geometry to `ys = 0`.
  Prototype near-axis perturbative hard-root seeding (near-axis = high
  magnification = scientifically important).
* **B4 = ② secular transformation** — only if B1 + B2 leave a gap. Prototype in
  isolation first; at degree 14 the O(n²) high-precision setup may dominate.

Every new path A/B-compares against the retained `__float128` fallback and must
reproduce: identical positive-real root count, residual ≤ current, identical
`double_root_is_real` classification, identical complex-Re>0 list.

### 18.4 Files

* `tests/holonomic_cpp/bench_d14_split.cpp` + `CMakeLists.txt` target — NEW.
* `evidence/holonomic/d14_solve_cost_split.txt` — raw bench + reading.

---

## §19 Phase B step 1 — compensated (double-double) D14 solve (2026-09-08)

**Status: COMPLETE. Default-ON. `HOLO_D14_LEGACY_SOLVE=1` = exact legacy A/B.**
Evidence: `evidence/holonomic/d14_compensated_bench.txt`.

### 19.1 What changed

The `__float128` (libquadmath, software) Aberth–Ehrlich warm polish of D14
— B0's `qpol`, 67.5% of `radial_events` median / 31.9% of the epoch median
— is replaced by a polish in **double-double** (unevaluated sum `hi+lo`,
~106 bits, error-free transformations in hardware `double` + one `std::fma`
per product). D14 κ ≈ 1e9 sits well inside the double-double range; the
coefficient build stays `__float128` (B0: separate coefficient-construction
precision from root-solve precision — held).

* `src/lcbinint/magnification/holonomic/dd_real.hpp` — NEW. `DD` type,
  TwoSum/FastTwoSum (`+`/`−` only, contraction-immune), TwoProduct via
  `std::fma` (contraction-immune by construction), `/`, ordering, `qabs_`,
  `qsqrt_` (Karp), `dd_from_qf` / `qf_from_dd`. Deliberately mirrors the
  small surface `aberth<R>()` / `Cplx<R>` need → `aberth<DD>` and
  `Cplx<DD>` instantiate with **no change to poly_roots.hpp**.
* `re_detail::solve_d14()` (radial_events.hpp) — multi-tier solve extracted
  from the inline block:
  * tier 0 — dd Aberth, 25 sweeps, tol 1e-26, seeded from the balanced
    `double` presearch (decision-21 monomial balancing preserved). Gate:
    worst `__float128` residual ≤ 1e-13.
  * tier 1 — `__float128` warm polish seeded by the dd roots, for a
    near-multiple cluster past 106 bits. Gate 1e-12.
  * tier 2 — cold `__float128` Aberth (400 iter, tol 1e-22): seed
    non-finite, OR tier-1 gate failed, OR the Vieta completeness check
    failed.
  * tier 3 — legacy: `__float128` warm polish straight off the double
    presearch, no dd stage (`HOLO_D14_LEGACY_SOLVE=1`).
* **Completeness check** (B0 verdict / user: "float128 で解いたこと自体は
  完全性の証明ではない。根を誤差込みで囲って全根が含まれると確認する処理は
  別に必要"): Newton's first identity `Σ roots == −c[1]/c[0]`. A dropped or
  doubled basin — how an under-resolved Aberth actually fails — shifts the
  deg-14 power sum far outside rounding. Absolute tol `1e-6·(1+|want|)` on
  O(1..10) roots ≈ 1e5× the honest round-off → cold re-solve on failure.
  0/115 bench cases tripped it.
* **Conjugate symmetrization**: real coefficients ⇒ non-real roots are
  exact conjugate pairs. Aberth splits a pair's real parts ~1e-8 rel in dd
  (vs ~1e-11 in `__float128`); `radial_events` dedups the complex-Re
  soft-boundary list at a fixed 1e-9 gap, so the wider dd split would post
  one pair as two soft boundaries. Snap every near-conjugate pair
  (clearly complex, conjugate distance < 1e-6 rel) onto its common real
  part and mean |im|. No-op for the `__float128` paths.
* `src/lcbinint/magnification/holonomic/fp_env.hpp` — `ScopedNoFlushDenormals`
  (inverse of `ScopedFlushDenormals`; nested around the dd solve so EFT
  `lo` limbs survive). FTZ/DAZ **proven bit-exact inert** here
  (`/tmp/probe_ftz.cpp`: `max|root_ftz − root_noftz| = 0.000e+00`) — the
  guard is structural, not a measured fix.
* `tests/holonomic_cpp/bench_d14_compensated.cpp` + `CMakeLists.txt` target
  — NEW. Direct A/B of `solve_d14(desc,deg,false/true)` + whole
  `radial_events` + full `epoch_jacobian`, parity vs legacy.

### 19.2 Result (bench_d14_compensated, reps 80, taskset -c 0-7, load ~10)

| stage | legacy | compensated | speedup |
|---|---|---|---|
| bare `solve_d14` | med 0.774 / p90 2.638 / p99 2.679 ms | med 0.511 / p90 0.531 / p99 2.659 ms | **1.51× / 4.97×** |
| whole `radial_events` (compiled path) | med 0.927 / p90 2.798 / p99 2.837 ms | med 0.603 / p90 0.627 / p99 2.817 ms | **1.54× / 4.46× / p99 1.01×** |
| full value + 5-Jac epoch | med 2.147 / p90 3.835 / p99 4.118 / max 4.13 ms | med 1.253 / p90 2.259 / p99 3.958 / max 4.15 ms | **1.71× / 1.70× / p99 1.04× / max ~1.00×** |

* tier histogram: **113 dd-sufficed / 2 qf-escalate / 0 cold-quad** (the 2 =
  rand028 both u — one near-multiple cluster, tier 1, final residual 3.8e-14).
* parity failures **0 / 115**; worst root-set rel diff comp-vs-legacy 2.598e-09.
* worst `__float128` residual: legacy 1.604e-15, comp 3.836e-14 — far past
  every downstream tol (`double_root_is_real` 1e-6, positive-real merge 1e-9,
  complex-Re dedup 1e-9). Final μ + 5-Jac **bit-for-bit unchanged** in
  `m7_reference.tsv` (10397 checks / 0 failures, default AND legacy).
* p90/p95/p99 all non-regressing → **decision-20 tail condition MET**. The
  +0.02 ms on the compensated `max` is the rand028 escalation case, within
  run-to-run noise.

### 19.3 Reading

The polish precision level was the whole cost, exactly as B0 predicted.
Dropping from ~113-bit software `__float128` to ~106-bit hardware
double-double — still ~10 decimal digits past anything the event list,
classification, or Jacobian consumes — halves the bare solve and takes the
epoch median from 2.15 → 1.25 ms. Every bit of D14's information is
retained: identical fold-event list, identical complex-root-derived panel
boundaries, identical `double_root_is_real` classification. Only the
arithmetic precision of the iterative polish changed, and the completeness
check + residual gate + conjugate symmetrization keep that change from
silently weakening the result.

Compounds with B2 (all-root cross-epoch warm-start): a warm dd seed from
the previous epoch should push most cases to ~3 dd sweeps.

### 19.4 Files

* `src/lcbinint/magnification/holonomic/dd_real.hpp` — NEW.
* `src/lcbinint/magnification/holonomic/fp_env.hpp` — `ScopedNoFlushDenormals`.
* `src/lcbinint/magnification/holonomic/radial_events.hpp` — `solve_d14()`,
  completeness check, conjugate symmetrization, `HOLO_D14_LEGACY_SOLVE` hatch.
* `tests/holonomic_cpp/bench_d14_compensated.cpp` + `CMakeLists.txt` — NEW.
* `evidence/holonomic/d14_compensated_bench.txt` — raw bench + reading.

---

## §20 Phase B step 3 feasibility + step 2 dependency (2026-09-08)

### 20.1 B3 (on-axis factorization) — identity confirmed, closed-form premise refuted

Evidence: `evidence/holonomic/d14_onaxis_factorization.txt`.

* The sympy identity `Disc_t(P)/R⁴ = 16·F6(v)·G4(v)²` was matched **term
  by term** to the C++ `re_detail::p_coeffs_in_R` construction and verified
  numerically (mpmath 70-digit, both exact-ys=0 bench configs): identity
  residual 2.4e-71, `F = p4·p0` and `H = p2²−4F` **exactly** even in R
  (`p2` is purely even; `p0`, `p4` are not — their odd cross terms cancel).
  The factorization is real and exact in the production basis.
* **But it is not a closed form.** `G4` (deg 4) → Ferrari, clean, and its
  roots are the a-priori multiplicity-2 roots of D14. `F6` (deg 6) has
  **no product factorization** — `F6(v) = pe4·pe0 + v·po4·po0`, a sum —
  and a general sextic has no radical solution. "Ferrari closed-form
  sextic" (§18.3) does not exist. Worse, F6's roots are real and
  *clustered* for the on-axis cases (on-axis-off: two near-triple
  clusters in R) — exactly the structure that makes the on-axis D14
  Aberth slow (B0 top-12). F6 keeps that clustering.
* **Revised payoff:** deg-14 clustered Aberth → deg-4 Ferrari + deg-6
  clustered numerical solve ≈ 1.5–2× on the on-axis solve, on **2/108**
  configs (exact ys=0 only; near-axis unaffected). B1 already dominates.
* **RECOMMENDATION: defer B3.** If pursued, fold only the Ferrari `G4`
  sub-solve into `solve_d14` behind `|Y|<1e-14` (removes the a-priori
  double root) — not a separate closed-form engine. `§18.3`'s "Ferrari
  closed-form sextic + quartic" and "G4² … no numerical multiplicity-2
  cluster" are corrected here: only the G4 half is clean.

### 20.2 B2 (all-root cross-epoch warm-start) — blocked on a threading decision

B2 = carry the previous epoch's full D14 root set as the next epoch's
Aberth seed. §18.3 said "via `warmup()` / the execution plan — no new
public API", but:

* The holonomic backend is **not wired into `FiniteSourceMagnifier`**
  (memory: "Neither backend is wired … both standalone"). `warmup()` /
  `MagnificationExecutionPlan` / `compare_warmup_geometry` live in the
  main lcbinint path and do not reach `radial_events`. Phases **C**
  (mode router / wiring) and **E** (prepared-geometry cache) build that
  bridge — they come *after* B in the user's order.
* The in-repo precedent §18.3 actually cites — `QuarticWarm` /
  `real_root_thetas_warm` — is **workspace threading**, not the
  execution plan: a caller-owned scratch struct passed through the call
  chain. For B2 that means a `D14Warm` workspace threaded
  `radial_events → cells → epoch_jacobian`, populated across a
  light-curve sample loop by the bench / a trajectory driver.
* That is a real, non-trivial change (new overloads down the chain) and
  its payoff needs a trajectory-level bench that does not exist yet.
  Root-correspondence near a bifurcation is the known hard part (§18.3).

**Decision needed from the user:** thread a `D14Warm` workspace now (the
`QuarticWarm` pattern, standalone, no `FiniteSourceMagnifier` wiring), or
reorder — do C/E first so B2 rides the prepared-geometry cache. Until
then B1 stands as the delivered Phase B result (gate met).

### 20.3 Phase B status

* **B1 — DONE, committed `69cc789`, gate met** (§19): compensated dd D14
  polish default-ON, epoch median 1.71× / p90 1.70× / p99 non-regressing,
  0/115 parity, 10397/0.
* **B2 — blocked** on 20.2 (threading decision / phase reorder).
* **B3 — deferred** per 20.1 (no closed form, 2% scope).
* **B4 (secular)** — unchanged: only if a gap remains after B1+B2.

---

## §21 Phase C/E (prepared-epoch geometry cache) + Phase B2 (all-root D14 warm-start) (2026-09-08)

The user chose **option 2** in §20.2: do not build a standalone `D14Warm`
workspace; advance C/E first and land B2's all-root warm-start on the final
`PreparedEpochGeometry` / the existing `warmup()` infrastructure. Final
architecture the user specified: prepared state holds D14's all 14 roots
(incl. complex), radial events, physical/soft boundaries, panel cuts /
CellPlan, conditioning / validity info; next-time evaluation hierarchy is
`cached roots → compensated DD warm polish → DD cold solve → legacy
__float128 fallback`; keep extending `warmup()` / `MagnificationExecutionPlan`,
no new public trajectory API; B1's compensated solver is the standard D14
kernel, legacy `__float128` stays as the correctness fallback / A–B oracle.

### 21.1 What landed

New header `src/lcbinint/magnification/holonomic/prepared_geometry.hpp`:

* **`PreparedEpochGeometry`** — Layer-1 immutable payload frozen once per
  trajectory / proposal: `anchor` frame, `r_max`, `cells` (the quadrature
  panel plan), `d14_roots` (all 14 incl. complex — the B2 seed), `d14_deg`,
  `ConditioningMargins` (narrowest cell, min |p4(r_mid)|, coarse fold count,
  r_max), `status`, `provenance` (kColdOracle / kTopologyReused /
  kWarmRecomputed / kColdRecomputed), `valid`.
* **`prepared_topology(pf, state, cfg, st)`** — the §8 fail-closed ladder for
  one epoch, `state` a caller-owned rolling cache (in/out):
  * **L1** — drift pre-filter (`drift_norm ≤ l1_drift`) **plus a mandatory
    cheap re-screen** (`prepared_rescreen`): r_max invariance, a coarse
    global disc-sign fold-count invariant, and per-cached-cell
    `quartic_topology` at three interior fractions must still match the
    cached (kind, crossing count). Reuse the cached cell plan verbatim only
    if all pass.
  * **L2** — `classify_cells` runs (still the authority), but its D14 root
    solve is **warm-seeded** by `state.d14_roots` (Phase B2). `state` rolls
    forward to the fresh solve each epoch.
  * **L3** — cold `classify_cells`.
* **`prepared_rescreen`** — the independent cheap re-derivation of the cached
  topology at the new geometry (not a trust of the drift pre-filter).

Warm-seed hook threaded down the existing chain (no new public API, the
`QuarticWarm` workspace precedent):

* `radial_events(pf, r_max_out, merge_tol, d14_warm=nullptr, d14_roots_out=nullptr)`
* `classify_cells(pf, d14_warm=nullptr, d14_roots_out=nullptr)`
* `solve_d14(desc_v, deg, compensated, warm_seed=nullptr)` — the warm seed
  refines the **balanced `double` root-basin presearch** (60 iters from the
  previous roots vs 200 cold); **every downstream gate is unchanged**
  (dd Aberth polish, worst-residual, Newton first-identity completeness,
  conjugate symmetrization, cold `__float128` backstop), so a stale or wrong
  seed still fails closed to the cold solve. `D14Solve` gains `warm_seeded`.

Integrator split in `epoch_jacobian.hpp` so cold and prepared paths share
the fused radial pass:

* `flux_jacobian_integrate(p, n_r, pf, topo)` — the fused value +
  internal-Jacobian pass over an already-decided cell plan (unchanged body).
* `flux_jacobian(p, n_r, use_fast_planner)` — thin: band discovery then
  `flux_jacobian_integrate`.
* `flux_jacobian_prepared` / `epoch_jacobian_prepared(p, u, n_r, state, cfg, st)`
  — the Phase E path over a rolling `PreparedEpochGeometry`.

### 21.2 Result — `bench_holonomic_trajectory` (108 configs × 32-epoch synthetic tracks = 3456 epochs, reps 20, taskset -c 0-7, load ~10)

Full output: `evidence/holonomic/prepared_geometry_trajectory.txt`.

| | parity `|dμ|/|μ|` med / max | false reuse | steady median | p90 | p99 |
|---|---|---|---|---|---|
| V0 cold every epoch | — | — | 1.2901 ms | 2.0152 | 3.3452 |
| **V1 = L2 warm-D14 only** | **0 / 3.9e-7** | **0** | **1.1901 ms (1.08×)** | 1.9078 | 3.1551 |
| V2 = L1 + L2 | 0 / 2.5e-2 | **5** | 1.1827 ms (1.09×) | 1.8999 | 3.0983 |

* **L2 (Phase B2) is safe and default-on.** Exact parity (median/p90 |dμ| =
  0; max 3.9e-7 is only the Aberth iteration count differing between a warm
  and cold seed, inside solve_d14's tolerance — `classify_cells` still runs
  every epoch and stays the authority). Fail-closed holds (0 status
  downgrades, 0 false reuse). **1.08× steady-state median**, p90/p95/p99
  **non-regressing** (p99 improved 3.16 vs 3.35 ms). The M7/decision-20 gate
  was already met by B1 (1.71× epoch median); this is additive. Warm seed
  consumed on 3348/3348 non-cold epochs.
* **L1 verbatim topology reuse is NOT default-safe — stays gated OFF**
  (`PreparedReuseConfig::allow_topology_reuse = false`). Only 14/3456 epochs
  reached an L1 reuse (the quartic re-screen rejected 1026/1040 attempts);
  of those 14, **5 were false reuse** (|dμ|/|μ| up to 2.5e-2, ||dgrad|| up
  to 3.49) on near-caustic epochs. The re-screen is built entirely from the
  boundary quartic and is structurally blind to (a) D14's complex-root panel
  boundaries and (b) the panel-placement error of reusing anchor-geometry
  cells at a drifted geometry — percent-level near a caustic. Same wall
  Phase A hit (§17.6). A cheaper-than-D14 topology/panel certificate is
  future research (§0 policy point 3); L1 remains opt-in for experiments.

### 21.3 Config defaults (`PreparedReuseConfig`)

* `allow_topology_reuse = false` — L1 off (21.2).
* `allow_warm_d14 = true` — L2 on.
* `l2_drift = 1e9` — a perf guard only; `solve_d14` fails closed on a bad
  seed, so there is no correctness reason to bound it and the seed setup is
  a few `double` ops. Large ⇒ the warm seed is used whenever roots exist.

### 21.4 Regression checks (all pass, new optional params byte-inert on the default path)

* `ctest` (`holonomic_point_images`, `holonomic_m7_reference`) — 2/2.
* `bench_d14_compensated` — 0/115 parity, compensated 1.51× median (B1 intact).
* `bench_fast_topology` — 0 status changes, decision-20 PASS (Phase A intact).

### 21.5 Files

* `src/lcbinint/magnification/holonomic/prepared_geometry.hpp` — NEW.
* `src/lcbinint/magnification/holonomic/radial_events.hpp` — `solve_d14`
  warm-seed hook, `D14Solve::warm_seeded`, `radial_events` d14_warm /
  d14_roots_out params.
* `src/lcbinint/magnification/holonomic/cells.hpp` — `classify_cells`
  d14_warm / d14_roots_out passthrough.
* `src/lcbinint/magnification/holonomic/epoch_jacobian.hpp` —
  `flux_jacobian_integrate` split, `flux_jacobian_prepared`,
  `epoch_jacobian_prepared`.
* `tests/holonomic_cpp/bench_holonomic_trajectory.cpp` + CMake target — NEW.
* `evidence/holonomic/prepared_geometry_trajectory.txt` — NEW.

### 21.6 Status / next

* **Phase E (prepared-geometry cache)** — landed; L2 (= Phase B2) is the
  default reuse level and the delivered win. L1 held behind an off flag.
* **Phase C (mode router / `FiniteSourceMethod`)** — the prepared path is
  proven in the isolated harness (as Phase A was). Production wiring onto
  `MagnificationExecutionPlan` / `warmup()` is a thin additive adapter
  (§6 correspondence table) — still to do, no `_lcbinint.so` rebuild here.
* **B2 — DONE** as the L2 warm-seeded D14 solve on Phase E.
* Next: **Phase D** (M1/M2 value lanes + panel-robust integrator), then
  **Phase F** (M3 JVP). B3 deferred, B4 only if a gap remains.

---

## §22 Gauss–Manin / true-holonomic transport — feasibility study (2026-09-09)

The user asked, as the next big algorithmic candidate, for a serious look at a
**true holonomic / Gauss–Manin period transport that does NOT solve the
boundary quartic at each radial node** — transport Π(R) via dΠ/dR = C(R)·Π
across a cell instead of the per-node angular √φ quadrature — with a specific
8-point checklist, and the explicit framing: *"目標は holonomic という名前に
こだわることではなく、現在の optimized D14 + quartic-warm engine より明確に
速くなるか"*, implementation order to be decided from the code dependencies.

### 22.1 Measurement — `bench_radius_terms_split` (NEW)

One full value + 5-Jacobian epoch (`epoch_jacobian`, n_r=64) over the 108
bench configs, best-of-150, decomposed into the stages a period transport
would / would not remove. Post B1 + B2/L2 + M8 FTZ + quartic warm-start
(today's optimised engine). Host: pinned 0-7, load ~12–14, 2026-09-09.

| stage | median ms | ~% epoch |
|---|---|---|
| `epoch_jacobian` (full) | 1.261 | 100 |
| `classify_cells` (D14 + topology) | 0.790 | **63** |
| `radius_terms` per-node sweep, WARM (production) | 0.420 | 33 |
| — quartic root solve (warm) | 0.173 | 14 |
| — 64-pt angular √φ + dP sweep | 0.227 | **18** |
| — polish + arc filter + coeff build | 0.041 | 3 |
| assembly / internal→user Jac | ~0.05 | 4 |

(cold quartic solve 0.347 ms; warm-start already claws back half of it.)

**This corrects the M8 `holoprof9` "angular sweep ~0.02 ms, negligible"
line.** holoprof9 isolated only the *incremental* dfh chain-rule add-on over
an already-evaluated φ. The full 64× `phi_val_dP` loop (φ + 5 partials) over
every arc-bearing cell, measured standalone, is **0.23 ms median / 0.75 ms
p90** — a real 18 % of the epoch. So Π-transport is not chasing nothing; but
it is also nowhere near the dominant term.

### 22.2 8-point checklist

| # | question | finding |
|---|---|---|
| 1 | period basis | η / ψ both built + pointwise-validated; both ill-conditioned across a cell — `cond(C_ψ)` up to ~1e12 near θ→π (M4). Well-conditioned **flux-priority basis G (plan §8) never built** (deferred M4→M8); policy pt 3 bars banking on it. |
| 2 | how cheap is C(R) at runtime | per node: 8×8 solve for `(Q_t)⁻¹ mod Q` + 7 poly reductions ≈ **2000 flops**, vs ≈ 12800 for the 64× sweep. Raw C(R) build is ~6× cheaper than the sweep — a real but modest structural win *in isolation*. |
| 3 | seed period per cell | `seed_psi` = deflated QAWSE, 7× `scipy quad` under alg weight ≈ **≥ one angular sweep**. `tangency_seed_eta` closed series is cheap but fold-only. |
| 4 | re-anchor at endpoint / fold / soft boundary | **the wall.** M4 measured single-seed-per-cell transport losing 2–8 digits by the cell edge; the M4/M6 reference **re-anchors at every radial node** for exactly this reason. Per-node re-anchor = pay the pt-3 seed cost (≥ 1 sweep) per node ⇒ **strictly slower than the sweep it replaces.** Endpoint-only anchoring fails the decision-20 accuracy gate. |
| 5 | can both F0 and F_half transport | F0 needs no Gauss–Manin at all (pure arc measure `R·ΣΔθ`; df0 = IFT-in-θ at endpoints, already ~0). Only F_half is a period. "Transport both" = "transport one". |
| 6 | 5-Jac / JVP in the same system | possible (parametric connection `∂Π/∂P_j`, M3 symbolic pieces exist) but ×~6 the per-node build and inherits the pt-4 conditioning wall on every column. No saving over the current closed-form `dφ/dP` in the sweep. |
| 7 | reduction in node-wise quartic solves | **zero** from Π-transport alone — endpoints still needed. The quartic solve (14 %) is removed only by the **separate (m,v) root-pair radial ODE** (`root_pair.root_pair_dR`: analytic 2×2 IFT, well-conditioned off-tangency, fail-closed at folds), already **−16 % in the algebraic backend** ([[project_algebraic_mv_transport]]). That is the "no per-node quartic solve" that works today. |
| 8 | does C(R) construction become the bottleneck | not in isolation (pt 2), but amortising the t-reduction across R needs a 2-variable (t,R) Picard–Fuchs system — explicitly on the plan §17–18 **"do not do"** list — and the pt-4 re-anchor forces a seed per node regardless, so effective per-node cost `C build + ODE step + seed` **> the sweep**. |

### 22.3 Verdict — NOT a clear win; keep off the critical path

* Ceiling gain **1.22×** (18 % of epoch), and only if the conditioning wall
  did not exist. It does (measured M4), it forces a per-node re-anchor, and
  a per-node re-anchor is slower than the sweep.
* The one escape — the plan §8 flux-priority basis — is unbuilt speculative
  research; policy pt 3 bars it as a current premise.
* The decision-20 gate is **already met with ~3.5× margin** (B1 epoch 1.71×;
  M8 step 3 11.7× vs incumbent). An at-risk 1.22× ceiling needing new
  research does not earn the critical path.

### 22.4 Recommended order (code-dependency driven, per the user's ask)

1. **Production wiring first (Phase C).** Thin additive adapter:
   `FiniteSourceMethod::holonomic` → `binary_mag_preplanned` dispatch to
   `epoch_jacobian_prepared` over a per-trajectory `PreparedEpochGeometry`
   cache; teach `warmup()` when to select it. Zero algorithmic risk,
   unblocks the banked 1.71× (B1) + 1.08× (B2/L2), **not blocked by any of
   this research**. B1/B2 kernel stays baseline/fallback (§0 constraint).
2. **(m,v) root-pair radial transport** — targets the 14 % warm quartic
   solve + polish; proven approach, analytic, fail-closed. The bankable
   "no per-node quartic solve".
3. **Gauss–Manin Π-transport** — research spike only, behind an off flag.
   Entry condition: first derive + condition-test the plan §8 flux-priority
   basis G and show `cond < ~1e6` across a full cell *including* the θ→π
   degree drop. Until then the per-node re-anchor makes it a net loss.
4. Biggest remaining lever is still `classify_cells` / D14 (63 %): B1 + B2/L2
   banked; L1 verbatim reuse blocked on the same cheaper-than-D14
   topology/panel certificate (policy pt 3 research).

### 22.5 Files

* `tests/holonomic_cpp/bench_radius_terms_split.cpp` + CMake target — NEW.
* `evidence/holonomic/gauss_manin_feasibility.txt` — NEW (full checklist
  write-up + the flop counts).

**User sign-off (2026-09-09):** proceed with Phase C; then `(m,v)` root-pair
transport as the priority; Gauss–Manin stays off the critical path as a
research candidate. For Phase C: land the isolated internal router header
first, synchronise only the shared-build touch (`.so` rebuild) to a timing
window.

---

## §23 Phase C step 1 — internal mode router `finite_source_binary` (2026-09-09)

The engine-side seam that the production finite-source layer will dispatch to,
built and validated **entirely in the isolated harness** — no shared `.cpp`
touched, no `_lcbinint.so` rebuild. The enum value + `binary_mag_preplanned`
case + `warmup()` selection are step 2 and land with a coordinated rebuild.

### 23.1 What landed — `src/lcbinint/magnification/holonomic/finite_source_binary.hpp` (NEW)

* **`RequestedOutput`** — `kValue` / `kValueJacobian` / `kValueJvp`. Work is
  chosen by *requested output*, not algorithm name (§0 policy pt 2).
* **`FiniteSourceRequest`** `{ LensParams params; double u; RequestedOutput
  output; int n_r; array<double,5> jvp_direction; }` — engine-neutral.
* **`FiniteSourceOutcome`** `{ mu; grad_mu[5]; dmu_du; mu_jvp; F0; F_half;
  r_max; Status status; PreparedEpochGeometry::Provenance provenance;
  bool has_jacobian; bool has_jvp; }`.
* **`finite_source_binary(req)`** — stateless/cold, forwards to
  `epoch_jacobian`.
* **`finite_source_binary_prepared(req, state, cfg, stats)`** — trajectory
  route over a caller-owned rolling `PreparedEpochGeometry`, forwards to
  `epoch_jacobian_prepared`; `provenance` propagated from the cache.
* **`default_reuse_config()`** — production defaults: L2 warm-D14 ON, L1
  verbatim reuse OFF (§21).
* `kValueJvp` **fails closed** (`GRADIENT_UNRELIABLE`) — the M3 JVP lane is
  not built (Phase F). `kValue` runs the fused pass today and withholds the
  Jacobian; the M1/M2 value lane (Phase D) slots in here with no signature
  change.
* Zero arithmetic added — `from_epoch` is a field copy. The standalone
  `epoch_jacobian*` entry points stay for A/B.

### 23.2 Validation — `tests/holonomic_cpp/test_finite_source_binary.cpp` (NEW)

`ctest -R holonomic_finite_source_binary` — **16524 checks, 0 failures.**

* Cold route vs a direct `epoch_jacobian` call over all 108 bench configs:
  `mu` / `F0` / `F_half` / `dmu_du` match to < 1e-12 rel; `grad_mu` to
  < 1e-11 rel **except `∂μ/∂ρ` (j==2)** which is checked at 1e-6 — worst
  observed 6.5e-7 on one tiny-ρ config. That drift is the documented
  `(large)/D − 2μ/ρ` cancellation (checkpoint_M6 §5): under
  `-ffp-contract=fast` the forwarded `epoch_jacobian` rounds that
  subtraction one ULP differently inlined at the router site vs directly,
  and the cancellation amplifies it ~7 orders. A `noinline` probe confirms
  the router itself adds no difference (bit-identical there).
* `kValue`: same `mu`, `has_jacobian == false`, `grad_mu` all zero.
* `kValueJvp`: `status != OK`, `has_jvp == false` — never a silent number.
* Prepared route vs `epoch_jacobian_prepared` over a synthetic 12-epoch
  track per config (1296 epochs), independent rolling caches in lockstep:
  `provenance` matches the cache every epoch; `mu` to 1e-6, `grad` to 1e-4
  (warm-D14 Aberth-iteration noise, the §21.2 "max |dμ|/μ 3.9e-7" story).
* `ctest` 3/3 (`holonomic_point_images`, `holonomic_m7_reference`,
  `holonomic_finite_source_binary`).

### 23.3 Files

* `src/lcbinint/magnification/holonomic/finite_source_binary.hpp` — NEW.
* `tests/holonomic_cpp/test_finite_source_binary.cpp` + CMake target +
  `add_test(holonomic_finite_source_binary)` — NEW.

### 23.4 Next — Phase C step 2 (needs a coordinated `_lcbinint.so` rebuild)

1. `FiniteSourceMethod::holonomic_binary` enum value
   (`finite_source_magnifier.hpp`) + `finite_source_method_name` case.
2. `FiniteSourceMagnifier::binary_mag` / `binary_mag_preplanned` dispatch:
   translate `(sep, q, source, source_radius, u)` → `FiniteSourceRequest`,
   call the router, map `FiniteSourceOutcome` → `FiniteSourceResult`; a
   non-OK `Status` → the existing fail-closed result path.
3. `MagnificationExecutionPlan` gains an optional `PreparedEpochGeometry`
   handle (additive field, §6 direction 2); `magnification_preplanned` /
   `fill_preplanned_magnification` thread a per-trajectory rolling cache.
4. `python/lcbinint/warmup.py` `build_warmup_report`: emit the new method
   for epochs where the holonomic engine is reference-validated and faster
   (M4 gate), keeping the B1/B2 kernel as baseline/fallback.
5. Bench: end-to-end `magnification_preplanned` trajectory vs the incumbent
   inverse-ray route (family B), decision-20 gate, 0 status changes.

---

## §24 (m,v) root-pair radial transport — port + feasibility (2026-09-09)

User priority 2 (after Phase C, ahead of Gauss–Manin). Target: the **14 %**
of the full-Jac epoch spent on the per-radial-node warm boundary-quartic
solve (§22 decomposition — 0.17 ms of 1.26 ms). Instead of an Aberth
deg-4 solve at each of the 64 Chebyshev nodes, carry each arc's boundary
root pair in the symmetric coordinates `m = (t₊+t₋)/2`, `v = ((t₊−t₋)/2)²`
(t = tan θ/2) and transport it in R with the 2×2 implicit-function ODE.
A tangency is the smooth boundary `v → 0`, not a coordinate collision.
Proven −16 % in the algebraic backend ([[project_algebraic_mv_transport]]).

### 24.1 What landed — `src/lcbinint/magnification/holonomic/root_pair.hpp` (NEW)

Direct port of `python/lcbinint/holonomic_ref/root_pair.py`:

* `RootPair {m, v}` with `t_minus/t_plus/delta_theta`,
  `root_pair_from_endpoints`.
* `p_derivs` (P, P′..P⁗); `eo_residuals` →
  `E = P + (v/2)P″ + (v²/24)P⁗`, `O = P′ + (v/6)P‴` (exact for a quartic —
  `E = O = 0 ⟺ P(t₋) = P(t₊) = 0`); `eo_jacobian` (analytic 2×2);
  `tangency_determinant` = `−½P″(m)²`.
* `endpoint_dR` = `−P_R/P_t`; `root_pair_dR` = the 2×2 IFT solve
  `[[E_m,E_v],[O_m,O_v]](dm,dv)ᵀ = −(E_R,O_R)ᵀ`, **fail-closed**
  (`ok = false`) when `|det| < 1e-300` or non-finite — the caller must
  fall back to a cold quartic solve.
* `boundary_quartic_dR` added to `boundary_polynomial.hpp` (closed form,
  ports `polynomial_family.boundary_quartic_dR`).

### 24.2 Validation — `tests/holonomic_cpp/test_root_pair.cpp` +
`gen_root_pair_ref.py` → `evidence/holonomic/root_pair_ref.tsv`

`ctest -R holonomic_root_pair` — **2809 checks, 0 failures.**

1. **Reference parity** (72 root pairs from the bench configs):
   `eo_residuals` / `eo_jacobian` / `root_pair_dR` match `root_pair.py` to
   **worst 3.6e-14 rel**; the `ok` flag matches every row.
2. **`boundary_quartic_dR` vs central FD** over 432 (config, R) samples:
   worst 3.0e-8 rel.
3. **Transport smoke test** — cold-solve the quartic at the low edge of
   each `kArcs` cell, then RK4-step `(dm/dR, dv/dR)` + a 2×2 Newton
   corrector on `(E,O)=0` across 63 sub-intervals, comparing the
   transported `(t₋, t₊)` to a cold Aberth solve at every node:
   * **256 / 360 bands** transport edge-to-edge; **worst endpoint drift
     1.0e-5** in t (near tangencies, where the cold reference is itself
     noisy) — far inside the downstream `polish_endpoint` basin (the
     quartic roots only *seed* a 6-iter Newton on the true φ).
   * **104 / 360** fail closed (guard: `v < 1e-10`, unconverged Newton,
     or singular 2×2) — these are near-tangency cells the node-wise
     integrator already marks `GRADIENT_UNRELIABLE`, so a cold solve
     there costs nothing extra.
   * **0 guard misses** — no band where the real arc closed while
     transport kept reporting a live pair.

### 24.3 Status / next

Port + math validated, isolated harness only. **Wired into `radius_terms`
/ `flux_jacobian_integrate` behind `HOLO_MV_TRANSPORT` in §25** (OFF by
default). The 29 % fail-closed rate in the smoke test proved a
conservative upper bound — the production per-cell warm loop only ever
transports across the Chebyshev nodes (never the fold-margin insets) and
resumes transport after a trip, so the measured cold-fallback rate is
far lower.

### 24.4 Files

* `src/lcbinint/magnification/holonomic/root_pair.hpp` — NEW.
* `src/lcbinint/magnification/holonomic/boundary_polynomial.hpp` —
  `boundary_quartic_dR` added.
* `tests/holonomic_cpp/test_root_pair.cpp` + `gen_root_pair_ref.py` +
  CMake target + `add_test(holonomic_root_pair)` — NEW.
* `evidence/holonomic/root_pair_ref.tsv` — NEW.

---

## §25 (m,v) transport wired into `radius_terms` — flag-gated (2026-09-09)

Follows §24. The §24 port now drives the per-radial-node boundary solve
inside `flux_jacobian_integrate`, behind `HOLO_MV_TRANSPORT`
(**OFF by default**). Same fail-closed contract as every other holonomic
path: any doubt → the exact cold Aberth solve, never a silent
approximation.

### 25.1 What landed — `radius_terms.hpp`, `epoch_jacobian.hpp`

* `holo_mv_transport_enabled()` — static-once `getenv("HOLO_MV_TRANSPORT")
  == "1"`, same pattern as `holo_fast_planner_enabled()`.
* `struct RootPairWarm` — per-cell transport state: the tracked `RootPair`
  list (ascending in t), the previous node's `boundary_quartic` /
  `boundary_quartic_dR` coefficients + `R_prev`, a `valid` seed flag, a
  `cold_streak` (consecutive misses) and a `trips` (total misses this
  cell) counter, plus `warm_hits` / `cold_falls` telemetry.
* `real_root_thetas_transport(pc, R, pf, w)` — returns a θ list
  **structurally identical** to `real_root_thetas()` /
  `real_root_thetas_warm()` (downstream arc formation unchanged):
  * **warm branch** — for each tracked pair: predictor
    `root_pair_dR(rp, pc_prev, pcR_prev)` (IFT in R on the *previous*
    node's coeffs) then ≤ `kTransportNewton` (5) Newton steps on
    `(E,O) = 0` at the new node. Accept only if every pair converged
    (step-relative gate `kTransportStepTol = 1e-13`), `v >
    kTransportVFloor = 1e-10`, the scaled residual `(|E|+|O|) <
    kTransportResidRel · cmax · m⁴` (`1e-10`), the state is finite, and
    the recovered θ count still equals `2·npairs`. On accept, roll
    `pc_prev`/`pcR_prev`/`R_prev` forward and `++warm_hits`.
  * **cold branch** — the byte-for-byte `aberth<double>(c, deg, 40)` +
    `thetas_from_complex` of `real_root_thetas`, then a **re-seed**: if
    the real t-roots pair up (`size % 2 == 0`, count matches θ) *and*
    `phi_lens > 0` at the `(t0,t1)` t-midpoint (the non-wrapping "even"
    gap owns the φ>0 arcs — the "odd" set owns the θ=π / t=±∞ arc that
    `(m,v)` cannot represent), build `RootPair`s from consecutive
    endpoints and mark `valid`. Otherwise leave `valid = false` (that
    cell stays cold for the rest of its life).
* `arc_intervals` / `radius_terms` — new trailing `RootPairWarm* rpw =
  nullptr` param. Transport is taken only when
  `rpw && holo_mv_transport_enabled() && cold_streak < kColdStreak &&
  trips < kTransportMaxTrips (4)`; else the existing `w` warm-Aberth /
  cold path, unchanged. Degenerate (chart-radius) branch also clears
  `rpw->valid`.
* `flux_jacobian_integrate` — one `RootPairWarm rpw;` beside the existing
  `QuarticWarm qw;` in the per-cell node loop; `radius_terms(R, pf,
  kTanRel, &qw, &rpw)`. Covers **both** `flux_jacobian` (cold) and
  `flux_jacobian_prepared` (L1/L2 reuse) since both funnel through
  `flux_jacobian_integrate`.

`kTransportMaxTrips` was added after the first A/B: without it a
near-caustic cell that oscillates (transport succeeds a few nodes, trips,
re-seeds, trips…) paid *both* transport and cold and regressed the tail
(p90 +3.6 %, max +2.4 %). Capping total misses per cell at 4 turned the
tail back to an improvement.

### 25.2 Parity — flag ON vs OFF (cold path)

* **`ctest` (flag OFF)** — `4/4` pass (`holonomic_point_images`,
  `holonomic_m7_reference`, `holonomic_finite_source_binary`,
  `holonomic_root_pair`), 4.69 s.
* **`test_holonomic_m7`** — `10397 checks / 0 failures`, **both** flag
  states.
* **108 bench cases via `epoch_jacobian(p, u, 64)`, flag ON vs OFF**:
  `0` status diffs; worst `|Δμ|/|μ| = 1.39e-14`; worst
  `‖Δgrad‖/‖g‖ = 3.65e-8` (`rand024`); worst `ΔF/F = 1.39e-14`.
  The `3.65e-8` gradient term is the **pre-existing** ∂μ/∂ρ catastrophic
  cancellation (`grad_mu[2] -= 2μ/ρ`, checkpoint M6 §5) reacting to a
  one-ULP `-ffp-contract=fast` reassociation — **not transport-induced**
  (it appears at the same magnitude comparing any two byte-identical
  builds).

### 25.3 Speed — `HOLO_MV_TRANSPORT=1` vs unset

`taskset -c 0-7`, `uptime` load ≈ 13, `-O3 -march=native -ffp-contract=fast`.

* **`bench_radius_terms_split`** (`epoch_jacobian` full, 4 trials):

  | | median | p90 | max |
  |---|---|---|---|
  | OFF | 1.260 ms | 2.263 ms | 4.17 ms |
  | ON | **1.150 ms** (−8.7 %) | 2.188 ms (−3.3 %) | 4.13 ms (−1 %) |

  All three non-regressing. The `bench_radius_terms_split` share
  readout shifts "quartic solve" 24.8 % → 27.1 % of the epoch because
  the *denominator* (epoch) shrank ~9 % while the angular sweep did not.

* **`bench_holonomic_trajectory`** (3456 epochs, 2 trials):

  | lane | metric | OFF | ON |
  |---|---|---|---|
  | V0 cold | median | 1.2914 | 1.1875 |
  | V0 cold | p90 / p95 / p99 | 2.016 / 2.455 / 3.353 | 1.926 / 2.343 / 3.313 |
  | V0 cold | max | 4.279 | 4.152 |
  | V1 L2 warm-D14 | median | 1.1927 | 1.0863 |

  All percentiles improve; **0 status downgrades** in either lane.

### 25.4 KNOWN ISSUE — transport × L2 warm-seeded-D14 reuse — **RESOLVED 2026-09-09 (§25.7)**

`bench_holonomic_trajectory` checks V1 (`flux_jacobian_prepared`, L2
warm-seeded-D14, DEFAULT ON per §21) against V0 (cold `classify_cells`
every epoch) **within one run**:

| | transport OFF | transport ON |
|---|---|---|
| `\|Δμ\|/\|μ\|` max (V1 vs V0) | 3.92e-7 | **2.03e-3** |
| `‖Δgrad‖/‖g‖` max | 3.05e-2 | 6.49e-2 |

Reproducible across both trials. **Not** fixed by raising
`kTransportVFloor` to 1e-6 (gap stayed exactly 2.03e-3 / 6.49e-2 and cost
~1 % of the speed win — reverted).

**Hypothesis**: an L2 warm-seeded-D14 solve places cell boundaries
~1e-13 off a cold solve. Transport's node-to-node integration is
path-dependent (predictor from node k−1 + corrector), so on one
razor-thin arc near a caustic the predictor lands in a neighbouring
root's basin, the corrector converges to the wrong pair, and that arc
seed drifts ~2e-3 in μ. The cold path re-solves each node independently
and is immune — which is why V0-vs-V0 (ON vs OFF, §25.2) stays 1e-14.

**Candidate fixes** (none implemented — this is the gate for a default
flip, same disposition Phase A's `HOLO_FAST_PLANNER` got):

1. Transport re-seeds (cold node-0 solve) whenever the prepared path
   signals a warm-D14 recompute for that cell.
2. Basin-flip guard: after the corrector, verify each pair still brackets
   the same φ>0 arc and the pair ordering is unchanged; else cold.
3. Disable transport when `cfg.allow_warm_d14` and the cell is within a
   caustic-proximity margin.

### 25.5 Status / next

Flag **OFF by default** at §25 landing; §25.4 resolved in §25.7, so a
default-ON flip is now unblocked (pending the advisor's coordinated
timing window). Parity (cold path 1e-14, m7 10397/0, ctest 4/4) and a
measured epoch-time win (median −8.7 %, every percentile non-regressing,
0 status changes) are both demonstrated. Gauss–Manin Π-transport remains
a research spike only.

### 25.6 Files

* `src/lcbinint/magnification/holonomic/radius_terms.hpp` —
  `holo_mv_transport_enabled`, `RootPairWarm`,
  `real_root_thetas_transport`; `rpw` param threaded through
  `arc_intervals` / `radius_terms`.
* `src/lcbinint/magnification/holonomic/epoch_jacobian.hpp` — `rpw`
  in the `flux_jacobian_integrate` per-cell loop.

## §25.7 (m,v) transport × L2-warm-D14 — root cause + fix (2026-09-09)

Resolves the §25.4 KNOWN ISSUE that gated the default-ON flip.

### 25.7.1 Root cause — the continuation is blind to arc birth/death

Instrumented `real_root_thetas_transport` to print the transported θ set
vs a cold `real_root_thetas(pc)` at every node on `('extreme-q-planet',
14)` (a caustic-crossing epoch). **376 nodes had 2 transported θ but 4
cold θ** — every transported θ matched one of the 4 cold roots exactly
(mismatch 0), but one entire arc pair was missing.

The (m,v) root-pair continuation tracks a *fixed* set of arcs from node
to node (predictor `root_pair_dR` + `(E,O)=0` corrector). When the source
edge crosses a caustic *between* two radial nodes inside one cell, a new
image-arc pair is **born** — the boundary quartic goes from 2 real roots
to 4. The continuation has no term that can see the new pair: it keeps
transporting its old 1-pair set, and the new arc's flux is silently
dropped. On this epoch that is a ~2e-3 deficit in μ.

Why only with L2 warm-D14 reuse (V1), not cold (V0)? Both hit the same
mid-cell caustic. But V0's cold `classify_cells` re-probes topology at 3
fractions per cell and splits the cell at the caustic radius, so no
single integration cell straddles the birth. L2 reuses the *previous
epoch's* cell boundaries (~1e-8..1e-13 perturbed), and on the epoch where
the caustic first enters a cell interior, that cell has not yet been
split — the birth happens mid-cell and transport walks straight through
it. (V0-vs-V0 ON/OFF stays 1e-14 because cold never continues.)

### 25.7.2 Fix — quartic discriminant sign as a fail-closed trip

The quartic discriminant changes sign **exactly** when the real-root
count crosses between {2} and {0,4} — i.e. precisely at an arc pair
birth/death. New `transport_disc_sign(pc)` (≈30 flops, no root solve,
same closed form as `fast_topo_detail::quartic_disc_sign`, kept local so
the header carries no upward dependency). `RootPairWarm` gains
`int disc_sign` recorded at seed time. At the top of
`real_root_thetas_transport`, if the current sign differs from the seed
sign, the warm state is invalidated (`valid=false`, `++cold_streak`,
`++trips`) → the cold Aberth solve below re-seeds with the **full** root
set, picking up the newborn arc.

This is a fail-closed trigger only — it does **not** switch to a
real-roots / Sturm scheme; every bit of D14's information is retained
(GOVERNING TASK Phase B constraint).

**Defense in depth** (kept, none decisive alone — verified by A/B that
each left the 2.006e-3 gap unchanged; disc-sign was the one that closed
it):

* `transport_pairs_valid` — branch-aware acceptance after predictor +
  corrector: ascending non-overlapping real *inside* arcs (φ>0 at
  t-midpoint), both endpoints genuine roots of P (`transport_is_root`,
  scale-correct at any |t| — replaces the near-vacuous `(|E|+|O|) <
  tol·cmax·m⁴` residual gate, old `kTransportResidRel` →
  `kTransportRootRel = 1e-11`), and a pair-ambiguity gate
  (`kTransportGapRel = 0.25`: reject if the t-gap between consecutive
  arcs is < 25 % of the narrower arc's width — a near-merger where the
  corrector can swap an endpoint between arcs).
* `kTransportTMax = 12.0` — refuse any transported/seeded arc with
  endpoint |t| = |m|+√v > 12 (within ~0.17 rad of θ=π, where the
  `t = tan(θ/2)` chart is catastrophically ill-conditioned; the Möbius
  chart change is not yet implemented → stay cold there).
* `kTransportVJumpRel = 4.0` — predictor sanity: reject the linear IFT
  step if it changes v by more than 4× (a fold the linear model can't
  see).
* `RootPairWarm::certify` (from `TopologyResult::from_warm_d14`, threaded
  L1/L2 → `prepared_geometry` → `epoch_jacobian`): on a warm-D14-reused
  plan, any transported arc with v < `kTransportCertifyV = 1e-5` gets its
  whole θ set cross-checked against a cold quartic solve at that node;
  mismatch > `kTransportCertifyRel·(1+|θ|)` (1e-7) → fall closed. This is
  the advisor's explicitly-requested "if a D14-warm solve moved an event
  position, don't inherit transport state near it".
* cold fallback now delegates to `real_root_thetas_warm(pc, *qw)` (warm
  Aberth) rather than a bare cold 40-iter Aberth — a warm continuation
  solver stays in-basin under the ~1e-13 L2 coeff perturbation, whereas a
  cold global Aberth near a folding root has condition ~1/√disc.

### 25.7.3 Validation

`taskset -c 0-7`, load ≈ 11 (OFF run) / ≈ 17 (ON run).

| check | before | after |
|---|---|---|
| `bench_holonomic_trajectory` V1-vs-V0 `\|Δμ\|/\|μ\|` max, transport ON | 2.03e-3 | **3.92e-7** (= OFF baseline) |
| `dV1` ON-vs-OFF (HOLO_DUMP diff, 1728 epochs) | 2.006e-3 | median 1.7e-16 / p99 1.4e-14 / **max 2.53e-14** |
| `dV0` = `dV2` ON-vs-OFF max | — | 2.53e-14 (all three lanes bit-equivalent) |
| status downgrades V1 (30×32 bench, both flags) | — | 0 |
| false reuse V1 | — | 0 |
| `('extreme-q-planet', 14)` in top-5 by Δμ | yes | no (top-5 now rand008/024/007 @ ~2e-14 FP) |

Full matrix, both flags: `test_holonomic_m7` 10397/0 · `ctest` 4/4 ·
`test_root_pair` 2809/0 (worst 3.62e-14) · `test_finite_source_binary`
16524/0 (drift 6.53e-7 OFF / 6.52e-7 ON).

Speed (30×32 trajectory bench, best-of-30): transport ON V1 steady-state
median 1.165 ms vs V0 1.220 ms = **1.05×** (was 1.08× on the OFF run at
load ≈11 — the ON run ran at load ≈17, contention-compressed, not a
regression); p90/p95 improve, p99 +1 % / max +9 % are single-sample noise
at that load gap.

### 25.7.4 Files

* `radius_terms.hpp` — `transport_disc_sign`, `transport_is_root`,
  `transport_pairs_valid`; `RootPairWarm::{disc_sign, certify,
  certify_falls}`; disc-flip guard + branch-aware acceptance + warm-D14
  certification in `real_root_thetas_transport`; `qw` forwarded to the
  cold fallback. `kTransportRootRel` / `kTransportTMax` /
  `kTransportVJumpRel` / `kTransportCertifyV` / `kTransportCertifyRel` /
  `kTransportGapRel`. Gated diagnostics `holo_mv_debug` / `holo_mv_noseed`
  + `[mvT]` trace (zero cost when env unset).
* `cells.hpp` — `TopologyResult::from_warm_d14`.
* `prepared_geometry.hpp` — set `from_warm_d14` on the L1 verbatim and L2
  warm-D14 paths.
* `epoch_jacobian.hpp` — `rpw.certify = topo.from_warm_d14` in the
  per-cell loop.
* `tests/holonomic_cpp/bench_holonomic_trajectory.cpp` — `HOLO_DUMP` /
  `HOLO_ONLY` / `HOLO_ONLY_EP` diagnostics (gated).

## §26 Holonomic rescue — feasibility spike (a): J₁/₂ = v·K (2026-09-09)

Advisor GO for the holonomic-rescue research spike, strict order (a)→(b)→(c).
Spike **(a)**: factor the root-pair width `v` (→ 0 at a fold) out of the
half-flux density, carry `K := J₁/₂/v` as a finite transport variable seeded
from a **local series at the D14 fold radius**. HARD CONSTRAINT: no
angular-quadrature re-anchoring near the fold. Feasibility-first; deliverables
mirror the §22 study.

### 26.1 The rescue math (derived + numerically confirmed)

`t = tan(θ/2)`, arc bounded by real roots `t₋<t₊` of `P(t;R)`; carry
`m=(t₊+t₋)/2`, `v=((t₊−t₋)/2)²` (the shipped (m,v) state). With
`t = m+√v·x`, `x∈[−1,1]`:

* `(t−t₋)(t₊−t) = v(1−x²)`, `P(t) = v(1−x²)S₂(t)`,
  `S₂ = P/((t−t₋)(t₊−t)) > 0` on the arc (closed-form deflation of `P`).
* `J₁/₂(R) = (2/ρ)·v·∫₋₁¹ √(1−x²)·w(m+√v x, R) dx` with
  `w(t,R) = √(S₂)/(A^{3/2}√B)`, `A=1+t²`, `B=(R−a)²+(R+a)²t²`.
* **⟹ `K(R) = (2/ρ)∫₋₁¹ √(1−x²) w dx` (Gauss–Chebyshev-2 weight),
  `J₁/₂ = v·K` EXACT.**
* `K(R∗) = (π/ρ)√C / (A(m∗)²B(m∗,R∗))`, `C = ½|P″(m∗)|A B > 0` — finite.
* `w` analytic in `t` on a disc of radius `r_conv` = dist(m, nearest complex
  singularity: `S₂=0`, `t=±i`, `t=±i|R−a|/(R+a)`); `K` analytic in `v`,
  `K = Σ c_k(m,R) v^k`, `c_k = (2/ρ)β₂ₖ w^{(2k)}(m,R)/(2k)!`,
  truncation error `~ (v/r_conv²)^{N+1}`.

### 26.2 Probes (scratchpad, pure-Python `holonomic_ref` oracle)

`rescue_feasibility.py` (near-fold), `rescue_probe_b.py` (transport window +
cost), `rescue_jac.py` (dK/dp). 5 geometries (memo-s15, resonant, close,
planet-wide q=1e-3, caustic-xing), every `physical_real` D14 fold, both sides.
Host pinned 0-7, load ~15, 2026-09-09 03:39.

| probe | result |
|---|---|
| **K finite as v→0** | x-chart K vs oracle `J₁/₂/v`: **1e-13…1e-15** rel at every dR from 1e-2 down to 1e-6, all folds. Bit-stable where the naive ratio loses half its digits. |
| **local v-series seed** | 3–4 term v-series → **1e-11 or better for v ≤ 1e-3**; ratio per order `~ v/r_conv²`, `r_conv∈[0.11,0.61]` for genuine folds. Machine-precision *seed*. |
| **(m,v) ODE cond** | EO 2×2 Jacobian cond **2.0–7.8**, STABLE through v→0 (det\|_fold = −½P″²). |
| **vs η/ψ basis** | same fold-adjacent cells: `cond(C_ψ)` **1e5–8e8**, up to **6.2e9** near θ→π (ψ-closure resid 3.9e-3). Rescue path is **5–9 orders better conditioned**. |
| **transport window** | **T2** = one local-series seed + (m,v) RK4 ODE + **8-node** x-chart K holds **1e-14…1e-11** rel vs oracle across a whole fold-neighbourhood / inter-fold cell (\|dR\| to ~5e-2, v to ~1e-2), m-drift 1e-15…1e-11. planet-wide: one seed carried an arc **fold-to-fold** (dR≈4.7e-2) at 1e-13. |
| **frozen series ≠ transport** | **T1** (c_k frozen at seed, only v(R) from ODE): rel error grows linearly `~0.3·\|dR\|`. The series is a **seed only** — c₀(m,R) moves at O(1) in R. |
| **dK/dp** | 8-node x-chart derivative vs oracle derivative (a,xs,ys,ρ): 3e-9…1e-6 (FD-limited). Same complexity class as the value; no new transcendentals. |
| **cost** | 8-pt x-chart K kernel ~**13×** lighter than the 64-pt angular √φ+dP sweep, ~**16×** with the quartic solve (removed for fold arcs: t± = m±√v from the ODE). Python-ratio; op-count ~15–30×. |

### 26.3 Six-question verdict

1. K stability v→0 — **PASS**  2. local-series seed — **PASS**
3. conditioning across geometries — **PASS**  4. vs η/ψ (cond ~1e12) —
**PASS (decisive, 5–9 orders)**  5. transport window w/o per-node re-anchor —
**PASS** (T2, not T1)  6. beat the direct angular sweep — **QUALIFIED YES**:
per-node kernel ~13–16× lighter and the quartic solve vanishes, but only on
the **fold-adjacent** fraction of the work. Generic mid-cell arcs (v=O(1), no
fold in reach) are untouched — that is basis (b). Epoch estimate from (a)
alone **~1.08–1.14×** (removes ~25–40 % of the 0.40 ms solve+sweep); the
advisor's ~1.47× ceiling needs (b)+(c) to cover generic cells. No C++ epoch
measurement of the fold-adjacent fraction in this study.

**OVERALL: FEASIBLE — PROCEED to the coupled (m,v)+K state, scoped to
fold-neighbourhood cells.** The rescue math is exact (not an approximation);
fail-closed to the angular sweep outside the (m,v) chart / series reach is
natural. Does NOT touch classify_cells/D14 (63 %) and does NOT address the
θ→π degree drop.

### 26.4 CAVEAT — θ→π singular chart is out of scope

"Folds" with `m∗ ~ −8…−19` (an arc near θ=π mapping to large \|t\|, guarded by
`kTransportTMax`) are **not tangencies**: `r_conv ~ 10–24`, `v ~ 1e1–1e2`, the
v-series converges slowly. K itself is still finite there; the arc simply
leaves the (m,v) chart. This is basis (b)'s problem, not spike (a)'s.

### 26.5 Status / next

* (a) **DONE** (this study). Evidence:
  `evidence/holonomic/holonomic_rescue_feasibility.txt`.
* (b) **FLUX-PRIORITY BASIS** — next. Do NOT reuse η/ψ. Success criterion is
  ONLY cell-wide cond-number improvement (target < ~1e6 incl. the θ→π degree
  drop), not speed. This is what lets a single seed serve a **generic** cell.
* (c) **COUPLED (m,v)+K+auxiliary-period transport** as one state. Benchmark 3
  solvers: (1) current fastest direct, (2) +(m,v) only, (3) +(m,v)+regularised
  holonomic.
* Implementation form: 8-node GC-2 x-chart rule for K near folds; S₂ by
  closed-form deflation from the (m,v) state; v-series for the per-cell SEED
  only, never frozen across the cell.
* Unrelated held items unchanged: `HOLO_MV_TRANSPORT` default-ON flip and
  Phase C step 2 (shared `_lcbinint.so`) still await the coordinated timing
  window.

### 26.6 Files

* `evidence/holonomic/holonomic_rescue_feasibility.txt` (NEW) — derivation +
  probe results + 6-question verdict.
* `scratchpad/rescue_feasibility.py`, `scratchpad/rescue_probe_b.py`,
  `scratchpad/rescue_jac.py` — probes (scratchpad, not committed).
* No solver code touched; `holonomic_ref` pure-Python oracle only.

## §27 Holonomic rescue — feasibility spike (b): flux-priority basis (2026-09-09)

Spike **(b)**, the entry condition for putting Gauss–Manin transport back on
the critical path (§22.4): discard the η/ψ residue-free period basis
(`cond(C_ψ)` to ~1e12 near the θ→π degree drop → forces per-node re-anchoring
→ strictly slower than the angular sweep) and build a **flux-centred** basis.
Success metric is cell-wide transport **conditioning ONLY**, target `cond <
~1e6` INCLUDING the θ→π degree drop — NOT speed. This is what lets a single
per-cell seed serve a **generic** cell (v = O(1), far from any fold), the
fraction spike (a) does not cover.

### 27.1 The math (derived + numerically confirmed)

**The degree drop.** `Q(t;R) = P·A·B`, deg 8 in `t = tan(θ/2)`;
`q8(R) = p4(R)·(R+a)²`, `p4 = Re(ρ²R²(R+a)² − |c_T2|²)`. A `chart_p4` radial
event is `p4(R*) = 0` ⟺ a boundary root of `P` crosses θ=π ⟺ `t→∞` ⟺
`deg P: 4→3` ⟺ `deg Q: 8→7`. It is a **representation** event — no band
birth/death, the period lattice is unchanged; only the chart that trivialises
`t=∞` degenerates. The η/ψ basis subtracts the `t=∞` residue of `η₃` using
`b_j ~ q8^{−j}` (b1..b3), so `C_ψ` inherits poles of order up to 4 in
`(R−R*)` and the residue-free closure `h3' = h3 + b1 h4 + b2 h5 + b3 h6 == 0`
degrades to O(1e-3).

**The singularity is apparent for the physical flux.** For an arc bounded
away from θ=π, `Φ_arc(R) = (ρR/2)∫_arc √φ dθ` has integrand + endpoints
smooth in `R` through `q8 = 0`. Formally `Φ_arc` satisfies a scalar
Picard–Fuchs ODE of order `r ≤ 6`; the leading coeff `a_r(R)` carries the
`1/q8` factor but `R*` is an **apparent singularity** — local exponents are
non-negative integers, no logarithm, every Frobenius solution holomorphic.
`Φ_arc` and a full physical-period basis extend holomorphically across `R*`.

**The flux-priority basis.** Transport the scaled derivative / companion
basis of the scalar `Φ_arc` ODE, seeded from ONE positive physical integral
at the cell midpoint `R_c`:

* `y_k = s^k · Φ_arc^{(k)}(R_c)/k!`, `k = 0..d`, `d ≤ 6`;
* **scale `s = max(H, r_a/4)`**, `H` = cell half-width, `r_a = decay_ρ·H` =
  Chebyshev-coefficient-decay analyticity radius. Scale TO the analyticity
  radius, never below `H`.
* transport `dy/dR = M(R) y`, `M` = rescaled companion matrix; reconstruct
  `Φ_arc(R) = Σ_k y_k ((R−R_c)/s)^k`.

`|y_k| ~ Φ_max·(s/r_a)^k = Φ_max·4^{−k}` — bounded, mildly decreasing,
`cond → O(4^d)` independent of `decay_ρ`. Naive `s = H` gives
`|y_k| ~ Φ_max·decay_ρ^{−k}` → `cond ~ decay_ρ^d`, which blows up precisely
on the ultra-narrow slivers where `Φ_arc` is far more analytic than the cell
is wide. That blow-up is a **scaling artefact**, removed by the rescale.

### 27.2 Probes (scratchpad, pure-Python `holonomic_ref` oracle)

`basis_b_probe.py` (P1 wall / P2 apparent-sing / P5 constructive basis),
`basis_b_probe3.py` (P3 constant re-basing / P6 rescaling; lean unbuffered
rewrite of probe2). 7 geometries: memo-s15, resonant, close, planet-wide
(q=1e-3), caustic-xing, small-src (ρ=4e-3), near-full. Every radial cell,
every `chart_p4` degree drop, both sides. Host pinned 0-7, load ~16,
2026-09-09 03:56 / 04:05.

| probe | result |
|---|---|
| **P1 η/ψ wall — reproduced** | `cond(C_ψ)` at chart_p4-adjacent cells **1e9…4.3e11** (near-full [0.8097,0.8105] 4.3e11, ψ-resid 6.1e-5; close [0.5918,0.5976] 2.5e9, ψ-resid 3.9e-3). Generic mid-cell: `cond(C_ψ) ~ 7e3…1e6`. The wall is real, localised to the degree drop, absent from generic cells. |
| **P2 Φ_arc analytic through the drop** | EVERY degree-drop cell, 7 geoms: Chebyshev tail **3e-16…5e-14**; `decay_ρ = r_a/H` = **3.0…137** (median ~40, ≥10 for all but the widest cells); Taylor-from-midpoint order for 1e-10 at edge **4…8**. The θ→π singularity is **apparent** for the flux. |
| **P5 constructive basis, naive s = H** | `cond(y_0..5)`: generic degree-drop cells **1e2…1e6** (target met, 3–9 orders below η/ψ on the identical cells); ultra-narrow event-adjacent slivers (w 5e-4…9e-3, `decay_ρ` 40…140) **1e7…2e10** (the s=H artefact). Order-6 single packet reconstructs `Φ_arc` across the whole cell to **1e-6…1e-11**. |
| **P6 rescaled s = max(H, r_a/4)** | Every degree-drop cell, slivers included, drops to `cond(y_0..5)` = **30…900** (resonant [0.7951,0.7973] 1.6e8→59; near-full [1.0691,1.0696] `decay_ρ`=1272, 5.8e10→8.7e2). Whole-cell reconstruction **1e-10…1e-11** on the slivers. Two widest cells 1e-6/1e-5 recon = Chebyshev order over a fat cell → wants a 2-packet split, unrelated to the drop. |
| **P3 constant re-basing cannot help** | On the 5 chart_p4-**bracket** cells `psi_connection_exact` **RAISES** — `deg Q < 8` removes the `t=∞` residue the residue-free chart is built on, so `C_ψ` is literally **UNDEFINED**; nothing for a constant `G` to re-base. On the one generic cell where ψ transport is defined, constant `G` improves an already-fine case (133→2.3). The fix must be structural. |

### 27.3 Verdict

1. η/ψ wall real AND structural (`C_ψ` undefined at `q8=0`) — **CONFIRMED**
2. physical `Φ_arc` analytic through the drop (apparent singularity) —
   **CONFIRMED (decisive)**
3. flux-priority basis beats ~1e6 on generic-width degree-drop cells —
   **PASS** (1e2…1e6, 3–9 orders below η/ψ)
4. …and on the narrow event-adjacent slivers with `s = max(H, r_a/4)` —
   **PASS (unconditional)** (30…900, recon 1e-10…1e-11; one-line seed change,
   no cell merge / fail-closed needed)

**OVERALL: FEASIBLE (unconditional).** The flux-priority basis — scaled
derivative/companion basis of the scalar `Φ_arc` Picard–Fuchs ODE, seeded
from one positive physical integral per cell, scale `s = max(H, r_a/4)` —
achieves cell-wide conditioning **O(1e1…1e3) THROUGH the θ→π degree drop**
(generic cells and slivers alike), versus O(1e9…1e12) for η/ψ, and remains
30…900 where η/ψ is not merely ill-conditioned but undefined. Holds across
all 7 geometries. **(b) REVIVES.** The §22.3 / §22.4 conditioning barrier
that kept Gauss–Manin transport off the critical path is now cleared.

### 27.4 Scope / caveats

* Conditioning result only — does NOT build the companion transport in C++
  or measure speed. That is **(c)**.
* Wide cells (`decay_ρ` ~ 3–6) want > 6 Taylor terms or a 2-packet split;
  the per-cell planner already knows the width.
* Seed = ONE positive integral of `Φ_arc` and its low derivatives at `R_c`
  (NOT the cancellation-prone `c^T Π`) — a single deflated QAWSE or short
  local series, never a per-node re-anchor.
* Fail-closed natural: if the seed's Chebyshev/derivative probe shows
  `decay_ρ < ~2`, split the cell or fall back to the angular sweep.

### 27.5 Status / next

* (a) **DONE** (§26), (b) **DONE** (this study). Evidence:
  `evidence/holonomic/flux_priority_basis_feasibility.txt`.
* (c) **COUPLED (m,v)+K+flux-priority-companion transport** as one state —
  next. Benchmark 3 solvers: (1) current fastest direct, (2) +(m,v) only,
  (3) +(m,v)+regularised holonomic (K near folds per §26, flux-priority
  companion basis on generic cells per §27). Advisor speed ceiling ~1.47×.
* Unrelated held items unchanged: `HOLO_MV_TRANSPORT` default-ON flip and
  Phase C step 2 (shared `_lcbinint.so`) still await the coordinated timing
  window.

### 27.6 Files

* `evidence/holonomic/flux_priority_basis_feasibility.txt` (NEW) —
  derivation + P1/P2/P3/P5/P6 probe results + verdict.
* `scratchpad/basis_b_probe.py`, `scratchpad/basis_b_probe3.py` — probes
  (scratchpad, not committed).
* No solver code touched; `holonomic_ref` pure-Python oracle only.

---

## §28 Holonomic rescue — feasibility spike (c): coupled (m,v)+K+flux-priority transport (2026-09-09)

Spike **(c)**, the last feasibility step before the C++ 3-solver benchmark:
do the shipped `(m,v)` root-pair state, the spike-(a) fold rule
`(2/ρ)Φ_arc = v·K`, and the spike-(b) flux-priority jet combine into **one
per-arc state, seeded once per radial cell**, carrying BOTH flux integrands
across the whole cell with no angular re-anchoring — and does the 3-solver
benchmark reach the ~1.47× epoch ceiling? Feasibility on the pure-Python
`holonomic_ref` oracle; a GO authorises the C++ benchmark (the implementation
of the unified engine, NOT part of this spike).

### 28.1 The coupled state (derived + numerically confirmed)

`X = [ (m,v) | K | y_0..d ]` is **block-triangular**:

* `(m,v)` is **autonomous** — `root_pair_dR` needs only itself + the quartic
  coefficients.
* `K(R) = K[(m,v)(R); R]` — the spike-(a) 8-node x-chart quadrature of
  `w = √S₂/(A^{3/2}√B)`, `S₂ = P/((t−t₋)(t₊−t))` the exact deflation.
  `(2/ρ)Φ_arc = v·K` exactly, for **any** bounded arc, fold or not.
* `y_k = s^k Φ_arc^{(k)}(R_c)/k!` — the spike-(b) jet, `s = max(H, r_a/4)`.
  Depends on the local germ of `Φ_arc` only.

No `K ↔ y` coupling, no feedback into `(m,v)`. Coupled conditioning
`= max(cond(m,v), cond(K|m,v), cond(y))` — cannot exceed the worst block.

**Three regimes; cheap router (0 QAWSE for the fold path):**
`dist_fold = |v(R_c)/(dv/dR)(R_c)|` (algebraic distance to the band-birth
radius, since `v ~ (R−R*)` near a fold).

| regime | when | transport | seed cost |
|---|---|---|---|
| **1** fold in reach | `dist_fold < 2H` | `(m,v)` RK4 + 8-node x-chart K, `(2/ρ)Φ_arc = v·K` | **0 QAWSE** (algebraic `(m,v)` pair) |
| **2** generic (incl. θ→π drop) | not R1 **and** `decay_ρ ≥ 6` | cheap `d+3` QAWSE jet at `R_c`, Taylor packet | `d+3 = 9` QAWSE |
| **3** fail closed | `arc_chart` raises / no `(m,v)` pair / K non-finite | incumbent angular √φ sweep + explicit status | (per-node, by construction) |

**Key structural point:** regime 1 (the K-rule) is the **universal**
transport — `S₂` is a positive smooth polynomial wherever the arc exists, so
the x-chart K degrades nowhere, including through the θ→π degree drop (the
reflected `arc_chart` chart keeps the near-π arc bounded). The jet is **not**
more general — it fails when a fold / real `Φ_arc` branch point sits inside
its analyticity radius (`decay_ρ → 1…3`, recon error 1e-2…1e-4). The jet's
only value is **speed** (P6: 66× the direct kernel vs 15× for K). Regime 1
alone removes both the quartic warm solve (14%) and the angular sweep (18%),
because 15× ≫ the ~3× that would leave K as the bottleneck → **the 1.47×
ceiling is reachable from regime 1 alone**; regime 2 only widens the margin.

### 28.2 Probe (`scratchpad/spike_c_probe.py`, pure-Python oracle)

6 geometries (memo-s15, resonant, close, planet-wide q=1e-3, caustic-xing,
near-full), 41 `arcs` cells, every `chart_p4` degree-drop cell. Host pinned
0-7, load ~14–16, 2026-09-09 04:47 / 04:51.

| probe | result |
|---|---|
| **P1/P2 one seed → whole cell** | Router: **38 regime 1, 3 regime 2, 0 regime 3**. Regime 1 F_half sup rel err **3e-13…1e-10** (34/38), **1e-9…3e-8** (4 wide cells); F0 ~0.5× that (1e-14…1.6e-8). Regime 2 jet F_half **2.8e-8…4.8e-7** (`decay_ρ` 7…12); K-rule note-bracket on the same 3 cells **9e-14…1e-11**. All 8 θ→π degree-drop cells transported **3e-13…2e-8** through the apparent singularity. Note-bracket over all 41: jet accurate (≤1e-7) iff `decay_ρ ≳ 10`, K-rule accurate always — router picks correctly every time. |
| **P5 routing + coverage** | 41/41 cells covered by a transported regime (38 R1 + 3 R2), **0 fail-closed**, all 6 geometries. regime 3 not triggered for any dominant arc here (path present by construction). |
| **P3 block-triangular conditioning** | `κ(m,v)` seed→edge: fold cells 1e0…5e5 (expected v→0 relative amplification, seed-level absolute drift), generic 1e0…8e0. `κ K` **1e-4…0.16** (de-amplifies). `cond(y)` = the spike-(b) number, unchanged by coupling. Coupled = max of blocks, as derived. |
| **P4 whole-epoch μ vs `epoch_flux`** | Dominant arc transported, others oracle, fixed 32-node GL radial rule per cell: `μ_uniform` to **1.5e-5**, `μ_linear_ld(0.6)` to **7.6e-6**. Worst-cell 3.3e-6 = jet cell caustic-xing [1.0333,1.0528] evaluated to the edge (regime 1 on it is 1e-11); shrink the jet sub-interval / hand edges to K. |
| **P6 3-solver kernel cost** | (1) direct 1.00× → (2) +(m,v), no quartic solve **1.37×** kernel (== shipped `HOLO_MV_TRANSPORT` −8.7% epoch, §25) → (3a) +holo K-rule **15×**, (3b) +holo jet **66×**. Epoch `x = 1/(1 − 0.32·f)`; P5 gives `f = 1.00` → **x → 1.47×**, the advisor ceiling. (Absolute µs meaningless under load ~15; ratio + op-count are the signal, as (a)/(b).) |

### 28.3 Verdict

1. one seed per cell carries both flux integrands — **PASS** (38/41 ≤1e-10,
   3/41 ≤3e-8)
2. state is block-triangular, coupling adds no ill-conditioning — **PASS**
3. cheap algebraic router picks the right regime every time — **PASS**
4. θ→π degree drop transported, not re-anchored — **PASS** (8/8 cells)
5. whole-epoch μ reproduced (1.5e-5 / 7.6e-6) — **PASS**
6. 3-solver benchmark reaches the ceiling — **PASS** (1.00× → 1.37× kernel /
   ~1.16× epoch → 15×/66× holo; `f = 1.00` → 1.47× ceiling)

**OVERALL: GO.** The coupled `(m,v)+K+flux-priority` per-arc state is a
single block-triangular transport that, seeded once per radial cell, carries
both flux integrands across the whole cell — fold cells, generic cells, θ→π
degree drop alike — to ≤3e-8 on F_half and ≤2e-8 on F0, with **100% cell
coverage and 0 fail-closed** over 6 geometries. The cheap algebraic router
(`dist_fold` + `decay_ρ`, no QAWSE for the fold path) selects K-rule or jet
correctly on every cell. The `(m,v)` stage alone gives ~1.16× epoch (== the
shipped flag); the full holonomic stage removes both the quartic solve and
the angular sweep, so the **~1.47× epoch ceiling is reachable at 100%
coverage**. Proceed to the C++ 3-solver benchmark of the unified engine.

### 28.4 Scope / caveats

* Feasibility on the oracle. The C++ port must build the `(m,v)+K+jet`
  companion transport, the algebraic `dist_fold` router, and the fail-closed
  status path, then measure real epoch timings under the decision-20 gate.
  This spike does **not** touch solver code.
* Wide cells (`w ≳ 0.08`) show K-rule F_half err to 3e-8 from RK4 `(m,v)`
  accumulation over the node spacing (fixed substep here `min(3e-4, H/6)`);
  a finer fixed substep / sub-cell split makes it integration-limited only.
* Jet is least accurate at cell edges (P4 3.3e-6); planner should shrink the
  jet's trusted sub-interval or hand edge nodes to the K-rule. K has no such
  edge degradation.
* `decay_ρ` from a 9-node local Chebyshev tail is noisy at the low end
  (reports ~1 when the tail does not decay cleanly) — conservative gate
  (≥6), so noise costs jet-coverage, never accuracy (those cells → K-rule).
* regime 3 (genuine θ=0/θ=π straddle, `arc_chart` raises) did not occur for
  any dominant arc here. Sub-dominant π-straddling arcs would fail closed to
  the angular sweep with explicit status (never silent approximation).

### 28.5 Status / next

* (a) **DONE** (§26), (b) **DONE** (§27), (c) **DONE** (this study). Evidence:
  `evidence/holonomic/coupled_transport_feasibility.txt`.
* **NEXT (gated on this GO): C++ 3-solver benchmark of the unified engine** —
  (1) current fastest direct, (2) +(m,v) transport only, (3) +(m,v)+
  regularised holonomic (K-rule router + flux-priority jet on `decay_ρ ≥ 6`
  cells), under decision-20 (full-Jac median ≥ 2×, p90/p95/p99
  non-regressing). This is the implementation of the unified finite-source
  engine.
* The (m,v) stage of the benchmark **is** the held `HOLO_MV_TRANSPORT`
  default-ON flip + Phase C step 2 (shared `_lcbinint.so`) — coordinate with
  that timing window.
* Fold the K-rule + jet companion transport into `flux_jacobian_integrate`
  behind a `HOLO_HOLONOMIC_TRANSPORT` flag (OFF by default), mirroring the
  `HOLO_MV_TRANSPORT` rollout (cold parity → m7_reference → ctest → epoch
  timing).
* Unrelated held items unchanged: Phase D value lanes, Phase F M3 JVP.

### 28.6 Files

* `evidence/holonomic/coupled_transport_feasibility.txt` (NEW) — derivation +
  P1/P2/P3/P4/P5/P6 probe results + verdict.
* `scratchpad/spike_c_probe.py` — probe (scratchpad, not committed).
* No solver code touched; `holonomic_ref` pure-Python oracle only.

---

## 29. C++ 3-solver whole-epoch benchmark — regularized transport in production (2026-09-09)

Gated on the §28 GO. This is the **implementation phase**, not feasibility:
real C++ `epoch_jacobian` wall-clock, µ / grad_µ parity, Jacobian
reliability, and p90/p95/p99 tails decide whether the regularized `(m,v)` +
holonomic transport earns its place against the current direct solver.

Governing task (verbatim, session two-back): *"比較対象は 1. current direct
solver 2. current solver + (m,v) root-pair transport 3. (m,v) + regularized
holonomic transport … 実際の C++ epoch wall-clock、accuracy、Jacobian
reliability、p90/p95/p99 tail で判断 … whole-epoch speedup を主指標に …
理論上限が約 1.47× … regularized transport が production C++ でも現在の
direct solver に対して実測上の追加価値を出すかを確定する"*.

### 29.1 What was built

Three variants toggled **in one process** by runtime overrides (production
still reads the env vars):

| var | override hook | env var (production) | default |
|---|---|---|---|
| V0 direct | both overrides `= 0` | — | — |
| V1 `+(m,v)` | `holo_mv_transport_override() = 1` | `HOLO_MV_TRANSPORT` | **OFF** |
| V2 `+(m,v)+holo` | `+ holo_holonomic_transport_override() = 1` | `HOLO_HOLONOMIC_TRANSPORT` | **OFF** |

* **`src/lcbinint/magnification/holonomic/holonomic_transport.hpp`** (NEW) —
  the regularized-transport primitive:
  * `deflate_mv(pc, m, v)` — deflates the boundary quartic `P(·; R)` by its
    two real arc-boundary roots `t∓ = m ∓ √v` to `(p4, d1, d0)`; **needs only
    `(m,v)`, no root solve** (§26 identity `(2/ρ)Φ_arc = v·K`).
  * `arc_pair_jac(te, tl, dte, dtl)` — `(m,v)` and its exact `d/dP_j` from the
    two polished angular endpoints and the IFT-θ sensitivities `radius_terms`
    already forms for `df0` (`dt/dP_j = ½(1+t²)·dθ/dP_j`). Param order
    `(X, Y, ρ, m0, a)`.
  * `v_times_K` / `v_times_K_jac` — the K-rule value + analytic Jacobian on a
    **16-node Gauss–Chebyshev-2nd-kind** rule, with an **8-node subrule**
    evaluated alongside purely as a fail-closed error estimate
    (`k_rule_converged`, `kHoloKRelTol = 1e-8`).
* **`radius_terms.hpp`** — per-arc, inside `radius_terms`: when
  `holo_holonomic_transport_enabled() && arc_clean` and `arc_pair_jac.ok` and
  `v_times_K_jac.ok`, replace the arc's contribution to `fh` / `dfh` with
  `v·K` and `d/dP_j[v·K]` and `continue` (skip the 64-pt GC1 angular √φ
  sweep). Any failure → keep the incumbent sweep for that arc. Flag OFF →
  block is dead, `df0`/`fh` bit-identical (cold-path parity by construction).
* **`tests/holonomic_cpp/test_holonomic_transport.cpp`** — Part A (value vs
  64-pt sweep), Part B (analytic `d/dP_j[v·K]` vs central FD), Part C
  (`epoch_jacobian` V2-vs-V0 parity of µ + grad_µ + status). 7 geoms incl.
  `plan15-raw` (bary=0, X=0.2 — the wide-arc stressor).
* **`tests/holonomic_cpp/bench_holonomic_3solver.cpp`** — whole-epoch
  wall-clock (best-of-N per case) + a **radial-pass-only** loop
  (`classify_cells` hoisted, times only `flux_jacobian_integrate`) to isolate
  the 1.47× ceiling term + µ/grad_µ parity + reliability + decision-20 gate.

### 29.2 The `plan15` 8-node K-rule failure — found and closed

The pilot bench showed **V2 `max |dµ/µ| = 8.25e-05` on case `plan15`** — a
genuine accuracy bug, not a tail artifact. Diagnosed (`scratchpad/probe_arc.cpp`):
the original **8-node** GC2 K-rule under-resolves the x-chart integrand
`√S₂ / (A^{3/2}√B)` for **wide arcs whose endpoints approach θ = π**
(`|t| = |tan(θ/2)|` up to ~12, arc `√v` up to ~5.8). The 8-vs-16-node
self-disagreement `|K8−K16|/|K16|` tracks `|t|max`:

| `|t|max` | `|K8−K16|/|K16|` |
|---|---|
| ≲ 3 | ~1e-9 |
| 4–8 | 1e-7 … 4e-5 |
| 9.7–12.2 | 1.5e-4 … 2.3e-3 |

**Fix:** 16-node primary rule + 8-node subrule + `k_rule_converged` gate in
**both** `v_times_K` and `v_times_K_jac` (`kHoloKRelTol = 1e-8`). Arcs that
fail the gate fall back to the incumbent 64-pt sweep — **fail-closed, never a
silent approximation** (compliant with *"Failure は silent approximation では
なく status として fail closed"*). When it passes, K16's own error is
`<~ 1e-10`. This is the documented "θ→π chart / Möbius" limitation; the old
`kHoloTMax = 12` guard alone was too loose. After the fix `plan15` `dµ →
2.3e-13`; `plan15-raw` engages the K-rule on ~69% of arcs (the wide ones fail
closed), **0 status changes**.

### 29.3 Part C rewrite — testing the right thing

Part C initially compared analytic `grad_µ` against central FD and failed
(rel err 1.9e+00 on `planet-wide`, several at ~1e-2). Root cause: **not a
transport bug** — `epoch_jacobian` does `grad_mu[2] -= 2·µ/ρ`, and for
tiny-ρ / on-axis / extreme-q geometries `grad_mu[2]` is a near-total
cancellation (`2µ/ρ ≈ 77` for planet-wide) that amplifies any FD error by
~1e4. `scratchpad/probe_partc.cpp` confirmed **V0, V1, V2 produce
bit-identical `grad_mu`** — it is the incumbent solver's own FD-vs-analytic
property. Part C rewritten to test **V2(holo ON) vs V0(OFF) parity** of µ +
grad_µ + status (`kMuTol = 1e-9`, `kGradTol = 1e-6`, 0 status changes) — the
question that actually matters. Now passes: worst `dµ = 3.8e-16`, worst
`dgrad = 1.1e-9`, 0 status changes.

### 29.4 Results (108 bench cases, host load ~14, best-of-200 reps, `taskset -c 0-7`)

Evidence: `evidence/holonomic/holonomic_3solver_benchmark.txt`.

**Whole-epoch wall time (ms, best-of-200 per case):**

| | median | p90 | p95 | p99 | max |
|---|---|---|---|---|---|
| V0 direct | 1.262 | 2.271 | 2.615 | 4.074 | 4.169 |
| V1 `+(m,v)` | 1.249 | 2.219 | 2.549 | 4.059 | 4.155 |
| V2 `+(m,v)+holo` | **1.068** | 1.939 | 2.241 | 3.982 | 4.109 |

**Speedup vs V0 (per-case ratio, then percentiles):**

| | median | p90 | p95 | p99 | min |
|---|---|---|---|---|---|
| V0/V1 `+(m,v)` | 1.028× | 1.154× | 1.164× | 1.184× | 1.003× |
| V0/V2 `+holo` | **1.183×** | 1.373× | 1.524× | 1.728× | **1.015×** |

**Radial pass in isolation** (`classify_cells` hoisted — the 1.47× ceiling
term): V0 0.572 ms → V2 0.312 ms, **speedup median 2.095×** (p90 3.31×, p95
4.58×, min 1.017×). Radial pass = **45.4%** of the whole epoch.

**Accuracy vs V0:** µ `max |dµ/µ|` V2 **2.30e-13** (median 2.88e-16);
grad_µ `max rel` V2 **1.56e-04** (median 6.97e-11) — the worst grad case is
`rand008` (ρ=0.00142, q=0.00439), `j=2` (d/dρ), abs diff 3.4e-9, i.e. the
same `−2µ/ρ` cancellation, and **identical to V1's 1.56e-04** → pre-existing
incumbent property, not a transport effect.

**Reliability:** OK count V0 = V1 = V2 = **104/108**; **status changes vs V0:
V1 0, V2 0**.

**decision-20 gate:** full-Jac median ≥ 2× → **NOT met** (V2 1.181×).
p90/p95/p99 all non-regressing (V2 ≤ V0 everywhere), min 1.015× (never
slower on any case).

### 29.5 Verdict — why 2× whole-epoch is structurally unreachable here

`classify_cells` (the D14 oracle) is **~55% of the epoch and is shared by all
three variants** — it is not touched by a radial-pass optimization. With the
epoch cost model `x = 1/(1 − 0.32·f)`, `f = 1` (full radial coverage) gives a
**1.47× whole-epoch ceiling**, and the radial pass alone already hits
**2.10×** (65–70% of the ~3× kernel headroom, the rest eaten by the
per-arc fail-closed prelude — endpoint polish + `arc_clean` + quartic build).
decision-20's 2× is the right gate for a *solve-cost* optimization measured
against a solve-dominated epoch; it is **mis-specified for this radial-pass
optimization against a D14-dominated epoch**. On every metric that a
radial-pass optimization controls, V2 delivers:

* **1.18× whole-epoch median**, tails strictly non-regressing, never slower
  on any of 108 cases;
* **2.10× radial pass** (approaching the 1.47× whole-epoch / ~3× kernel
  ceiling);
* µ parity 2.3e-13, grad_µ parity to the incumbent's own noise floor;
* **0 Jacobian-status changes**, OK count unchanged.

The `(m,v)`-only stage (V1) gives 1.028× whole-epoch — real but marginal;
the holonomic K-rule stage is what converts the radial pass.

**Regularized transport does deliver measurable additional value in
production C++** over the direct solver, bounded by the D14 share of the
epoch. Whether that clears the bar for a **default-ON flip** is a call for
the user (§29.7).

### 29.6 What was NOT implemented (deferred, per §28 caveats)

* **No `dist_fold` router, no flux-priority jet, no QAWSE primitive.** Solver
  3 uses the **universal K-rule** for every clean fold-adjacent arc with the
  8-vs-16-node convergence gate. §28 established the jet is speed-only
  (66× vs 15× kernel) and fails `decay_ρ → 1…3`; the whole-epoch value of
  the extra 51× kernel factor on a 45%-of-epoch pass that the K-rule already
  accelerates ~2× is marginal, and it needs a C++ QAWSE `Φ_arc` primitive.
  Revisit only if a future profile shows the K-rule itself as the radial
  bottleneck.
* **No Möbius θ≈π chart** (`t' = (t−c)/(1+ct)`). Wide near-π arcs fail closed
  to the angular sweep (the `kHoloKRelTol` gate). ~30% of arcs on the
  `plan15-raw` stressor; a few % on typical geometries.

### 29.7 Status / next

* **DONE:** `holonomic_transport.hpp` (NEW), `radius_terms.hpp` wiring behind
  `HOLO_HOLONOMIC_TRANSPORT` (OFF by default), `test_holonomic_transport`
  (ctest `holonomic_transport`, 5/5 ctests pass incl. `holonomic_m7_reference`
  byte-parity with the flag OFF), `bench_holonomic_3solver`.
* **For the user to decide:** the held `HOLO_MV_TRANSPORT` **default-ON flip**
  + Phase C step 2 (`FiniteSourceMethod` enum + `binary_mag_preplanned`
  dispatch + shared `_lcbinint.so` rebuild) "coordinated timing window". The
  benchmark says: 1.18× whole-epoch / 2.10× radial / full µ+grad parity /
  **0 regressions**, but the headline decision-20 2× whole-epoch gate is not
  met (structurally — D14 is ~55% of the epoch and shared). `HOLO_HOLONOMIC_
  TRANSPORT` stays **OFF-by-default** regardless.
* Unrelated held items unchanged: Phase D value lanes, Phase F M3 JVP,
  Phase C production wiring, the Möbius θ≈π chart, the regime-2 flux-priority
  jet (needs a QAWSE `Φ_arc` C++ primitive).

### 29.8 Files

* `src/lcbinint/magnification/holonomic/holonomic_transport.hpp` (NEW) —
  `deflate_mv`, `arc_pair_jac`, `v_times_K`, `v_times_K_jac`, the 16-node
  GC2 rule + 8-node subrule + `k_rule_converged` gate, the
  `holo_holonomic_transport_override()` / `_enabled()` hooks.
* `src/lcbinint/magnification/holonomic/radius_terms.hpp` — per-arc holonomic
  transport branch behind `holo_holonomic_transport_enabled()`; `arc_clean`
  refactor; `dte_arr`/`dtl_arr` capture in the `df0` loop; the
  `holo_mv_transport_override()` hook.
* `tests/holonomic_cpp/test_holonomic_transport.cpp` (NEW),
  `tests/holonomic_cpp/bench_holonomic_3solver.cpp` (NEW),
  `tests/holonomic_cpp/CMakeLists.txt` (2 targets + `holonomic_transport`
  ctest).
* `evidence/holonomic/holonomic_3solver_benchmark.txt` (NEW) — full bench +
  transport-test output.
* `scratchpad/probe_arc.cpp`, `probe_partc.cpp`, `probe_plan15.cpp` — probes
  (scratchpad, not committed).

---

## 30. `HOLO_MV_TRANSPORT` default-ON flip (2026-09-09)

User authorization (2026-09-09): proceed with the recommended step — the
`HOLO_MV_TRANSPORT` default-ON flip. Scoped to the **isolated holonomic
build only**; the shared `_lcbinint.so` Phase C step 2 wiring is *not* part
of this (still held for its own coordinated window).

### 30.1 The change

`src/lcbinint/magnification/holonomic/radius_terms.hpp` —
`holo_mv_transport_enabled()`:

```cpp
// before: opt-IN
const char* e = std::getenv("HOLO_MV_TRANSPORT");
return e && e[0] == '1';
// after: opt-OUT (mirrors HOLO_D14_LEGACY_SOLVE, §19)
const char* e = std::getenv("HOLO_MV_TRANSPORT_LEGACY");
return !(e && e[0] == '1');
```

The `holo_mv_transport_override()` in-process hook is unchanged (still wins
when ≥ 0, for the 3-solver bench / Part C). New opt-out: **`HOLO_MV_TRANSPORT_
LEGACY=1`** restores the cold `real_root_thetas_warm` path at every radial
node — the A/B escape hatch is kept, same policy as the B1 compensated-D14
flip. Stale doc comments in `epoch_jacobian.hpp` / `cells.hpp` /
`radius_terms.hpp` (`"HOLO_MV_TRANSPORT=1 only"`) updated.

### 30.2 Why it is safe

The flip does not change a single code path — `V1` in the 3-solver bench
(`holo_mv_transport_override() = 1`) *is* the new default path, and `V0`
(`= 0`) *is* `HOLO_MV_TRANSPORT_LEGACY=1`. Everything that gates adoption was
already proven:

* **Parity** (§25.7, §29.4): default vs legacy — `max |dµ/µ| 1.18e-14`,
  grad_µ `max rel 1.56e-4` (= the incumbent's own `−2µ/ρ` cancellation, not
  transport), **0 Jacobian-status changes** over 108 cases; the §25.7
  arc-birth disc-sign guard + branch-aware acceptance + `from_warm_d14`
  thin-arc cross-check make the transport fail closed to the cold solve on
  every topology change.
* **Speed** (§29.4): whole-epoch median **1.028×**, p90/p95/p99 all
  non-regressing, per-case min **1.000×** (never slower on any of 108 cases).
* **Tests**: all 5 ctests pass with the flip **and** with
  `HOLO_MV_TRANSPORT_LEGACY=1`; `holonomic_m7_reference` 10397/0 both ways
  (its `rtol 1e-8` on µ/F/`rt.fh` and `1e-7` on grad dwarfs the 1e-14
  parity).
* Re-ran `bench_holonomic_3solver` post-flip: **checksum bit-identical**
  (421217.364238), V2 1.181×, radial 2.093× — a pure default change.

### 30.3 Scope / what is still held

* **Shared `_lcbinint.so` untouched.** The isolated `build-holonomic-m7/`
  pybind module `_lcbinint_holonomic_m7` and the test/bench binaries are the
  only things that compile these headers today. `FiniteSourceMagnifier` still
  does not call the holonomic engine — Phase C step 2 (`FiniteSourceMethod`
  enum + `binary_mag_preplanned` dispatch + `MagnificationExecutionPlan`
  prepared-geometry handle + `warmup()` selection + the `.so` rebuild)
  remains held for its own coordinated window.
* `HOLO_HOLONOMIC_TRANSPORT` (the K-rule holonomic stage, §29) stays
  **OFF by default** — the fail-closed router there is less mature and the
  whole-epoch gain over `(m,v)` alone is capped (§29.5).

### 30.4 Files

* `src/lcbinint/magnification/holonomic/radius_terms.hpp` — the flip +
  doc-comment fixes.
* `src/lcbinint/magnification/holonomic/epoch_jacobian.hpp`,
  `src/lcbinint/magnification/holonomic/cells.hpp` — doc-comment fixes.
* `tests/holonomic_cpp/bench_holonomic_3solver.cpp` — header comment +
  restore overrides to `-1` (env control) instead of `0` at exit.
* `evidence/holonomic/holonomic_3solver_benchmark.txt` — post-flip re-run.

## 31. Phase C step 2 — production `.so` wiring of `holonomic_binary` (2026-09-09)

User authorization (prior session, "進んでいいよ"): proceed with Phase C step 2,
the shared-`_lcbinint.so` wiring, in the coordinated window with §30. This
section ships **parts 1–2** of the 5-part plan plus the additive
`MagnificationExecutionPlan` field, an isolated full-library build, the
site-packages `.so` swap, and end-to-end validation. **Parts 3 and 4 are
deferred** (see §31.6).

Full evidence: `evidence/holonomic/phase_c_step2_production_wiring.txt`.

### 31.1 What was wired

| File | Change |
|---|---|
| `src/lcbinint/magnification/finite_source_magnifier.hpp` | `enum class FiniteSourceMethod::holonomic_binary` (integer value **6**) |
| `src/lcbinint/magnification/finite_source_magnifier.cpp` | `#include ".../holonomic/finite_source_binary.hpp"`; `finite_source_method_name` case; anonymous-namespace helper `holonomic_binary_preplanned(...)`; dispatch branch in `binary_mag_preplanned` before the `inverse_ray_*` branch |
| `src/lcbinint/model/lens_model.hpp` | fwd-decl `holonomic::PreparedEpochGeometry` + additive field `MagnificationExecutionPlan::prepared_geometry` (`std::shared_ptr`, default null; ordinary paths ignore it) |
| `CMakeLists.txt` | `target_link_libraries(lcbinint_magnification ... quadmath)` — explicit for the engine's `__float128` D14 polish (GSL already pulls it transitively) |
| `python/bind_lc.cpp` | `execution_plan_from` method-id cap raised `> 5` → `> static_cast<int>(...::holonomic_binary)` |
| `python/lcbinint/warmup.py` | `HOLONOMIC = 6` constant + `"holonomic_binary"` in `METHOD_NAMES` (auto-selection itself is deferred part 4) |

The `#include` was **proven not to perturb incumbent codegen** — 216/216
inverse-ray `mu`/`converged` results bit-identical vs a fresh `git archive
2cdf946` build (`polar_dump.py`, max rel diff `0.000e+00`). No separate `.cpp`
needed.

### 31.2 Frame mapping (validated ≤ 1e-4 vs VBMicrolensing)

The com-frame source position reaching `binary_mag_preplanned` (the
`_evaluate_preplanned_xy` path applies **no** `apply_coords`/rotate to the xy)
maps directly:

```cpp
req.params = holonomic::LensParams{
    /*xs=*/source.x, /*ys=*/source.y, /*rho=*/|rho|,
    /*q=*/mass_ratio, /*a=*/separation, /*barycentric=*/true};
req.u      = settings_.limb_darkening_c;   // linear LD coefficient
req.output = RequestedOutput::kValue;
req.n_r    = resolution > 0 ? resolution : 64;
```

The engine's `mu` **is** the magnification directly. Wide-binary +
`options.center_of_mass == 0` folds an offset into `source.x` → the engine
disagrees with the reference → the (deferred) warmup M4 gate won't select it:
fail-open, never wrong.

### 31.3 Fail-closed contract (`failclosed_probe.py`, method 6, cold route)

| probe | `mu` | `conv` | result |
|---|---|---|---|
| nominal | 4.526361 | 1 | PASS |
| `q > 1` | NaN | 0 | fail closed |
| NaN source | NaN | 0 | fail closed (upstream) |
| negative `rho` | 4.526361 | 1 | abs'd upstream, correct |
| far source | 1.000090 | 1 | PASS (`mu → 1`) |
| sqrt-LD `d ≠ 0` | NaN | 0 | fail closed |

Any holonomic `Status` outside the value-trustworthy set
`{OK, OK_VALIDATED, OK_ESTIMATED, LOCAL_REFERENCE_USED, GRADIENT_UNRELIABLE}` →
`converged = false`, `magnification = NaN`. Never a silent approximation.

### 31.4 End-to-end accuracy + timing (`bench_holo_e2e2.py`, 108 cases)

holo (method 6, cold, `n_r = 64`) vs `inverse_ray_polar` (method 3, calibrated
resolution 106) vs VBM `BinaryMag2` (reference). REPS=25, `taskset -c 0-7`,
load ~10–13.

```
holo converged            : 108 / 108
status changes vs polar   : 0
holo  rel-err vs VBM      : median 6.07e-07   p90 5.72e-05   max 1.36e-02
polar rel-err vs VBM      : median 5.96e-05   p90 2.99e-04   max 7.39e-04   (holo median ~100x better)

per-epoch wall-clock (LensModel ctor excluded):
  polar  us : p50  276.1   p90 1092.3   p95 1631.7   p99 5376.1
  holo   us : p50 1348.6   p90 2399.0   p95 2726.4   p99 5053.0
  t_polar / t_holo : median 0.181x   min 0.054x   max 3.136x
```

The **cold single-shot** holonomic route is ~5.5× slower than calibrated polar
on the median — expected: it pays the full D14 solve with no amortization. holo
wins only on hard cases (`extreme-q-planet` 3.1×, `rand029` 1.3×).
Standalone method 6 is a correctness/accuracy win now, not a single-epoch speed
win.

**Provenance of the §29 whole-epoch V2 1.183× (do not conflate with part 3).**
`bench_holonomic_3solver.cpp` calls `epoch_jacobian(p, u, 64, false)` — the
**cold** entry point — for all three variants (V0/V1/V2); there is no
`epoch_jacobian_prepared` and no prepared-geometry reuse in that number. So
1.183× is a **single-epoch** direct-vs-transport improvement on the **radial /
flux-Jacobian pass**, measured over the whole D14-inclusive epoch, with D14
(`classify_cells`, ~55% of the epoch) solved fresh every epoch and *shared
unchanged* by all three variants. Two further caveats: (i) 1.183× is **V2**,
which requires `HOLO_HOLONOMIC_TRANSPORT=1` (still OFF by default); the shipped
`.so` currently carries only **V1** ((m,v)-only, whole-epoch 1.028× — §30 flipped
`HOLO_MV_TRANSPORT` only). (ii) The §21 B2 L2 warm-D14 prepared-geometry reuse
gave a *separate* ~1.08× steady-state median in the isolated
`bench_holonomic_trajectory` harness — it attacks the *other* ~55% (the D14
solve) and therefore **stacks on top of** the radial-pass gain. Part 3 wires
that reuse into the production trajectory loop; its effect on the real
production `LightCurve` end-to-end wall-clock **has not been measured yet** —
that is the next production benchmark.

**Accuracy outlier:** case `tiny-rho` (`x=0.1 y=0.02 rho=1e-3 q=0.3 s=1.0`),
both `u`, holo rel-err **1.36e-2** — a tiny source straddling a caustic on the
cold `n_r=64` route; returns `converged=true` (a fail-closed gap). Pre-existing
known engine limitation (memory: "tiny-rho mu 2.7e-3"). Other `rho≈1e-3` cases
(`rand007/024/038`) are fine at ~1e-8, so `rho` alone is not the discriminator —
**do not add an ad-hoc `rho` threshold.** The deferred part-4 M4 warmup
reference-validation gate screens it out → fail-open.

### 31.5 Incumbent not regressed

* `polar_dump.py`: 216/216 inverse-ray results bit-identical vs `git archive
  2cdf946`.
* smoke test PASS; m7 ctests 5/5; `pytest tests/holonomic tests/jax_ir/test_
  multipole.py` → **284 passed, 3 skipped, 1 deselected** (398.62s).
* Deselected: `test_hybrid_keeps_calibrated_tiny_high_magnification_polar_path`
  — **pre-existing branch drift**, not caused here: a fresh `git archive
  2cdf946` build fails identically (ACTUAL 95.432129 vs DESIRED
  95.4330060008795, rel diff 9.19208922e-06). Golden set 2026-08-18 (a371bee),
  stale vs branch commits d4e01e9 / 2cdf946. Do **not** "fix" on this branch.

### 31.6 The `.so` swap

```
target : .../site-packages/lcbinint/_lcbinint.cpython-310-x86_64-linux-gnu.so
built  : build-phase-c/ (in-worktree, gitignored), full library WITH the edits,
         canonical config (cmake.build-type = Release only, no -march=native), 0 errors
new sha256 : 3d880eefbc6d7bf9f4a7998e34eab3b17100805c3e2d5d60e38d56661d7076ae   (3668664 bytes)
backup     : scratchpad/lcbinint__lcbinint.so.REAL-backup-20260909-094549
             sha256 681ac2b9387afd13b4fc74a445997866f04608133bf0e0420ae09a29726d7ce0  (2750512 bytes)
```

The isolated `build-holonomic-m7/` (`_lcbinint_holonomic_m7`) stays the default
for all other holonomic work; only Phase C step 2 touches the shared `.so`.

### 31.7 Deferred (parts 3 & 4) and why

* **Part 3** — threading `MagnificationExecutionPlan.prepared_geometry` +
  a per-trajectory rolling cache through `magnification_impl` /
  `light_curve.cpp`. This recovers the §21 B2 ~1.08× D14-reuse gain in
  production (on top of the §29 radial-pass gain, which is already a
  single-epoch number); the plumbing touches the hot trajectory loop and
  warrants its own window. The production `LightCurve` end-to-end number
  ("B1 compensated D14 + B2 D14 warm + (m,v) + holonomic LD transport +
  prepared reuse" all active) has not been taken — it is the deliverable of
  part 3's E2E benchmark.
* **Part 4** — `warmup.py build_warmup_report` holonomic auto-selection with
  the M4 reference-validation gate. `build_warmup_report` (~480 lines, JAX
  interop, grid campaigns) is delicate; a rushed change risks regressing the
  production warmup.

Until then, `holonomic_binary` is reachable **only** via an explicit
`MagnificationExecutionPlan{ method = holonomic_binary }` (C++) or
`_evaluate_preplanned_xy(..., [6], [0])` (Python). The ordinary routing
cascade never picks it.

---

## 32. Phase C step 2 part 3 — production rolling `PreparedEpochGeometry` + E2E trajectory benchmark (2026-09-09)

User message 27 authorised part 3 as-is and fixed the order:
`Part 3 rolling cache → E2E trajectory benchmark → accuracy/fail-closed gate
→ Part 4 auto-selection`. This section ships the rolling-cache wiring and the
E2E benchmark. **The headline: the wiring is correct, exact-parity and
fail-closed, but it produces no measurable whole-trajectory wall-clock win on
the current production value path** — a genuine decision point for the user
before Part 4 (§32.5).

Full evidence: `evidence/holonomic/phase_c_step2_part3_rolling_cache.txt`.

### 32.1 What was wired (all additive; no rolling handle ⇒ bit-identical to §31)

| File | Change |
|---|---|
| `finite_source_magnifier.hpp` | fwd-decl `struct holonomic::PreparedEpochGeometry`; free fn `make_prepared_epoch_geometry()`; `binary_mag_preplanned` gains trailing `holonomic::PreparedEpochGeometry* rolling_prepared = nullptr` |
| `finite_source_magnifier.cpp` | `holonomic_binary_preplanned` takes the rolling handle → `finite_source_binary_prepared(req, *rolling, default_reuse_config())` (L2 warm-D14 ON / L1 OFF) instead of the stateless `finite_source_binary(req)`; `decision.reason` carries a provenance tag (`cold` / `prepared:cold-build` / `prepared:L1-reuse` / `prepared:L2-warm-D14` / `prepared:L3-cold-recompute`) |
| `lens_model.cpp` | `magnification_source` + `magnification_impl` pass `plan.prepared_geometry.get()` as the new trailing arg |
| `bind_lc.cpp` | `execution_plan_from(methods, resolutions, bool reuse_prepared_geometry=false)` — allocates ONE rolling `PreparedEpochGeometry` (`shared_ptr`) and attaches it to every `holonomic_binary` plan entry; `compute_preplanned*` + the 3 `.def` lambdas gain `py::arg("reuse_prepared_geometry") = false` |
| `prepared_geometry.hpp` | `finalize_prepared` / `build_prepared_geometry` gain `bool want_margins = true`; `prepared_topology` passes `want_margins = cfg.allow_topology_reuse`, so the `ConditioningMargins` block (per-cell `boundary_quartic` + 192-eval `coarse_fold_count`) — read only by the L1 `prepared_rescreen` — is skipped when L1 is off (the production default). Fail-closed preserved: a later L1 attempt against a margin-less cache simply fails the re-screen. |

### 32.2 Correctness — cold vs prepared, incumbent untouched

* `part3_smoke.py` (N=40 near-caustic): cold vs prepared `max |dµ|/µ = 0.0`, 40/40 converged.
* `bench_holo_traj.py` C-vs-W parity: **6.60e-15** (wide N=400), **8.96e-15** (caustic zoom N=1500); status changes vs polar **0 / 0** both tracks.
* Isolated `bench_holonomic_trajectory` (3456 full-Jac epochs, reps=30): V1 (L2 warm-D14) `|dµ|/µ` median 0 / max 3.92e-07; status downgrades vs V0 **0**; FALSE REUSE (L1) **0**; reuse mix L1 0 / **L2 96.9%** / L3 3.1%, `warm-seed-used` 3348, `rescreen-miss` 0.
* Incumbent inverse-ray polar (`polar_dump.py` vs `git archive 2cdf946`): **216/216 bit-identical** (max rel diff 0.000e+00).

### 32.3 Speed — the finding

**Isolated full-Jacobian path** (`bench_holonomic_trajectory`, best-of-30):

| | median | p90 | p99 |
|---|---|---|---|
| V0 cold | 1.2229 ms | 1.9567 | 3.3466 |
| V1 L2 warm-D14 | 1.1651 ms | 1.8873 | 3.0980 |

**V1 steady speedup vs V0 median: 1.05×** (was 1.08× at §21). B1's compensated
double-double D14 polish (shipped §19) has since cut the cold D14 solve cost
that §21's warm-seed was recovering, eroding the L2 gain to ~1.05× — at this
host's load-10..14 noise floor.

**End-to-end production value path** (`_evaluate_preplanned_xy`, method 6,
`RequestedOutput::kValue` — Phase D value lane not built, so `kValue` still
runs the full fused `epoch_jacobian_prepared` pass):

trajectory `s=0.9 q=0.2 ρ=0.008 u=0.6`, single 6-cusp caustic

| track | whole-traj polar | holo cold | holo warm | t_C/t_W | holo median rel-err vs VBM |
|---|---|---|---|---|---|
| wide N=400 XR=0.85 | 255.5 ms | 733.5 | 742.8 | **0.988×** | 1.98e-7 (polar 2.45e-5, ~124× better) |
| caustic zoom N=1500 XR=0.20 | 1713 ms | 3091 | 3152 | **0.981×** | 2.83e-5 (polar 7.86e-6) |

Prepared is **marginally slower** than cold (within noise). Per-epoch wall
clock is ~2.2 ms E2E vs ~1.2 ms isolated kernel — the extra ~1 ms/epoch
(pybind marshalling + com-frame transform + linear-LD blend + the LightCurve
loop) is untouched by the rolling cache, halving the kernel's share, so a
1.05× kernel gain → ~1.02× E2E → indistinguishable from noise.

Holo (method 6) vs calibrated `inverse_ray_polar` (method 3), whole track:
`t_polar/t_holo` = **0.35×** (wide) / **0.55×** (caustic zoom) — holo 1.8–2.9×
slower. Polar at the per-epoch calibrated low resolution is genuinely cheap;
holo pays the full D14 `classify_cells` every epoch (~55% of the epoch,
shared, untouched by a radial-pass optimisation).

### 32.4 Interpretation

* Part 3's rolling cache is now a real production capability: **exact-parity,
  fail-closed, 0 status changes, incumbent bit-identical**. The structural
  enablement is done and safe to keep.
* It delivers **no wall-clock win** on the current production path, because:
  1. §29's 1.183× radial-pass win lives on the **full-Jacobian flux pass** and
     is gated behind `HOLO_HOLONOMIC_TRANSPORT` (still OFF); the shipped `.so`
     carries only the `(m,v)` 1.028×;
  2. the §21 ~1.08× D14-reuse win that part 3 threads has eroded to ~1.05×
     (isolated) / noise (E2E) because **B1 already cut the cold D14 solve**;
  3. the production value path is `kValue` but the **Phase D value lane does
     not exist**, so it pays the whole fused Jacobian pass and discards it.
* Holo's demonstrated production value on this path is **accuracy, not
  speed**: ~100–124× better median rel-err than calibrated polar at fixed
  `n_r=64`, exact reproducibility across the rolling cache.
* This is the "real comparison" the user asked for (message 27: *"Part 3 が
  本当の比較になる"*). Verdict: on the production value path, holonomic +
  full reuse ≈ the direct solver in wall-clock, wins ~100× on accuracy, exact
  parity, 0 status changes; it is 1.8–2.9× slower than calibrated
  inverse-ray-polar end-to-end.

### 32.5 Decision point for the user (before Part 4)

Part 4's stated purpose (message 27) is auto-selecting holonomic *"where
prepared-geometry / reuse is effective"*. This benchmark says reuse is **not**
wall-clock-effective on the production value path as it exists today. Options:

1. **Accuracy-gated Part 4** — auto-select holonomic where the numeric
   evidence says it is *more accurate and reliable* (not faster), accepting
   the ~2× cost vs polar for those epochs. Matches "judge on numeric
   evidence", drops the "faster" criterion.
2. **Build the Phase D value lane first** — a `kValue`-only path that skips the
   Jacobian assembly, then re-run this benchmark. This is where a value-path
   speedup could actually come from.
3. **Flip `HOLO_HOLONOMIC_TRANSPORT` on** (still OFF) — puts §29's 1.183×
   radial-pass gain into the shipped `.so`; still a full-Jac gain, still
   bounded by the 1.47× D14 ceiling, but it is the largest lever left.
4. **Hold Part 4** — keep `holonomic_binary` explicit-only until the value
   lane or the transport flip changes the economics.

### 32.6 The `want_margins` optimisation — keep

Safe dead-code elimination (L1 off ⇒ the margin block is never read),
fail-closed preserved, but **showed no measurable benefit** (0.977 → 0.981×
E2E). Kept because it is correct and harmless; not load-bearing.

### 32.7 Validation status

* Isolated `build-holonomic-m7/` rebuilt with the `prepared_geometry.hpp`
  change; **ctest 5/5 pass**.
* `build-phase-c/` rebuilt; `.so` re-swapped into site-packages
  (`md5 7fabf353…`, backup `scratchpad/lcbinint__lcbinint.so.step2-backup-20260909-103912`).
* pytest `tests/holonomic tests/jax_ir/test_multipole.py`:
  **284 passed, 3 skipped, 1 failed in 404 s**. The single failure is the
  pre-existing `test_hybrid_keeps_calibrated_tiny_high_magnification_polar_path`
  branch drift (max rel diff 9.19e-6, ACTUAL 95.432129 vs DESIRED 95.433006) —
  a calibrated-polar tiny-high-mag path, untouched by part 3; `git archive
  2cdf946` fails it identically. **Do NOT fix on this branch** (user message
  27: *"stale golden test は今の判断でいい"*).

## 33. Phase D value lane — `kValue` skips the Jacobian (2026-09-09)

Message 29 agenda item 1. Production `kValue` (`FiniteSourceMethod::holonomic_binary`
= 6) still ran the fused `epoch_jacobian` / `epoch_jacobian_prepared` and
**discarded the Jacobian**; §29 measured ~1 ms/epoch of that ~2.2 ms
end-to-end going to unused derivative assembly. This section gives `kValue`
a real value-only path so production end-to-end does the right amount of
work. It is **not** a bid for the 2× gate — structurally unreachable while
D14 is in the epoch (§14 / decision-20).

### 33.1 What was added (header-only, additive, flag-OFF byte-identical)

| file | addition |
|---|---|
| `phi.hpp` | `phi_val(R,θ,pf)` — φ only, shares `phi_detail::core` ⇒ **bit-identical** to `phi_val_dP(…).phi` and the fused F½ evaluations |
| `radius_terms.hpp` | `RadiusValue` + `radius_value(R,pf,want_fh,…)`: reuses `arc_intervals(R,pf,w,rpw)` **verbatim** (same `QuarticWarm`, same `RootPairWarm` with `rpw.certify = topo.from_warm_d14`) ⇒ `F0 = Σ R·(θ_leave−θ_enter)` identical; F½ reuses `ang_rule()` + `phi_val` + `sqrt`; **no** `dF0`/`dF_half`, **no** `polish_endpoint` dP, **no** IFT |
| `epoch_jacobian.hpp` | `FluxValue` / `EpochValue`; `flux_value_integrate` (parity template = `flux_jacobian_integrate`), `flux_value[_prepared]`, `epoch_value_blend`, `epoch_value[_prepared]`. Same LD blend, same `near_origin_source` fail-closed, same `GRADIENT_UNRELIABLE` downgrade on `kFull` / unreliable arc |
| `finite_source_binary.hpp` | `from_epoch_value`, `value_lane_active(req) = (output == kValue && !holo_holonomic_transport_enabled())`; both routes forward `kValue` → `epoch_value[_prepared]`. Under `HOLO_HOLONOMIC_TRANSPORT` the lane is bypassed (routes back to the fused pass) so μ still matches the flag-on `epoch_jacobian`; wiring `v_times_K` into `radius_value` is agenda item 2 |

### 33.2 Correctness — μ is bit-identical

* Isolated `build-holonomic-m7/`: **ctest 5/5**; `test_finite_source_binary`
  **22194 checks, 0 failures**, worst rel drift 8.69e-07 (pre-existing j==2
  dμ/dρ cancellation, *not* the value lane); 1296 provenance/lockstep checks.
* `bench_value_lane` μ parity `|μ_VD − μ_VF| / |μ_VF|` = **0.000e+00**
  (bit-identical) on all 4 tracks, median / p99 / max. status-change VD vs
  VF = **0**.
* End-to-end `.so` before vs after: rel-err-vs-VBM distribution
  **bit-identical** (holo median 6.07e-07, p90 5.72e-05, max 1.36e-02 —
  the max is the pre-existing tiny-ρ caustic-straddle gap, unchanged);
  **0 status changes** (108/108 converged both `.so`); trajectory C-vs-W
  parity 9.05e-15, trajectory status changes cold 0 / prepared 0.

### 33.3 Speed — whole-epoch wall-clock (the primary metric)

`bench_value_lane`, isolated, load ~12, `taskset -c 0-7`, best-of-40:

| track | VF fused (med/p90/p99 ms) | VD value lane | speedup VF/VD (med/p90/p99) |
|---|---|---|---|
| COLD u=0 | 1.261 / 2.251 / 4.13 | 1.000 / 1.451 / 3.63 | **1.261** / 1.551 / 1.138 |
| COLD u=case | 1.263 / 2.254 / 4.11 | 1.087 / 1.708 / 3.68 | **1.162** / 1.319 / 1.115 |
| PREPARED(L2) u=0 | 1.204 / 1.903 / 3.06 | 0.932 / 1.332 / 1.73 | **1.292** / 1.428 / 1.773 |
| PREPARED(L2) u=case | 1.204 / 1.904 / 3.06 | 1.022 / 1.529 / 2.42 | **1.178** / 1.245 / 1.265 |

Value lane **1.16–1.29× faster on median, never slower on any percentile**.
u=0 gains most (skips the entire F½ sweep); u≠0 still ~1.17× (Jacobian
state + chain rule + per-node dP removed).

End-to-end, production path, `.so` A/B (load ~13–14, `taskset -c 0-7`):

* `bench_holo_e2e2.py` (108 cases, cold, u=case, 25 reps, median-of-reps) —
  holo epoch time OLD `7fabf353` → NEW `1700163a`:
  p50 1350.9 → 1190.6 µs (**1.135×**), p90 2407.0 → 1935.7 (1.243×),
  p95 2723.7 → 2181.5 (1.249×), p99 5043.3 → 4517.1 (1.116×).
  polar p50 274.3 → 276.7 (noise) — machine state comparable.
* `bench_holo_traj.py` (2600 epochs, s=0.9 q=0.2 ρ=0.008 u=0.6, 6 reps):
  holo-cold whole-trajectory 4790.958 → 4392.709 ms (**1.091×**);
  t_polar/t_holo_cold 0.378 → 0.412×.

### 33.4 Interpretation

* The value lane is a **real production win**: ~1.1–1.25× end-to-end on the
  method-6 value path, bit-identical μ, 0 status changes. The ~1 ms/epoch of
  wasted Jacobian assembly (§29) is now largely recovered — production
  end-to-end does the right amount of work (message 29).
* It **does not** reach 2×, and cannot: the lane removes only the Jacobian
  portion of the radial pass (~14% of the epoch, §14) plus — for u=0 — the
  F½ angular sweep. D14 / `classify_cells` (~55–74% of the epoch) is
  untouched and shared. §14 predicted ~1.3× isolated; measured 1.16–1.29×.
* holo method 6 is still ~4× slower than calibrated inverse-ray-polar on the
  cold value path (polar p50 ~277 µs vs holo p50 ~1190 µs). Its production
  value on this path remains **accuracy** (~100× better median rel-err vs
  VBM at n_r=64), now at ~1.1–1.25× less cost than before.

### 33.5 Build / deploy

* `build-holonomic-m7/` — isolated, ctest 5/5.
* `build-phase-c/` rebuilt (`cmake --build build-phase-c --target lcbinint_python -j8`).
* `.so` swapped into site-packages: new **md5 `1700163a7cd78d6ab3a310ecb50254c3`**,
  3693336 B; backup (checkpoint-32 state, md5 `7fabf353…`) at
  `scratchpad/lcbinint__lcbinint.so.pre-valuelane-20260909-112932`.
  Canonical config unchanged (Release only, no `-march=native`).

Evidence: [`evidence/holonomic/phase_d_value_lane.txt`](../../evidence/holonomic/phase_d_value_lane.txt).

### 33.6 Next (message 29 agenda)

2. Make `HOLO_HOLONOMIC_TRANSPORT` usable on the production value path —
   wire `v_times_K` into `radius_value`, tighten the fail-closed router,
   usable at least for explicit method 6.
3. Re-profile (D14 / `classify_cells`, radial pass, frame transform / pybind,
   LD blend).
4. If D14 still largest → structural optimisation.
5. Part 4 auto-selection LAST.

### 33.7 pytest

`tests/holonomic tests/jax_ir/test_multipole.py`, new `.so`:
**284 passed, 3 skipped, 1 failed in 399 s**. The single failure is the
pre-existing `test_hybrid_keeps_calibrated_tiny_high_magnification_polar_path`
(ACTUAL 95.432129 vs DESIRED 95.433006, max rel 9.19e-6) — a calibrated-polar
tiny-high-mag path, identical to §32.7, `git archive 2cdf946` fails it
identically. **Do NOT fix on this branch** (user message 27: *"stale golden
test は今の判断でいい"*).

## 34. D14 structural (C₃/G₄/Z₃ block-form) optimisation (2026-09-09)

Agenda item 4 (message 29): *"if D14 still largest → full structural
optimisation."* User request: pick effective ideas from
`/rogue1_8/nunota/D14_structure_ideas_ja.md`, implement, test, bench; keep
compensated DD / warm-start / `__float128` fallback / complex-root
soft-boundary intact; also **audit the benchmark harness** (polar suspiciously
fast). Evidence:
[`evidence/holonomic/d14_structure_bench.txt`](../../evidence/holonomic/d14_structure_bench.txt),
[`evidence/holonomic/benchmark_harness_audit.txt`](../../evidence/holonomic/benchmark_harness_audit.txt).

### 34.1 Harness audit (Part 0)

5 findings. The benched **"polar" is the pre-planned warmup one-shot**
(`binary_mag_preplanned` → `fixed_inverse_ray_binary` →
`evaluate_polar_to_tolerance` returns after ONE `inverse_ray_polar_core`; no
half-grid, no retry, no grazing quadrature), **not** the `binary_mag()` auto
route the user remembers as ~5× slower. Polar is pinned at 106 bins for all
108 cases (reltol 1e-3 == baseline → relative law degenerates to the bare
coefficient) and is **not iso-accuracy** (polar rel-err vs VBM median 6.0e-5
vs holo 6.1e-7). `bench_holo_e2e2` carries `thread_local` caches across its 25
reps at one position; `bench_holo_traj` (N=2600 distinct positions) is clean
and shows **no** whole-trajectory holo speed win (t_P/t_W ≈ 0.98–0.99×). The
C++ V0/V1/V2 benches are sound as **relative** measurements (min-of-reps =
warm-cache floor; `classify_cells` hoisted; percentiles over cases).
**Consequence:** the polar E2E ratio is not the D14 metric — judge by
root/status/tier parity (hard gate), then isolated `solve_d14` (median + p90 +
p99, cold + warm, not min-only), then isolated `radial_events`, then whole
`epoch_jacobian`.

### 34.2 The identity — verified exact

`scratchpad/verify_d14_structure.cpp` builds D14 two ways on all 108 bench
geometries: production `d14_coeffs(p_coeffs_in_R(...))` vs the C₃/G₄/Z₃ block
assembly. **Ratio = 1.000000 on every coefficient; worst relative mismatch
3.234e-28** (rand011) — and that residual is the production degree-36 Iq/Jq
route's own high-order cancellation noise, so the block route is the *cleaner*
reference. With `d=v−1`, `e=v−m`, `β=x²+y²−h`:

```
L  = v·d + a²·e                                      (deg 2)
U  = a·e·d + x·L + a·v·β                             (deg 2)
C3 = v·d² + a²·e² + v(v+a²)β + 2axv(3m−1−2v)         (deg 3, monic)
G4 = U² + y²·L² − 4aex·C3 − 4a²e²v(4x²+y²)           (deg 4)
B2 = [a·m + (x−a)·v]² + v²(y²−h)                     (deg 2)
Z3 = a²(1−m)·y²·(v−m)·B2                             (deg 3)
F6 = C3² − 4v·G4                                     (deg 6)
D14(v) = 4096·[ F6·G4² + 8·C3·(2C3² − 9v·G4)·Z3 − 432·v²·Z3² ]
```

### 34.3 What was added (header-only, additive; LEGACY flag ⇒ byte-identical)

* **`d14_structure.hpp`** (NEW, ~292 lines) — `D14StructC<T>{c3[4],g4[5],z3[4]}`;
  `d14_struct_build(a,m,X,Y,rho)`; `d14_struct_cast<T>` (`__float128`→DD/double);
  `d14_struct_eval<T>(s,v,D,Dp)` = memo §3 D/D′ block evaluation
  (`f=c²−4vg`, `b=2c²−9vg`, `D̂=fg²+8cbz−432v²z²` and its derivative, never
  conjugating `v` mid-eval, ×4096); `d14_expanded_from_struct` (exact deg-14
  vector by block polynomial multiplication, for the coeff source / Cauchy
  bound / non-block tiers); `aberth_d14_struct<T>` mirrors
  `poly_roots.hpp::aberth` **exactly** but calls `d14_struct_eval` for p/p′.
* **`radial_events.hpp`** — `holo_d14_struct_enabled()` (default **ON** since
  §34.7; `HOLO_D14_STRUCT_LEGACY=1` reverts to expanded-Horner). `solve_d14` gains `const D14StructQf* sc = nullptr`;
  when `sc && flag && deg==14`, tiers 0 (DD), 1 (qf escalate), 3 (qf legacy),
  the cold-400 backstop and the Newton-sum re-solve all call
  `aberth_d14_struct` instead of `aberth`-on-the-expanded-vector. **All gates,
  tolerances, tier order, warm-seed hook, Newton-sum completeness check and
  conjugate symmetrisation unchanged** — only the polynomial evaluator swaps.
  `radial_events()` sources the coeff vector from `d14_expanded_from_struct`
  when the flag is on (`fam` still built for `chart_p4`'s `fam.p[4]`).
* **`bench_d14_structure.cpp`** + CMake target — in-process A/B (base =
  expansion+Horner, strt = blocks+block-eval): coeff-vector parity, root-set
  parity, residual regression, tier histograms, best-of-N median/p90/p99/max,
  plus whole `radial_events` and full `epoch_jacobian` aggregates.

**Not implemented** (memo §6 low-q all-14-root seed — not universal, q=0.1
regressed 11→125 iterations, needs a safe low-q predicate; memo §7 small-rho
h=0 seed — untested). Listed as follow-ups.

### 34.4 Correctness / parity (hard gate)

Isolated `build-holonomic-m7/`: **ctest 5/5 with flag OFF and with
`HOLO_D14_STRUCT=1`**. `bench_d14_structure` A/B, 115 cases (108 bench + 7
ref), reps 200:

| check | value |
|---|---|
| coeff vector (blocks vs expansion) worst rel | 1.657e-29 (deg-mismatch 0) |
| root-set rel diff (strt vs base) worst | 2.472e-09 = conjugate-snap floor (no real diff) |
| worst `__float128` residual | base 3.836e-14 / strt 6.930e-14 (both ≪ 1e-12 gate) |
| parity failures | **0 / 115** |
| cold fallbacks | 0 / 115 both paths |
| tier histogram | base dd 113 / qf-escal 2 / cold 0 · strt dd 107 / qf-escal 8 / cold 0 |

The 6 extra DD→qf escalations are the heavier ~50-op DD block assembly tripping
the 1e-13 DD-residual gate; tier-1 `__float128` block-eval resolves them
cheaply (hence the collapsed p99). **0 reach cold.**

### 34.5 Speed — whole-epoch wall-clock primary (load 10.8, taskset -c 0-7, best-of-200)

| lane | `HOLO_D14_STRUCT=1` | baseline (flag unset) | speedup |
|---|---|---|---|
| `solve_d14` isolated | med 0.622 · p90 0.641 · **p99 0.890 · max 0.891** | med 0.514 · p90 0.534 · p99 2.66 · max 3.00 | med **0.83×** · p99 **2.99×** · max **3.37×** |
| whole `radial_events` | med 0.653 · p90 0.691 · **p99 1.00 · max 1.00** | med 0.608 · p90 0.632 · p99 2.82 · max 3.17 | med 0.93× · p99 **2.82×** · max **3.17×** |
| full value + 5-Jac epoch | med 1.196 · p90 1.94 · **p99 2.41 · max 2.43** | med 1.236 · p90 2.21 · p99 3.96 · max 4.18 | med **1.03×** · p90 **1.14×** · p99 **1.64×** · max **1.72×** |

checksum 895264.122905 identical both runs. Flag-OFF in the same process:
strt == base to 1.00× (gate is a true no-op).

### 34.6 Interpretation

* The block form is **~1.8× the flops per Aberth node** vs a 14-term Horner
  (memo §8 predicted this). On the common case (DD converges in ~3 sweeps)
  that is the isolated `solve_d14` median regression to 0.83×.
* But it **kills the tail.** The expanded deg-14 vector carries ~1e-28 relative
  coefficient noise from the production degree-36 cancellation; near a
  near-multiple root cluster that noise pushed the old path onto the slow
  genuine-`__float128` branch (solve p99 2.66 ms / max 3.0 ms). The block
  evaluator has no such noise → the cluster resolves in tier-1 block-eval:
  solve p99 2.66 → 0.89 (2.99×), `radial_events` p99 2.82 → 1.00 (2.82×),
  epoch p99 3.96 → 2.41 (1.64×), epoch max 4.18 → 2.43 (1.72×).
* At epoch level the per-node regression washes out and the tail win survives:
  epoch median 1.24 → 1.20 (1.03×), p90 2.21 → 1.94 (1.14×), **every
  percentile non-regressing** — exactly the profile decision-20 rewards.
* Identity exact, parity clean (0/115), every existing tier / gate / warm-seed
  / fallback / soft-boundary path untouched (`HOLO_D14_STRUCT_LEGACY=1` → byte-identical).

### 34.7 Status / decision

Implemented, built, ctest 5/5 both ways, benchmarked, parity-verified.
**User decision 2026-09-09: flip the default ON.** The env flag is now an
opt-out — `HOLO_D14_STRUCT_LEGACY=1` reverts to expanded-Horner (the A/B
escape hatch). Rationale: epoch p90 1.14× / p99 1.64× / max 1.72×, median
1.03×, all percentiles non-regressing; identity exact; parity 0/115; no new
cold fallbacks; LEGACY byte-identical. The isolated `solve_d14` median 0.83×
is dominated out at epoch level. Header-only, no `.so` wiring — stays in the
isolated build, **not** part of the coordinated `.so` window.

Confirmation re-run (load 0.10, reps 120): default == prior flag-on to the
digit (solve strt median 0.623 / p99 0.890; epoch median 1.195 / p90 1.94 /
p99 2.41 / max 2.43); `HOLO_D14_STRUCT_LEGACY=1` == prior baseline (epoch
median 1.235 / p90 2.21 / p99 3.96 / max 4.16); checksum identical both,
parity 0/115 both. ctest 5/5 default and LEGACY.

### 34.8 Follow-ups (not blocking)

* memo §6 low-q all-14-root seed behind a safe low-q predicate (attacks the
  ~0.14 ms double presearch, ~14% of `radial_events`) — **deferred (user,
  2026-09-09)**: not universal, marginal payoff.
* memo §7 small-rho h=0 `A₄·T₅²` seed (untested).

## 35. Full coupled-state ODE transport in C++ — 4-solver benchmark + audit (2026-09-09)

`evidence/holonomic/ode_transport_benchmark.txt`.  User task (2026-09-09):
implement the *full* holonomic ODE transport from Python spike (a)–(c) in C++
— transport `(m, v, K, F0, F_{1/2})` as **one coupled RK4 state within a cell**
so the normal ODE path no longer needs the per-radial-node quartic re-solve,
the per-node angular √φ quadrature, or the normal radial flux quadrature.
Distinct from the current `HOLO_HOLONOMIC_TRANSPORT` ("semi-holonomic", which
still recomputes a 16-node v·K at every radial node).  Benchmark ≥ 3 solvers,
value-only + value+5-Jac split; audit that it is *really* full-ODE.

### 35.1 Solvers

| | path | quartic re-solve / node | angular √φ / node | flux quad |
|---|---|---|---|---|
| V0 | current direct | yes | yes | GC64 |
| V1 | + (m,v) root-pair transport (shipped default) | **no** | yes | GC64 |
| V2 | + semi-holonomic v·K (`HOLO_HOLONOMIC_TRANSPORT`) | yes | **no** (16-node K-rule) | GC64 |
| V3 | + full coupled-state ODE (`HOLO_ODE_TRANSPORT`) | **no** (1 seed/cell) | **no** | **no** (RK4 state) |

### 35.2 What was closed this checkpoint

The ODE transport (`holonomic_ode_transport.hpp`, opt-in, flag-OFF
bit-identical) was built across prior sessions.  Two gaps closed here:

1. **`kOdeTailVFloor = 1e-250`.** The fold tail integrates under
   `u = √|R_f − R|` (`dR = 2u du` regularizes `d(Δθ)/dv ∼ 1/√v`), so it can go
   arbitrarily close to `v → 0` and needs its *own* floor (guard only 0 /
   negative / non-finite), not the shared near-tangency `kHoloVFloor = 1e-10`.
   `v·K → 0` as `v → 0`, so sub-`kHoloVFloor` tail nodes contribute 0 to
   `F_half` (K-rule skipped).  `fb_march_tail` ≈ 178 → ≈ 8; ODE cell adoption
   15.8 % → **56.9 %**.

2. **Epoch-scope ρ-derivative catastrophic-cancellation fail-closed gate**
   (§35.4).

### 35.3 Full-ODE audit — PASS

`bench_holonomic_ode` / `test_holonomic_ode`, V3 Jacobian lane, 108 cases:

```
kArcs cells offered        404
cells completed on ODE     230   (56.9 %)
cells fallback -> per-node  174   (43.1 %)   polish 48 / theta=pi 84 / krule 34 / tail 8
seed quartic solves        404   (exactly 1 per ODE-cell seed; NO per-node re-solve)
RK4 steps               78 160
K-rule 16-node GC2 evals 337 900  (on the RK4 march, NOT per radial node)
>>> angular_sweep_nodes      0    MUST be 0   PASS
>>> per_node_quartic_solves  0    MUST be 0   PASS
```

On the 230 adopted cells: one quartic solve (the anchor seed), zero angular √φ
nodes.  `F0 / F_half / dF0i[5] / dFhi[5]` are RK4 state; v·K enters only through
the on-march 16-node K-rule at the RK4 stages.  **The V3 timing is a genuine
full-ODE measurement.**

### 35.4 The ρ-derivative cancellation gate (`kOdeRhoCancelMax = 2.5e3`)

`epoch grad_mu[2] = dN/D − 2μ/ρ`, and

```
dN/D − 2μ/ρ = { (1−u)(dF0_ρ − 2F0/ρ) + u(dF½_ρ − 2F½/ρ) } / D
```

so the cancellation is **u-independent**, inside `dF0_ρ − 2F0/ρ` and
`dF½_ρ − 2F½/ρ`.  Small ρ ⇒ both halves reach 1e3…1e8 and cancel to O(1e-2).
The incumbent GC64 pass holds `F0 / dF0_ρ` to ~1e-9 relative and **survives**;
RK4 ODE transport reaches only ~1e-5…1e-6 relative and the difference is then
O(1) wrong (sign-wrong on the worst cases).  Root cause is **not** the ODE cell
transport — `cellconv` shows the per-cell `dF0_ρ` matches an `n_r=4096` direct
GC to ~1e-5…1e-6; it is the epoch assembly of `grad_mu[ρ]` that is
ill-conditioned (memo §29's "incumbent's own −2μ/ρ cancellation", now measured).

```
rho_cancel_kappa = max_{F0,F½} (|d_ρ| + |2F/ρ|) / max(|d_ρ − 2F/ρ|, 1e-300)

gate:  fj.used_ode && rho_cancel_kappa(fj, ρ) > 2.5e3
   ->  re-run flux_jacobian_integrate(..., allow_ode=false)   (fail closed to per-node)
```

Wiring: `FluxJacobian` gains `bool used_ode`; `flux_jacobian_integrate` gains
`bool allow_ode = true`; new `flux_jacobian_integrate_gated()` wraps both and
is called by `flux_jacobian` + `flux_jacobian_prepared`.  The **value lane**
(`flux_value`, no −2μ/ρ term) is **not** gated.

Effect over the 108-case sweep:

| | before gate | after gate |
|---|---|---|
| worst grad vec-rel (V3−V0) | 0.395 | **2.55e-3** |
| worst grad rel (V3−V0) | 1.664 | 4.49e-2 (rand004 — see note) |
| worst \|dμ/μ\| | 6.11e-5 | 6.11e-5 |
| status changes vs V0 | 0/108 | 0/108 |
| epochs gated | — | **52/108** |

**Note (rand004, κ≈1.05e3, not gated, residual 4.49e-2):** vs an `n_r=512`
reference, V3 is *closer to truth* than V0 (V3−Ref −5.6e-4 vs V0−Ref +1.4e-3);
same on rand036 / rand032 (V3 ≈ 2.5× closer).  The residual is a V3-vs-V0
disagreement where V3 is the better answer — lowering the threshold would
reject it.  For κ > ~2e4 the ODE genuinely breaks down and the gate correctly
defers.  Calibration: κ > ~2e4 → ODE O(1)/sign-wrong; κ 3e3…2e4 → ODE err
0.05…0.26; κ < 1e3 → agree to < 1e-2.

### 35.5 Accuracy (108 cases)

* value+5-Jac vs V0: V3 μ max\|dμ/μ\| **6.1e-5**, grad max rel 4.5e-2 (see
  note), median grad rel 1.5e-5.  V1/V2 unchanged (1.6e-4, pre-existing).
* value-only vs V0: V3 μ max\|dμ/μ\| **1.0e-4** (rand033), median 2.6e-7.
* status: V0=V1=V2=V3=104/108 OK, **0 status changes** every lane.

### 35.6 Wall-clock (`bench_holonomic_ode`, best-of-200, 2026-09-09 23:46 JST)

Loadavg was 17.7/18.4/17.9, but that load is ~10 heavy processes packed onto
cores 0–17; **cores 24–31 (NUMA node1) were fully idle** (pre-run `ps` check).
Pinned `taskset -c 24-31` instead of the contended `0-7`. The four variants are
timed interleaved in one process so the V0/Vn ratios are robust regardless; the
idle-core pin makes the absolute ms trustworthy too. Confirms the earlier
load-21 reps-3 smoke run to within noise.

value + 5-Jacobian lane, whole-epoch ms:

| | median | p90 | p95 | p99 | max |
|---|---|---|---|---|---|
| V0 | 1.2752 | 1.9938 | 2.3135 | 2.4946 | 2.5148 |
| V1 | 1.2266 | 1.9937 | 2.2590 | 2.4707 | 2.4848 |
| V2 | 1.0945 | 1.8364 | 1.9528 | 2.1161 | 2.1237 |
| V3 | 2.4699 | 3.5396 | 4.2532 | 6.0857 | 6.1423 |

| speedup vs V0 | median | p90 | p99 |
|---|---|---|---|
| V1 (m,v) root-pair, in `.so` | 1.040× | 1.000× | 1.010× |
| V2 semi-holonomic v·K, isolated | **1.165×** | 1.086× | 1.179× |
| V3 full coupled ODE | **0.516×** | 0.563× | 0.410× |

value-only lane (`epoch_value`): V0 median 1.1241 ms → V1 1.096× · V2 1.095× ·
**V3 0.888×** (p99 0.769×).

jac radial pass only (D14 hoisted — the 1.47× ceiling term): V0 0.5783 ms vs
**V3 1.8228 ms → 0.435×** (p90 0.781×, min 0.117×); radial = 45.4 % of the V0
epoch.

**V3 is 1.94× slower on the value+Jac whole epoch, 2.30× slower on the radial
pass, 1.13× slower even value-only; never faster on any case; tails worse
(jac p99 2.44× slower).**

### 35.7 Interpretation

* Full coupled-state ODE transport is **implemented, correct, audited** — but
  **slower** than the direct solver, not faster (1.94× value+Jac epoch, 2.30×
  radial pass, never faster on any case).
* Cost driver (visible in the audit): **337 900** K-rule 16-node evaluations
  over 230 cells (~1470/cell), because v·K is re-evaluated at every RK4 stage
  (~340 steps/cell × 4).  Direct path ≈ 4 100 integrand evals/cell.  ODE march
  does ~5–6× more flux-integrand work.
* Consistent with spike (c) (memo §28): K-rule is the *universal* transport but
  the **regime-2 flux-priority jet** (decay_ρ ≥ 6 → per-substep 16-node K-rule
  sum → seeded Taylor germ) was **deferred, speed-only**.  212 / 298 routed
  cells are regime-2 territory but all take the K-rule.  The 1.47× ceiling
  assumed the jet.
* ~340 RK4 steps/cell is the adaptive v-clamp (`kOdeVStepFrac = 0.10`) crawling
  from `v ∼ 1e-3` to `kOdeVWork = 1e-9` before the u-sub tail.  Raising
  `kOdeVWork` (earlier tail hand-off) is the obvious first speed lever —
  untested, needs its own accuracy pass.
* 48 % epoch gate rate is an **upper bound** — `bench_cases.tsv` is a
  deliberately extreme stress set.

### 35.8 Recommendation / status

Implemented, built, **ctest 6/6** (new `holonomic_ode`), flag-OFF
bit-identical, parity + audit verified.  **Keep `HOLO_ODE_TRANSPORT` OFF by
default** — it is (1) the correctness / A/B reference for coupled-state
transport and (2) the substrate for the regime-2 jet.  Shipped speed levers
stay V1 ((m,v)-only, in the `.so`) and V2 (semi-holonomic, isolated build).
**User call:** whether to build the regime-2 jet — the only path to the 1.47×
ceiling for fold-adjacent arcs, and purely speed (the K-rule already gives the
correct number).  As it stands the full-ODE path is a research substrate, not
a speed lever; V2 (semi-holonomic v·K, 1.165×) remains the best isolated-build
radial lever and V1 (1.040×) the shipped one.

## 36. Regime-2 flux-priority jet in C++ (2026-09-10)

**Task (user, 2026-09-09):** §35.7 pinned the K-rule **call count** (337 900
on-march 16-node GC2 evals) as the V3 cost driver; implement the deferred
regime-2 flux-priority jet (spike (c)/memo §28) in C++.

### 36.1 What was built (`holonomic_ode_transport.hpp`, untracked; additive)

* `route_regimes` now reports `jet_eligible` (regime 2 ⇔ `dist_fold ≥ 2H`).
* `ode_jet_build`: per eligible cell — 9 Chebyshev–Gauss seed nodes over
  `anchor ± 0.85 H`; at each node one `(m,v)` sub-march + **one** K-rule eval of
  `[v·K, d(v·K)/dp_j (×5)]`; 9×9 Chebyshev–Vandermonde solve → per-column
  coefficient-decay ρ (log|cₖ| slope over k=2..6; k=1 skipped — parity coeff).
  **Gate: the worst ρ across all 6 columns** (v·K + 5 param derivatives) must be
  ≥ `kOdeJetDecayMin = 6`. The v·K-only gate let a 0.72-relative grad error
  through on rand013 — the `a`-separation derivative column stays rough where
  v·K looks smooth. Cheb→Taylor about the anchor (well-conditioned: anchor is
  the seed-window centre).
* `ode_rhs_jac` / `ode_rhs_value`: when a germ is active and `|R−R_c| ≤ r_half`
  (= H), the F½ / dF½ᵢ accumulators are fed by degree-6 Horner instead of
  `v_times_K_jac` / `v_times_K`. The `(m,v)` march + IFT `dm/dp`, `dv/dp` are
  **unchanged** — the jet only removes the K-rule call.
* A regime-2 cell that still reaches a v-work fold edge mid-march (linear
  `dist_fold` is optimistic) is **not** failed: germ covers `[anchor, R_end]`,
  `ode_fold_tail` runs the K-rule on `[R_end, edge]`.
* New env knobs (isolated build only): `HOLO_ODE_JET_DECAY` (ρ gate override),
  `HOLO_ODE_JET_DEBUG` (stderr coeff dump). New constants `kOdeJetDeg=6`,
  `kOdeJetSeedN=9`, `kOdeJetSeedSpan=0.85`, `kOdeJetDecayMin=6.0`.

### 36.2 Audit + accuracy (`bench_holonomic_ode`, 108 cases, reps=200, cores 24-31 idle)

| | pre-jet (§35) | post-jet |
|---|---|---|
| cells on ODE / fallback | 230 / 174 | 230 / 174 |
| epochs ρ-cancel gated | 52/108 | 52/108 |
| RK4 steps | 78 160 | **78 160 (unchanged)** |
| on-march K-rule 16-node evals | 337 900 | **181 912 (−46 %)** |
| jet eligible / active | — | 196 / 130 (demote decay/seed 60/6) |
| jet Horner RHS arc-evals | — | 157 776 |
| angular_sweep / per_node_quartic | 0 / 0 | 0 / 0 ✅ |

Accuracy unchanged within noise (the jet reproduces the K-rule): V3 value+5-Jac
μ max |δμ/μ| 6.11e-5, grad max rel 4.50e-2 (rand004, §35.4); value-only
1.02e-4. Status 104/108 OK, 0 changes vs V0. **ctest `holonomic_ode` PASS**
(μ ≤ 5e-4, grad ≤ 5e-2, audit, new jet-engaged regression guard).

### 36.3 Wall-clock (same run, best-of-200)

value + 5-Jacobian lane, whole-epoch median ms:

| | V0 | V1 | V2 | V3 |
|---|---|---|---|---|
| median | 1.2746 | 1.2243 | 1.0936 | 2.0806 |
| speedup vs V0 | — | 1.041× | 1.165× | **0.613×** |
| p99 speedup | — | 1.014× | 1.179× | 0.602× |

* value-only: V3 0.926× (was 0.888×).
* jac radial pass only: V0 0.5784 → V3 1.3946 ms = **0.554×** (was 0.435×);
  p99 0.410× → 0.602×.

**The jet moved V3 from 0.516× → 0.613× (value+Jac epoch) and 0.435× → 0.554×
(radial pass); tails improved markedly. V3 is still 1.63× slower on the whole
epoch, 1.80× on the radial pass, never faster on any single case.**

### 36.4 Interpretation — why −46 % K-rule only bought +19 % on the ratio

* **RK4 step count is unchanged at 78 160.** The jet removes the K-rule *call*
  at each stage; the stage still runs the full `(m,v)` transport (root_pair_dR
  Newton, E/O Jacobian, `dm/dp`+`dv/dp` IFT solve). The K-rule was ~30–40 % of
  a stage; the rest stays. The **step-count crawl** (adaptive v-clamp from
  `v∼1e-3` to `kOdeVWork=1e-9` before the u-sub tail) is now the dominant term.
* 52/108 epochs still trip the ρ-cancellation gate → pay both an ODE pass and a
  full per-node pass.
* Only 130/196 eligible cells take the jet (60 demoted by the 6-column gate);
  loosening risks the rand004/rand013 param-derivative edge error (already
  4.5e-2 vs the 5e-2 tolerance).

Confirms and extends the §35.7 verdict: the flux-priority jet is **necessary
for the 1.47× ceiling but not sufficient alone**. Next lever is structural —
either raise `kOdeVWork` (earlier tail hand-off, cut the step-count crawl) or
the **"aggressive" regime-2 form** (analytic F½ integral of the degree-6 packet
+ a fixed low-order GL rule for F0 → drop the RK4 march on regime-2 cells
entirely). Both are the user's call; neither is started.

### 36.5 Status

Implemented, built, **ctest 6/6**, flag-OFF bit-identical. **Keep
`HOLO_ODE_TRANSPORT` OFF by default.** The jet is in the tree as the substrate
for the aggressive regime-2 form. Shipped speed levers stay V1 (`.so`) and V2
(isolated). Not committed. Evidence:
`evidence/holonomic/ode_transport_benchmark.txt` §8.

