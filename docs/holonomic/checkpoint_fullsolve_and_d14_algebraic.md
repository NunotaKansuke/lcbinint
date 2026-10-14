# Checkpoint — forced full-solve benchmark methodology + D14-into-algebraic experiment

Status: **COMPLETE** (autonomous, 2026-09-08). Verdict below. The
`classify_cells / arcs_at(3072)` deep optimisation (`checkpoint_M8.md` step 3)
was on hold pending this comparison; the comparison is now concluded and that
optimisation is **back ON** (see `## M8 priority decision`).

Motivating problem (user, 2026-09-08): the committed 3-way comparison
(`checkpoint_algebraic_vs_holonomic.md`) is **not a pure full-finite-source
comparison**. 74 / 108 cases let the algebraic-boundary backend resolve the
epoch with the shared point-source / hexadecapole / multipole router and never
run its deep polar integral. Holonomic always runs the full solve. So the
headline latency numbers compare "algebraic router + occasional deep solve" vs
"holonomic full solve" — a production-routing comparison, not an algorithm
comparison.

## Two benchmark classes (never mixed)

### A. Algorithm benchmark — forced full finite-source solve
Every backend runs the deep finite-source solver on **every** case. All
point-source / quadrupole / hexadecapole / multipole shortcuts disabled.

- holonomic M7/M8 — `epoch_jacobian` is already shortcut-free.
- algebraic-boundary — bypass `binary_mag()`'s router, call
  `algebraic_boundary_deep_solve` / `experimental_algebraic_boundary_integral`
  + `autodiff::experimental_algebraic_boundary_jacobian` directly
  (`ALG_FORCE_FULLSOLVE=1` env bypass patched into the isolated bench build;
  shared `.so` untouched, no branch commit).
- inverse-ray — `experimental_raster_polar_binary_mag` (deep polar solve, no
  router); Jacobian = central FD x11 (inverse-ray has no analytic Jacobian —
  labelled as such, not treated as its production gradient speed).

Measured, split value-only vs value+5-component-Jacobian:
median / p90 / p95 / p99 / max latency; accuracy vs trusted reference;
fail-closed rate; silent-failure count; Jacobian reliability / delivery rate;
root-solve count; lens-map evaluation count; radial cell / subdivision count;
fallback count; phase-level wall-clock breakdown.

Dedicated full-solve case set `evidence/holonomic/fullsolve_cases.tsv`:
ordinary binary, planetary low-q, caustic-crossing, small-rho, wide-binary,
each x {u=0, u=0.5}.

### B. Production benchmark — router enabled, end-to-end
`bench_three_way` unchanged (router + multipole shortcut live). Reported
separately as real-world operational performance; **not** an algorithm
comparison. Numbers already in `checkpoint_algebraic_vs_holonomic.md`;
re-run here only for the same-machine cross-reference table.

## D14-into-algebraic experiment

The holonomic speed-up mixes two independent ideas:
1. **D14 radial-event enumeration** — the degree-14 discriminant D14(v=R^2)
   whose real positive roots are exactly the radial tangency radii (band
   birth/death). Replaces the algebraic backend's seed-anchored exponential
   march + boolean bisection + Newton tangency search in `find_radial_bands`.
2. **Holonomic period transport + fused LD/Jacobian** — the per-cell
   Gauss-Chebyshev radial pass that produces mu + full 5-component grad-mu +
   d mu/d u in one sweep.

To separate the two contributions, four backends on benchmark A:

| # | backend | radial band discovery | integration / LD / Jacobian |
|---|---------|----------------------|-----------------------------|
| 1 | current algebraic     | `find_radial_bands` (march+bisect+Newton) | unchanged algebraic |
| 2 | algebraic + D14       | `find_radial_bands_d14` (holonomic `radial_events`, frame-mapped) | **unchanged** algebraic |
| 3 | algebraic + D14 + cont | *(not implemented — see Results §Verdict 2)* | — |
| 4 | holonomic M7/M8       | `radial_events` | holonomic transport + fused LD/Jac |

Frame map (algebraic native -> holonomic PrimaryFrame). The two
fixed-radius boundary **quartics** are the same polynomial (m0 <-> m2), but
the degree-14 radial discriminant D14(v) built from them is **not**
conditioning-equivalent: holonomic centres on the dominant mass, the
algebraic polar integral centres on the mass-m2 lens, which for q < 1 is the
*minor* mass -> ill-scaled D14 -> the `__float128` cold Aberth path fires
(see Results). Map used:

    pf.a   = lens.lens_position
    pf.m0  = lens.m2                              (mass at the origin)
    pf.X   = source.x + lens.lens_position * lens.m1
    pf.Y   = source.y
    pf.rho = source_radius

Variant 2 changes **only** band discovery; `integrate_radial_interval`,
`evaluate_radial_integrand`, the quartic solves, the `(m,v)` transport, the LD
moment series and the ForwardJet<5> Jacobian are byte-for-byte the current
backend.

Variant 3 was scoped to use the D14 cell boundaries to bound `(m,v)`
cold re-solves. **Not implemented** — the D14 precondition fails in the
algebraic frame and the cold-fallback cost it targets is < 0.2 ms/epoch
(Results §Verdict 2).

### D14 correctness policy (same as holonomic)
- plain-`double` Aberth root output is **not** a completeness certificate;
- NaN-safe gates on every mapped parameter and every returned radius;
- `__float128` discriminant build + `__float128` Aberth, balanced-double
  pre-search, 113-bit polish, cold `__float128` fallback (holonomic
  `radial_events` / `poly_roots` as-is);
- any ambiguity (D14 root cluster, seed radius not covered by the D14 band
  set, `radius_has_image` inconclusive at a probe midpoint) -> **fall back to
  `find_radial_bands` for the whole epoch and count the fallback**. D14 can
  never make the answer silently wrong.

### Fallback ratio tracking
Current algebraic deep solve (per the fresh profile): ~2060 quartic solves,
~1640 `(m,v)` transports, ~420 cold `(m,v)` re-solves per epoch on the
caustic-crossing case. The experiment reports, per variant:
band-search cost %, quartic-solve count %, `(m,v)` cold-fallback ratio %,
value-only latency, value+analytic-Jac latency, LD overhead, non-termination
count, Jacobian delivery rate.

## Decision rule
Pick the fastest configuration **at equal accuracy and equal fail-closed
policy**. If `algebraic + D14` (or `+cont`) beats holonomic, report it as the
adoption candidate honestly. If holonomic still wins, quantify how much of the
win is D14 (variant 4 vs variant 2 delta) and how much is the period
transport + fused LD/Jacobian (variant 2 vs variant 4 residual).

## Results

Status: **COMPLETE**. Machine: shared host, `taskset -c 0-7`, load avg ~12–15,
uptime 3d4h. Case set `/tmp/bench_cases_ext.tsv` (108 rows = 54 configs ×
{u=0, u=0.5}; 46 ordinary-binary, 62 planetary low-q, 10 caustic-ish, 40
small-rho, 40 wide). Trusted reference = `binary_ray_shooting` (`m0_mu`).
Harness: `tests/holonomic_cpp/bench_fullsolve.cpp` (class A),
`tests/holonomic_cpp/bench_three_way.cpp` (class B). Best-of-N latency
(N=15), pathology guard: a warm-up deep solve > 40 ms marks the case a
forced-full-solve blow-up and is recorded once, not looped.

Isolated build: throwaway worktree `algebraic-bench-cc5e55d` (detached, no
branch commit, shared `.so` untouched). `build-bench/` = pristine current
algebraic; `build-bench-d14/` = same tree + 2 objects rebuilt with
`-DALG_ENABLE_D14` (D14 band finder + `ALG_FORCE_FULLSOLVE` env bypass).

### A — Algorithm benchmark (forced full finite-source solve, all shortcuts OFF)

Latency, ms. **A1** current algebraic, **A2** algebraic+D14, **H** holonomic
M7 (fused value+5-Jac), **I** inverse-ray deep polar (value only).

| slice | metric | A1 | A2 (+D14) | H (fused) | I |
|---|---|---|---|---|---|
| ALL (n≈106) | value-only p50 | **0.179** | 8.590 | 3.006 | 0.290 |
| | value-only p90 | 1.380 | 20.007 | 5.840 | 1.572 |
| | value-only p99 | 678.8¹ | 636.9¹ | **6.154** | 3.342 |
| | value+5-Jac p50 | 0.303 | 16.299 | **3.006** | — (FD ×11) |
| | value+5-Jac p90 | 4.139 | 39.904 | **5.840** | — |
| | value+5-Jac p99 | 658.0¹ | 682.1¹ | **6.154** | — |
| ordinary binary (n=46) | value-only p50 | **0.163** | 1.790 | 2.482 | 0.272 |
| | value-only p90 | 1.539 | 6.795 | 6.009 | 1.766 |
| | value+5-Jac p50 | **0.465** | 3.339 | 2.482 | — |
| | value+5-Jac p90 | 6.432 | 16.299 | **6.009** | — |
| | value+5-Jac p99 | 11.15 | 22.51 | **6.13** | — |
| planetary low-q (n=60) | value-only p50 | **0.182** | 11.126 | 3.037 | 0.293 |
| | value-only p99 | 802.9¹ | 753.6¹ | **6.216** | 3.559 |
| caustic-ish (n=10) | value-only p50 | **1.129** | 1.992 | 5.132 | 1.035 |
| | value+5-Jac p50 | **4.139** | 3.509² | 5.132 | — |
| small-rho ρ≤0.02 (n=40) | value-only p50 | **0.168** | 9.534 | 3.994 | 0.409 |
| wide a≥2 (n=38) | value-only p50 | **0.216** | 17.646 | 2.792 | 0.243 |

¹ p99/max driven by `rand007` (q≈5e-8): A1/A2 take 700–930 ms with a
741 000-quartic-solve blow-up **and** fail closed (`a1_ok=0`, µ 0.90 vs 3.21).
H solves it in 3.45 ms. `rand002` (×2) is hard non-termination for A1/A2
(excluded); H solves it in 1.29 ms.
² caustic-ish is the one slice where A2 value-only p50 (1.99) beats A1's
Jacobian path — but see the correctness note below.

**Phase-level wall clock (holonomic, `/tmp/holoprof.cpp`, best-of-200):**

| case | full | classify_cells | (radial_events⊂classify) | radial+angular |
|---|---|---|---|---|
| plan15 | 2.37 | 0.77 (32%) | 0.58 | 1.60 |
| resonant | 5.78 | 2.35 (41%) | 2.01 | 3.43 |
| caustic-cross | 5.25 | 3.96 (75%) | 3.78 | 1.29 |
| wide-planet | 1.96 | 1.02 (52%) | 0.71 | 0.93 |

`radial_events` (the D14 solve) is 75–95 % of `classify_cells`.

**Band-discovery phase, isolated (`experimental_algebraic_boundary_radial_bands`):**

| slice | A1 march+bisect+Newton | A2 D14 event enum |
|---|---|---|
| ALL p50 | **0.034 ms** | 8.336 ms |
| ordinary binary p50 | **0.031 ms** | 0.828 ms |
| wide a≥2 p50 | **0.043 ms** | 16.90 ms |

The march A2 was meant to replace costs **0.03 ms**. D14 in the algebraic
integration frame costs 0.03→8 ms and **falls back to the march anyway on
89 % of ALL / 74 % of ordinary-binary cases** (still paying the D14 solve).

### Counters (per-epoch mean, value path)

| slice | metric | A1 | A2 (+D14) | Δ |
|---|---|---|---|---|
| ALL | root_solve_count | 13344 | 13339 | **−0.0 %** |
| | real_quartic_solve | 13302 | 13296 | −0.0 % |
| | lens_eval_count | 157588 | 157512 | −0.0 % |
| | mv_transport_fallback (cold) | 118.8 | 126.1 | **+6 %** |
| ordinary binary | root_solve_count | 479.0 | 460.6 | −3.8 % |
| | real_quartic_solve | 474.6 | 456.5 | −3.8 % |
| | mv_transport_fallback | 106.9 | 119.7 | +12 % |
| caustic-ish | root_solve_count | 962.6 | 798.8 | **−17.0 %** |
| | real_quartic_solve | 954.2 | 790.4 | −17.2 % |
| | mv_transport_fallback | 187.1 | 160.8 | −14 % |

D14 `physical_real` events returned: 3.7 mean; **D14-derived bands used on
11 % of ALL / 26 % of ordinary-binary / 20 % of caustic** (else fallback).

Where D14 *does* drive the bands its narrower support cuts quartic work
(caustic −17 %), but the fallback ratio it was supposed to help (`(m,v)`
cold re-solve, ~107–119/epoch) is **unchanged-to-worse**, and the ~2060
quartic solves per caustic epoch live in `evaluate_radial_integrand`, not in
band discovery, so band-discovery replacement cannot touch them.

### Accuracy (rel. `binary_ray_shooting`)

| slice | A1 | A2 (+D14) | H | I |
|---|---|---|---|---|
| ALL median | 8.6e-6 | 8.6e-6 | 2.7e-5 | 1.7e-4 |
| ALL p90 | 9.7e-5 | 1.1e-4 | 1.7e-4 | 1.0e-3 |
| ALL max | 3.5e-4 | **9.6e-2** | 7.7e-3 | 1.9e-3 |
| ordinary binary median | 4.1e-6 | 4.1e-6 | 5.9e-5 | 2.2e-4 |
| caustic median | 4.1e-5 | 4.1e-5 | 1.3e-4 | 6.6e-5 |
| small-rho median | 5.9e-6 | 5.9e-6 | 1.7e-5 | 9.6e-5 |

Holonomic max 7.7e-3 = `tiny-rho` (µ 38.03 vs 38.34, a known holonomic
ρ-underestimate at ρ≪caustic); also `very-wide` µ 1.0275 vs 1.0325 (5e-3).
Both are H-side, pre-existing, not touched here.

### Jacobian delivery / quality

| | A1 | A2 (+D14) | H |
|---|---|---|---|
| reliable / total (excl. hang) | **20 / 106 (19 %)** | 16 / 106 (15 %) | **102 / 106 (96 %)** |
| analytic vs own Richardson-FD, median | 5.1e-4 | (≈A1) | 6.6e-3 |
| analytic vs own Richardson-FD, p90 | **1.37 (137 %)** | — | 1.00 |

The algebraic forward-mode Jacobian is only reliable on the forced-full-solve
path for **1 case in 5**; in class B (router live) it reaches ~57 % because
the router hands hard epochs to the multipole-only analytic Jacobian. The
holonomic fused Jacobian is reliable on 96 % and fails **closed**
(`GRADIENT_UNRELIABLE`) on the other 4.

### Failure / fail-closed

| backend | value-fail | grad-unreliable (closed) | **silent** | hard non-term |
|---|---|---|---|---|
| A1 current | 6 / 107 | 86 / 107 | **0** | 2 (`rand002`) |
| A2 (+D14) | 6 / 107 | 90 / 107 | **2** (`rand036`) | 2 |
| H holonomic | — | 4 / 107 | **0** | 0 |
| I inverse-ray | 0 / 107 | — (no analytic Jac) | **0** | 0 |

**A2 regression: `rand036` (q=1.5e-3).** D14 reported a complete band set
that passed the seed-coverage gate but had dropped a real support band →
A2 integrated incomplete support → µ 3.74 vs 4.14 (**−9.6 %**), returned
`success`. A1 gets this case right (µ 4.1379). This is a **fail-closed-policy
violation introduced by the D14 port**: point-image seed coverage is not a
completeness certificate for the finite-source band set, and D14's own
`__float128` completeness handling is calibrated for the holonomic
dominant-mass-centred frame, not the algebraic small-mass-centred integration
frame.

### B — Production benchmark (router live), same machine, cross-reference

| slice | metric | algebraic | holonomic | incumbent (M0) |
|---|---|---|---|---|
| ALL | value+Jac p50 | **0.163** | 2.998 | 14.894 |
| | value+Jac p90 | 3.082 | **5.843** | 145.3 |
| | value+Jac p95 | 5.782 | **6.063** | 4391.9 |
| | value+Jac p99 | 9.515 | **6.132** | 4627.6 |
| | value+Jac max | 12.573 | **6.146** | 4643.4 |
| ordinary binary | value+Jac p50 | **0.272** | 2.481 | 15.110 |
| | value+Jac p90 | 6.412 | **5.996** | 58.6 |
| planetary low-q | value+Jac p50 | **0.051** | 3.034 | 13.618 |
| | value+Jac p90 | **1.081** | 5.357 | 4364.8 |

Jacobian delivery B: algebraic 62/108 reliable (**57 %**), holonomic
104/108 (**96 %**), incumbent always (FD, but 4.6 s tail). Silent: 0 / 0.
Value accuracy B: algebraic median 7.9e-6, holonomic 2.7e-5.

**Decision-20 gate (holonomic vs incumbent, median ≥ 2× + non-regressing
tail + same accuracy/coverage + analytic Jac):**
median 14.894 / 2.998 = **4.97×**; p90 145 → 5.8 (**25×**); p99 4628 → 6.1
(**760×**); accuracy median 2.7e-5 vs FD reference; analytic Jac 96 %
reliable, fail-closed. **GATE MET, all sub-criteria pass.**

## Verdict

1. **D14 event planner into the algebraic backend — rejected.** It replaces a
   0.03 ms march with a 0.03–17 ms degree-14 solve, falls back on ~89 % of
   cases (still paying the solve), gives **−0.0 %** root-solve reduction at
   the ALL level, does **not** reduce the `(m,v)` cold-fallback ratio, and
   **introduces a silent −9.6 % error** (`rand036`) that violates the
   fail-closed policy. Not an adoption candidate on any axis (speed, counters,
   robustness, safety).

2. **Variant 3 (D14 + `(m,v)` continuation) — not implemented.** Its
   precondition (cheap D14 cell structure) is false in the algebraic
   integration frame, and variant 2 already shows the cold-fallback ratio is
   the wrong target — it is 107–119 transported-pair re-solves/epoch at
   ~1 µs each, < 0.2 ms total, dwarfed by the ≥ 8 ms D14 cost. The
   cold-fallback reduction, if wanted, is better pursued **without** D14
   (carry the previous epoch's accepted fixed-radius root sets into
   `RootContinuationWorkspace`); logged as a separate future item, not on the
   holonomic critical path.

3. **Current algebraic-boundary is the fastest value-only backend**
   (p50 0.16–0.18 ms) but is *not* a viable standalone full-solve engine:
   19 % analytic-Jacobian delivery on the forced path, a 700–930 ms
   blow-up + fail-closed on `rand007`, hard non-termination on `rand002`.
   It works in production **because** the router keeps ~70 % of epochs out of
   its deep solver. Keep it as the incumbent experimental backend and the
   value-only fast path; do not adopt it as the Jacobian backend.

4. **Holonomic M7 is the backend to carry forward for value+Jacobian.**
   Forced-full-solve p50 3.0 ms with a **bounded** p99 of 6.2 ms (vs
   658–4600 ms for the alternatives), 96 % analytic-Jacobian delivery,
   fail-closed on the rest, 0 silent failures, 0 non-termination. Against the
   incumbent it is 5.0× faster median / 760× better p99 with analytic
   gradient quality — decision-20 gate met.

5. **The holonomic speed-up is *not* from D14.** D14 (`radial_events`) is 75–
   95 % of `classify_cells` and `classify_cells` is 32–75 % of the holonomic
   epoch — i.e. the D14 solve is *most of holonomic's cost*, not its
   advantage. The advantage is the fused period-transport radial+angular pass
   that produces µ + ∇µ + ∂µ/∂u in one sweep with a bounded node budget
   (variant 2 vs 4: same D14 event data, algebraic integration = 8–17 ms and
   19 % Jac; holonomic integration = 3 ms and 96 % Jac). **The entire
   holonomic win is the transport + fused LD/Jacobian layer.**

## M8 priority decision

`classify_cells / arcs_at(3072)` optimisation is **back ON**. It is 32–75 %
of the holonomic epoch (75 % on the caustic-crossing case that sets the p90/
p99), and holonomic is now the confirmed value+Jacobian backend. D14 cannot
be removed (it is the band-structure oracle and the completeness certificate)
but `arcs_at(3072)` oversampling and the per-cell classification are the
reducible part. Resume `checkpoint_M8.md` step 3.

## Reproduce

```
AB=.../algebraic-bench-cc5e55d ; HOL=.../lcbinint-holonomic-solver-b7d3cd
# variant .a: 2 objects with -DALG_ENABLE_D14 -I $HOL/src, ar r into build-bench-d14/
g++ -std=c++17 -O3 -march=native -funroll-loops -ffp-contract=fast -fno-math-errno \
  -I $AB/src -I $AB/include -I src -I third_party/skowron_gould \
  -isystem $CONDA/include tests/holonomic_cpp/bench_fullsolve.cpp \
  -Wl,--start-group $AB/build-bench-d14/liblcbinint_lightcurve.a \
                    $AB/build-bench-d14/liblcbinint_magnification.a -Wl,--end-group \
  -lgsl -lgslcblas -lm -lquadmath -o build-holonomic-m7/bench_fullsolve
taskset -c 0-7 ./build-holonomic-m7/bench_fullsolve /tmp/bench_cases_ext.tsv 15
```

Raw logs: `evidence/holonomic/fullsolve_bench_A.txt`,
`evidence/holonomic/fullsolve_bench_B.txt`.

The D14-into-algebraic patch (variant 2, applied to the throwaway
`algebraic-bench-cc5e55d` worktree) is captured verbatim as
`evidence/holonomic/d14_into_algebraic_experiment.patch` so the rejected
experiment stays reproducible after that worktree is removed. The worktree
itself is kept for now because `bench_fullsolve` (class-A regression harness)
links `build-bench-d14/*.a` — M8 needs before/after class-A numbers.
