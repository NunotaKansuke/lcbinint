# Checkpoint M1 — mathematical foundation + radial event enumeration

Date: 2026-09-07
Branch: `claude/lcbinint-holonomic-solver-b7d3cd`
Status: **complete** (adoption gate for M1 met)

M1 has two halves, both required by the goal procedure before any C++:

1. re-verify every boxed formula of `implementation_plan_ja.md` symbolically /
   numerically, independently of the plan's own (absent) prototype scripts;
2. implement radial-event enumeration standalone and prove completeness
   against a brute-force reference.

---

## 1. What was implemented

| File | Purpose |
|---|---|
| `checks/holonomic/symbolic_checks.py` | 18 symbolic/numeric identity checks, re-derived from the lens equation |
| `evidence/holonomic/symbolic_checks.{json,txt}` | recorded PASS log |
| `python/lcbinint/holonomic_ref/polynomial_family.py` | the single source of `P(t;R)`, `T(t)`, `B(t;R)`, `phi` |
| `python/lcbinint/holonomic_ref/radial_events.py` | exact `D14(v)` construction + full event enumeration |
| `tests/holonomic/test_symbolic_identities.py` | pytest wrapper over the symbolic checks |
| `tests/holonomic/test_radial_event_completeness.py` | brute-force completeness + stress battery |

`holonomic_ref` is a **pure-Python reference package** (correctness first, speed
never). It is import-isolated from the shipped `lcbinint` package and touches
no existing solver.

---

## 2. Mathematical basis — verification results

All 18 checks PASS (`evidence/holonomic/symbolic_checks.txt`), run time ~9 s.

| Identity (plan section) | How verified | Result |
|---|---|---|
| `phi = P / (rho^2 R^2 A B)` (§2) | lens eq. vs `P` at random complex `z`, 3 tuples | max rel err `4.98e-12` |
| `P` is quartic in `t`, real (§2) | symbolic degree/coeff | exact |
| `Disc_t P = R^4 · D14(R^2)`, `deg_v D14 = 14` (§3) | exact discriminant, 3 rational tuples; strip `R^4`; check even-in-`R` | R-powers `[4,6,…,32]`, `deg_v D14 = 14`, odd part `≡ 0` |
| `Res_t(P, P_t) = p4 · Disc_t P` (§3) | symbolic | exact |
| `Res(P,A) = 256 a^2 R^4 (m0−R^2)^2 |zeta|^2` (§3) | resultant with matching numeric `a`, 3 tuples | exact |
| `Res(P,B) = 256 a^2 R^4 m1^2 [L^2 + ys^2 vR^2 (vR−a^2)^2]` (§3) | resultant, 3 tuples | exact |
| `Disc A = −4`, `Disc B = −4(R^2−a^2)^2`, `Res(A,B) = 16 a^2 R^2` (§3) | fully symbolic | exact |
| LD reduction `P/(A Y) dt = H/Y dt + d_t(S Y / A)`, `S = −t/(4 a R)` (§5–6) | substitute, expand `(residual)·2Q` | **residual exactly 0** |
| `deg_t H ≤ 6` (§6) | symbolic degree | ✓ (= 6) |
| Gauss–Manin `−½ t^k Q_R = C_k Q + S_{k,t} Q − ½ S_k Q_t`, `deg C_k ≤ 6` (§7) | build `S_k = t^k Q_R (Q_t)^{-1} mod Q`, all tuples, all `R`, `k = 0..6` | exact, `deg C_k ≤ 6` |
| root-pair `E`, `O` closed forms (§8) | expand `P` around `(m,v)` | exact |
| `det ∂(E,O)/∂(m,v)|_{v=0} = −½ P''(m)^2` (§8) | impose `P'(m)=0` | exact on the double-root locus; raw excess `= (c3+4c4 m) P'(m)` |
| `R_max = (a + W + sqrt((a−W)^2 + 4))/2`, `W = |zeta|+rho` (§3) | 3000 configs × 3 radii × 36 angles | no image outside |

Two identities initially reported FALSE; both were bugs in the *check*, not the
plan (mismatched numeric/symbolic `a` in a resultant; the Jacobian identity
holds only modulo `P'(m)=0`). Recorded in `symbolic_checks.py` comments.

**Conclusion: the entire mathematical foundation of the design document is
sound.** No design change was required at M1.

---

## 3. Radial event enumeration

`radial_events(a, m0, xs, ys, rho) -> ([RadialEvent], R_max)`.

Events emitted:

* **`physical_real`** — positive real root `v` of `D14`, whose associated
  multiple root `t*` of `P(·; R=√v)` is real. Image-band birth/death.
* **`physical_complex`** — real root of `D14` with complex `t*`, or a complex
  root of `D14` (real part used). Representation-only; kept as a safe cell
  boundary.
* **`R_eq_a`**, **`R_eq_sqrt_m0`** — representation degeneracies.
* **`L_root`** — on the axis (`ys = 0`), positive roots of
  `L(v) = (a−xs) v (v−a^2) − a v + a^3 m0`.
* **`chart_p4`** — positive roots of `p4(R)` (leading `t`-coefficient of `P`):
  a boundary point crossing `theta = pi`.

### Key finding — float64 enumeration is unsafe

The monomial basis is ill-conditioned (plan §15 quotes `kappa ~ 1.9e9`).
`numpy`/`polyroots` companion-matrix root-finding on the degree-14 `D14`
**silently converts real double roots into spurious complex pairs and misses
band births** (`a=1.1, m0=1/1.4, xs=0.075, ys=0`: a real 4→0 topology change
at `R≈0.89862` was invisible to float64; exact 30-digit isolation recovers a
real double root `v=0.807519`).

Decision (autonomous, per "fix the design, record the reason"): the **reference**
enumerator builds `D14` and all auxiliary polynomials in **exact rational
arithmetic** (`sympy`, from the exact binary values of the inputs) and isolates
real roots exactly. The float64 path is retained only as
`radial_events.d14_coeffs_fast` for future benchmarking of the C++ port, which
will need a conditioned (scaled/shifted) basis — not tighter tolerances — to be
trustworthy. This is logged as an open risk for M7/M12.

### Completeness evidence

`tests/holonomic/test_radial_event_completeness.py` (76 cases, all pass):

* **constancy** — on every open sub-interval between consecutive enumerated
  events, the brute-force crossing count (sign changes of `phi(R, theta)` on a
  4096-point `theta` grid, straight from the lens equation) is constant;
* **no unexplained jump** — on a 2500-point `R` sweep every crossing-count
  change lies within `3·ΔR` of an enumerated event (5 configs + a 15-config
  caustic march across the central caustic, on- and off-axis);
* **physical events bite** — ≥ 60 % of isolated `physical_real` events flip the
  crossing count (the rest are external tangencies — legitimate boundaries);
* **well-formed** — 60 random configs (`q ∈ [1e-6, 1]`, `a ∈ [0.1, 10]`,
  `rho ∈ [1e-4, 0.32]`, planetary / resonant / wide): sorted radii strictly
  inside `(0, R_max)`, no same-kind collisions below `merge_tol`.

Plan §15 reference case `a=6/5, m0=2/3, zeta=1/5+i/7, rho=1/8` reproduces the
prototype's 8 physical radii
`{0.61086, 0.68433, 0.72464, 0.89133, 0.91110, 1.06417, 1.51770, 1.58413}`
plus `√(2/3)=0.81650` and `a=1.2`, and additionally surfaces 3
representation-only complex-`D14` events.

---

## 4. Benchmark results

None yet — M1 has no runtime-critical path. `radial_events` currently costs
~0.15–0.5 s/config (exact `sympy`), acceptable for a reference and for
generating oracles. The production speed target is an M7+ concern.

---

## 5. Open risks carried forward

1. **Conditioning (M7/M12).** The C++ port cannot use the monomial basis with
   float64. Need scaled/shifted/target-moment-aligned bases with monitored
   condition numbers; float `d14_coeffs_fast` must be validated against the
   exact path over the §16 audit grid before it is trusted.
2. **`physical_complex` classification.** Currently a soft boundary. If M2 cell
   construction proves it never matters, it can be dropped to cut cell count;
   if it does matter near cusps it must be kept — decide with data at M2.
3. **On-axis degeneracy (`ys = 0`).** Produces `D14` double roots and extra
   `L_root`/`chart_p4` events; handled, but M6 singular-cell logic must treat
   the simultaneous `±t*` births.
4. **Baseline for A/B.** `algebraic_boundary.cpp` (plan's assumed baseline,
   commit `54781b3`) does not exist in this repo; A/B is against
   `finite_source_magnifier.cpp` (see `correspondence.md`).

---

## 6. Next — M2

Topology classification per radial cell (0/2/4 circle crossings, ≤ 2 interior
arcs; empty vs full-circle by representative-angle sign), building the
`CellPlan` list the transport stage consumes. Reference-first, in
`holonomic_ref`.
