# Checkpoint M3 — period reduction reference

Date: 2026-09-07
Branch: `claude/holonomic-solver-m1` (M1–M3 share the feature branch;
the `…-b7d3cd` name in the docs is the worktree, not the checked-out branch)
Status: **complete** (adoption gate for M3 met — agreement at 3e-12, gate asks 1e-10)

M3 builds the algebraic period-reduction chain the transport stage (M4) runs
on every cell:

```
  Phi_arc(R) = ∫ (P/A) dt / sqrt(Q)          (the linear-LD half-moment)
             = Σ_{k=0}^{6} h_k I_k            (LD reduction, plan §5–6)
             = c^T Π ,  Π = W · I             (residue-free 6-form basis, plan §7)
```

and verifies every step — the coefficient formulas, the residue conditions,
the reduced period *values*, and (softly) the Gauss–Manin connection — against
the bare angular integral straight from the lens equation.

---

## 1. What was implemented

| File | Purpose |
|---|---|
| `python/lcbinint/holonomic_ref/period_reduction.py` | `H(t;R)` coefficients, residue coefficients `b1,b2,b3`, `psi_reduction_matrix` `W`, `residue_at_infinity`, `observed_covector_exact`; numeric periods: `arc_chart`, `half_period_eta` (the `I_k`), `half_period_obs_angular` (ground truth), `half_period_obs_reduced` (7D / 6D), `closed_period_eta` (contour period vector) |
| `python/lcbinint/holonomic_ref/connection.py` | exact Gauss–Manin: `_ck_sk_exact` builds `S_k` (deg ≤ 7) and `C_k` (deg ≤ 6) with a zero-residual check, `connection_matrix` (7×7 float64), `gm_polynomials` (float coeff arrays for a pointwise re-check), `s_polynomials`, `q_coeffs[_exact]` |
| `tests/holonomic/test_period_reduction.py` | 25 cases over 6 physical configs: `b`-formula = Laurent series, GM coefficient identity, residue conditions, 7D/6D period value = angular integral, soft closed-contour connection ODE |

Still a **pure-Python reference** — exact `sympy` for every coefficient, and
`scipy.integrate.quad` (QAWSE, spectral against the endpoint square-root
weights) for the period values. Correctness first, speed never. Touches no
existing solver.

### Public API additions (`holonomic_ref`)

* `arc_chart(a, m0, xs, ys, rho, arc) -> (params, (t_lo, t_hi), shift)` — the
  chart in which `arc` is a bounded `t`-interval.
* `half_period_eta(R, …, arc) -> [I_0 … I_6]` — the monomial periods.
* `half_period_obs_angular(R, …, arc)` — `(ρR/2) ∫_arc sqrt(phi) dθ`, ground truth.
* `half_period_obs_reduced(R, …, arc, basis="7D"|"6D")` — the reduced value.
* `closed_period_eta(R, …, arc) -> ([∮ η_0 … ∮ η_6], (t1, t2))` — the complex
  contour period vector used to spot-check the connection.
* `connection_matrix(R, …)` (7×7), `gm_polynomials(R, …)`, `s_polynomials(R, …)`.

---

## 2. Mathematical basis — verification results

### 2.1 LD reduction and `H` (plan §5–6)

`P/(A Y) dt = H/Y dt + d_t(S Y / A) dt` with `S = -t/(4 a R)` was proven exact
at M1 (`symbolic_checks.py`, residual identically 0). M3 uses the boxed closed
form

```
  H = ( 2 (R+a)^2 P  +  t ( B P_t + B_t P ) ) / (8 a R) ,     deg_t H ≤ 6
```

`h_coeffs_exact` returns `[h_0 … h_6]`; `deg_t H > 6` raises (fail closed).

### 2.2 Residue coefficients `b1, b2, b3` (plan §7)

Of `η_0 … η_6` only `η_3` carries a residue at `t = ∞`. The `b_j` that kill it
are the Laurent coefficients of `(Q/q_8/t^8)^{-1/2}` at `t = ∞`:

```
  b1 = -q7/(2 q8)
  b2 = -q6/(2 q8) + 3 q7^2/(8 q8^2)
  b3 = -q5/(2 q8) + 3 q7 q6/(4 q8^2) - 5 q7^3/(16 q8^3)
```

**`test_residue_b_formula_equals_series`** — the closed formula
(`residue_b_exact`) equals an independent `sympy.series` expansion
(`laurent_b`) to `sp.simplify == 0`, exact, every arc of every config.

### 2.3 Residue conditions (exact, all 6 configs)

**`test_residue_conditions_exact`**:

| Claim | Result |
|---|---|
| `h3' := h3 + b1 h4 + b2 h5 + b3 h6 == 0` (P/(A Y) dt is 2nd kind) | `sp.simplify(h3') == 0` exact |
| every `psi` row (`e0, e1, e2, e4-b1 e3, e5-b2 e3, e6-b3 e3`) residue-free at ∞ | `residue_at_infinity == 0` exact |
| `η_3` alone carries a non-zero residue | `residue_at_infinity([0,0,0,1,0,0,0]) ≠ 0` |

So `Phi_arc = c^T Π`, `c = (h0, h1, h2, h4, h5, h6)`, `Π = W · I` with `W` the
6×7 `psi_reduction_matrix`.

### 2.4 Gauss–Manin connection (plan §7)

`_ck_sk_exact` constructs, for one rational `R`:

```
  S_k = t^k Q_R (Q_t)^{-1} mod Q                 (deg ≤ 7)
  T_k = (S_k Q_t - t^k Q_R) / (2 Q)              (exact division; nonzero remainder raises)
  C_k = T_k - S_{k,t}                            (deg ≤ 6; deg > 6 raises)
```

and then **asserts the exact identity residual is 0**:

```
  -1/2 t^k Q_R  ==  C_k Q + S_{k,t} Q - 1/2 S_k Q_t      for k = 0..6
```

Any nonzero residual raises `ValueError` — the matrix is never returned unless
it is exactly right. `connection_matrix` is `(C_k)_j` as float64.

**`test_gauss_manin_coefficient_identity`** re-checks the *returned float*
polynomials (`gm_polynomials`) at five sample `t` to `1e-6 · scale`
(`scale = max|Q|,|Q_R|` coefficient) — an independent guard against a
marshalling bug. Passes everywhere the connection is non-degenerate.

---

## 3. Numeric period values — the two-chart reduction

### 3.1 The chart problem

`t = tan(θ/2)` blows up at `θ = π`, so an arc that contains or approaches
`θ = π` is an unbounded (or huge-`|t|`) interval and the deflated quadrature
loses all precision. **`arc_chart`** picks between:

* **chart 1** — `t = tan(θ/2)`, params `(a, m0, xs, ys, rho)`, singular at `θ = π`;
* **chart 2** — `u = -1/t = tan((θ-π)/2)`, params `(-a, m0, -xs, -ys, rho)`,
  singular at `θ = 0`.

**Chart 2 is exactly the original boundary family reflected through the
origin** (`a → -a`, `ζ → -ζ`). Verified numerically:
`boundary_quartic(R, -a, m0, -xs, -ys, rho)` evaluated at `u` equals
`u^4 · P(-1/u)` to machine precision, and `B` picks up
`(R+a)^2 + (R-a)^2 u^2 = B_coeffs(R, -a)`. So the *entire* pipeline
(`boundary_quartic`, `_H_expr`, `q_coeffs`, the connection) composes with the
reflected parameters unchanged — no separate chart-2 algebra.

The rule: use chart 2 when the arc midpoint is in the left half-plane
(`cos(θ_mid) < 0`), i.e. always put the arc near `t = 0`. If the arc still
straddles the chosen chart's singular angle (an arc longer than a half turn
spanning both `θ = 0` and `θ = π`), `arc_chart` raises — **fail closed**.

### 3.2 Deflated quadrature

`half_period_eta` deflates the boundary pair `(t_1, t_2)` out of `P` in closed
form (`_deflated_quadratic`, exact synthetic division) so that

```
  g(t) = Q(t) / ((t-t_1)(t_2-t)) = -(p4 t^2 + d1 t + d0) A(t) B(t)
```

is smooth and positive on `[t_1, t_2]`; with `t = mid + half·u`,

```
  I_k = ∫_{-1}^{1} (mid+half u)^k / sqrt(g)  · (1-u^2)^{-1/2} du
```

is a smooth integrand against the Chebyshev weight — QAWSE (`weight="alg"`,
`wvar=(-1/2,-1/2)`, `epsabs=epsrel=1e-13`) is spectral. `g(mid) ≤ 0` raises.

`half_period_obs_angular` integrates `sqrt(phi)` (which *vanishes* linearly at
each end) against `wvar=(1/2, 1/2)`.

### 3.3 Result — 7D and 6D vs the bare angular integral

**`test_reduced_period_matches_angular_integral`** — for every arc of every
`arcs` cell of all 6 configs, both `Σ h_k I_k` (7D) and `c^T(W·I)` (6D) vs
`(ρR/2) ∫_arc sqrt(phi) dθ`:

| config | (xs, ys, ρ, q, a) | worst rel. err (7D and 6D) |
|---|---|---|
| plan §15 | (1/5, 1/7, 1/8, 0.5, 1.2) | 4.9e-14 |
| resonant | (0.05, 0.02, 0.05, 0.3, 0.9) | 3.0e-12 |
| planetary wide | (1.4, 0.10, 0.03, 1e-3, 2.5) | 3.2e-14 |
| close | (0.4, -0.05, 0.09, 0.8, 0.55) | 5.5e-13 |
| on-axis | (0.30, 0.0, 0.04, 0.6, 1.35) | 2.8e-14 |
| big source on centre | (0.0, 0.0, 0.25, 0.5, 1.1) | 7.1e-14 |

**Grand worst: 2.99e-12** (resonant `a=0.9`, a narrow near-birth cell where
`Phi_arc → 0`). Covers 2-arc cells and `θ=π`-straddling arcs (close `a=0.55`
cells 5–6, on-axis `a=1.35` cell 3, `a=1.1` cells 5–6 all use chart 2). The
M3 gate ("rtol ~1e-10") is met with ~2 decades of margin.

### 3.4 Soft connection check

**`test_closed_period_satisfies_connection_ode`** — `closed_period_eta`
integrates `t^k dt / sqrt(Q)` over a **fixed complex ellipse** enclosing
exactly the two branch points (minor axis shrunk until exactly two `Q`-roots
are inside, else raise). This contour does not touch the branch points, so it
*is* smooth in `R`, and a 5-point central difference of `∮ η` matches
`connection_matrix @ ∮ η` to `rel < 1e-4` on the wide cells of the §15 config.

**Why the closed contour and not the segment periods `I_k`:** the ODE
`d/dR ∮ = C ∮` holds only for a *closed* cycle over which `S_k/Y` is
single-valued. `half_period_eta` uses endpoints `t_1, t_2` frozen at the base
`R`; at `R ± dh` those are no longer roots of `Q`, the deflated integrand grows
a spurious `1/sqrt` singularity, and the finite difference is meaningless
(observed rel. err ~1 to ~90). `closed_period_eta` was added precisely so the
connection has an end-to-end numeric witness. Verified separately:
`∮ η_k = 2 · I_k` exactly (`Π / I = 2.0 ± 1e-16`).

---

## 4. Documented degenerate locus — `xs = ys = 0`

With the source **exactly** on the lens axis and at the origin
(`xs = ys = 0`), `Q(t; R)` carries a **structural degree-2 repeated factor at
every `R`** (`gcd(Q, Q_t)` has degree 2 for all `R`, not just isolated radii).
`_ck_sk_exact` correctly raises `"Q has a repeated t-root at this R (branch
collision)"` — the connection is genuinely undefined on this locus, and the
module **fails closed** rather than returning a wrong matrix.

The `a=1.1, ρ=0.25` CASE sits exactly here. `test_gauss_manin_coefficient_identity`
asserts `seen == 0` for it (every sample degenerate, all fail closed); the
reduction, residue, and period-value checks need only `q_coeffs`, `h_coeffs`,
and quadrature — **not** the connection — and still pass there (7D/6D 7.1e-14).

Perturbing to `xs = 1e-9` removes the degeneracy entirely (`gcd` degree 0).
For M4/M5 this means: an on-axis centred source is a **singular patch** that
the transport stage must route around (seed a fresh basis on either side, or
use the reflection symmetry `t ↔ -t` directly), never transport through.

---

## 5. Open risks carried forward

1. **Connection FD degrades near events / in narrow cells.** The closed-contour
   FD check is 1e-10…1e-7 on wide cells but ~1e-2 on the `a=1.35` narrow
   near-event cells (branch points nearly collide → the isolating ellipse is
   forced thin → contour quadrature loses conditioning). The *exact* connection
   is still correct there (`_ck_sk_exact` residual is 0); only the numeric
   witness weakens. M4 transport must monitor the branch-point separation and
   the `Q_t`-inversion condition number, and fail closed
   (`CONNECTION_ILL_CONDITIONED`) below a threshold set from M4 data.
2. **Float dynamic range of `gm_polynomials`.** For the wide planetary config
   (`a=2.5`) the `Q` / `Q_R` coefficients span ~8 decades, so the pointwise
   float identity check only reaches `5e-8` (vs `1e-12` elsewhere). The exact
   path is unaffected; the C++ port (M7) will need a scaled `t`/`R` basis here,
   same conclusion as the M1 conditioning risk.
3. **`arc_chart` half-turn limit.** An arc longer than π spanning both `θ = 0`
   and `θ = π` has no regular `tan` chart and raises. Not seen in any test
   config (max arc measure ~2.4 rad). If M4/M5 stress configs produce one, it
   needs splitting at a regular angle before reduction — flagged, not built.
4. **`closed_period_eta` branch-point identification.** Uses the two `P`-roots
   nearest the arc endpoints (`np.roots` on the float quartic). Robust in all
   tested cases but float root-finding on the monomial quartic inherits the M1
   conditioning caveat; the function is a soft check only, so this is acceptable.

---

## 6. Verification summary

`tests/holonomic/test_period_reduction.py` — **25 passed** (~75 s).
Full holonomic suite: **179 passed** (M1 + M2 + M3), ~157 s
(`taskset -c 0-7`, load average ~11 on a shared 64-core host).

| Check | Assertion | Result |
|---|---|---|
| `b`-formula = series | `sp.simplify(b_formula[j] - b_series[j+1]) == 0` | exact, all configs |
| GM coefficient identity | float `gm_polynomials` satisfy `-½ t^k Q_R = C_k Q + S_k' Q - ½ S_k Q_t` at 5 sample `t` | `< 1e-6·scale` where non-degenerate; fail-closed on the `xs=ys=0` locus |
| residue conditions | `h3' == 0`; 6 `psi` rows residue-free at ∞; `η_3` residue ≠ 0 | exact, all configs |
| reduced period value | 7D `Σ h_k I_k` and 6D `c^T W I` vs `(ρR/2)∫sqrt(phi)dθ` | grand worst **2.99e-12** |
| connection ODE (soft) | 5-pt FD of `∮ η` vs `connection_matrix @ ∮ η`, wide §15 cells | `rel < 1e-4`, ≥ 2 cells checked |

**Conclusion: the period-reduction chain of the design document is sound and
numerically realised.** No design change was required at M3; the `xs=ys=0`
structural degeneracy is documented as a singular patch for M4/M5 and the
module fails closed on it.

---

## 7. Next — M4

Normal-cell period transport + flux (Python reference): integrate
`dΠ/dR = C(R) Π` across each cell from a seed period vector, assemble the
uniform flux `F_0` and the linear-LD flux `F_{1/2}`, and check both against a
direct 2-D quadrature of the magnification integral over several cells.
Record the transport tolerance and the connection condition number per cell.
Files `holonomic_ref/transport.py`, `seed.py`, `root_pair.py`.
