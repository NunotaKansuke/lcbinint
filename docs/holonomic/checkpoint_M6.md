# Checkpoint M6 — value / JVP consistency (the 5-component Jacobian)

Date: 2026-09-07
Branch: `claude/holonomic-solver-m1` (M1–M6 share the feature branch; the
`…-b7d3cd` in the milestones header is the worktree, not the branch)
Status: **complete** (M6 gate met — central-difference convergence,
chart-change invariance, gradient agreement with the independent M4 value
path; fail-closed status preserved)

M6 adds `∂μ/∂p` for `p = (xs, ys, ρ, q, a)` — plus `∂μ/∂u` — as a
**finite reconstruction that returns value and gradient from one numerical
object** (plan §11). No autodiff framework: the epoch flux is re-derived
as a fixed per-cell Gauss–Chebyshev radial pass that, at every node,
enumerates the arc endpoints from the boundary-quartic roots and carries a
closed-form `φ`-gradient through the same quadrature weights.

```
  epoch_jacobian(params, u) -> EpochJacobian(mu, grad_mu[5], dmu_du, F0, F_half, status, notes)
  solve_epoch(params, u, with_jacobian=True) -> EpochResult(..., jacobian={grad_mu, dmu_du, param_order, status, notes})

  value  = ((1-u) F0 + u F_1/2) / (π ρ² (1 - u/3))
  grad   = ((1-u) dF0 + u dF_1/2) / D  - (2μ/ρ) e_ρ        [D = π ρ² (1-u/3)]
  dmu_du = π ρ² (F_1/2 - 2 F0 / 3) / D²                    (numerator const in u)
```

---

## 1. What was implemented

| File | Purpose |
|---|---|
| `python/lcbinint/holonomic_ref/jacobian.py` | **new.** `phi_grad` (closed-form `φ`, `∂φ/∂θ`, `∂φ/∂(X,Y,ρ,m0,a)` in the primary frame); `polish_endpoint` (Newton on `φ(R,θ*)=0`); `arc_intervals` (arc endpoints as the real roots of `boundary_quartic(R)`); `radius_terms` (per-`R` integrands `f0, f_1/2` and their 5 derivatives); `endpoint_dtheta_theta_chart` / `endpoint_dtheta_t_chart` (IFT-in-θ and IFT-in-t forms of `∂θ*/∂P`); `flux_jacobian` (per-cell Gauss–Chebyshev radial assembly + primary→user chain rule); `epoch_jacobian` (μ, `grad_mu`, `dmu_du`, status) |
| `python/lcbinint/holonomic_ref/solver.py` | `solve_epoch(..., with_jacobian=False)`; when set, attaches `.jacobian` and **adopts the jet reconstruction's `F0`/`F_1/2`/`μ`** so the returned value and gradient are mutually consistent; folds `GRADIENT_UNRELIABLE` into the epoch status |
| `python/lcbinint/holonomic_ref/polynomial_family.py` | `boundary_quartic_dp(R,a,m0,xs,ys,rho)` — `∂[p0..p4]/∂(xs,ys,ρ,m0,a)` in closed form (used by the IFT-in-t cross-check) |
| `python/lcbinint/holonomic_ref/__init__.py` | exports the M6 names |
| `tests/holonomic/test_value_jvp_consistency.py` | **new**, 17 tests — `phi_grad` vs FD; endpoint chart invariance; `grad_mu` central-difference convergence in `n_r`; `grad_mu` vs `epoch_flux` central diff; loose A/B vs `binary_ray_shooting` FD; `dmu_du` sign + magnitude; on-axis `ys` symmetry; `solve_epoch(with_jacobian=True)` wiring + value/JVP consistency; fail-closed on the singular patch, the degenerate-quartic node, and the full-circle radius |

Still a **pure-Python reference**. Touches no existing solver; the default
`solve_epoch` path is byte-for-byte unchanged (`jacobian=None`).

---

## 2. Math basis (plan §11)

### 2.1 Flux derivatives

With arcs `[θ_enter, θ_leave]` on the circle `|z| = R` (sign of `φ` = sign
of the boundary quartic `P`):

```
  F0    = ∫₀^Rmax R Σ_arcs (θ_leave − θ_enter) dR
  F_1/2 = ∫₀^Rmax R Σ_arcs ∫_arc √φ dθ dR
```

* `∂F0/∂P_j = ∫ R Σ_arcs (∂θ_leave/∂P_j − ∂θ_enter/∂P_j) dR`.
* `∂F_1/2/∂P_j = ∫ R Σ_arcs ∫_arc (∂φ/∂P_j)/(2√φ) dθ dR`. The endpoint
  boundary terms `√φ · ∂θ*/∂P_j` vanish because `φ(θ*) = 0`.

### 2.2 Closed-form `φ` gradient

`g = f − ζ`, `z = R e^{iθ}`, `f_z̄ = m0/z̄² + m1/(z̄−a)²`, `r² = ρ²`:

```
  ∂φ/∂θ  = −(2/r²) Re( ḡ (i z − i z̄ f_z̄) )
  ∂φ/∂X  =  (2/r²) Re g          ∂φ/∂Y = (2/r²) Im g
  ∂φ/∂ρ  =  2 |g|² / ρ³
  ∂φ/∂m0 = −(2/r²) Re( ḡ (−1/z̄ + 1/(z̄−a)) )
  ∂φ/∂a  = −(2/r²) Re( ḡ (−m1/(z̄−a)²) )
```

Verified to `< 1e-6` (relative) vs central differences at every
`(R, θ)` probe.

### 2.3 Endpoint derivatives — chart-free (IFT-in-θ)

The arc endpoint `θ*` solves `φ(R, θ*; P) = 0`, so

```
  ∂θ*/∂P_j = − (∂φ/∂P_j) / (∂φ/∂θ)   evaluated at θ*.
```

This is the primary form. The `t = tan(θ/2)` form
(`∂t*/∂P_j = −(∂P/∂P_j)/(∂P/∂t)`, then `∂θ*/∂P_j = 2/(1+t*²) ∂t*/∂P_j`)
is algebraically identical but carries a removable `2/(1+t²)` pole at
`θ* = π`; it is kept only as a cross-check. On every arc endpoint away
from `θ = π` the two agree to **2.8e-14** (relative).

### 2.4 μ and `∂μ/∂u`

`D = π ρ² (1 − u/3)`. `∂μ/∂P_j = ((1−u) ∂F0/∂P_j + u ∂F_1/2/∂P_j)/D` for
`P_j ≠ ρ`; the `ρ` component additionally gets `−2μ/ρ` from `D`'s explicit
`ρ²`. `∂μ/∂u = π ρ² (F_1/2 − 2 F0/3) / D²` — the numerator is **constant
in `u`**, so μ is monotone in `u` with a geometry-dependent sign (design
note 11).

### 2.5 Primary → user chart

Internally `P = (X, Y, ρ, m0, a)` in the primary frame; the returned
gradient is in `p = (xs, ys, ρ, q, a)`. With `m0 = 1/(1+q)`,
`dm0/dq = −1/(1+q)²`, and (barycentric only) `X = xs + m1 a`:

```
  ∂/∂xs = ∂/∂X                       ∂/∂ys = ∂/∂Y
  ∂/∂q  = ∂/∂m0 · (−1/(1+q)²)  [+ ∂/∂X · a/(1+q)²      if barycentric]
  ∂/∂a  = ∂/∂a_pf              [+ ∂/∂X · q/(1+q)        if barycentric]
```

---

## 3. Verification (real numbers, `taskset -c 0-7`, load ≈ 10–15)

### 3.1 `∂F0/∂P` — the arc-enumeration fix

`topology.arcs_at`'s grid sign-scan (step `2π/n_grid`) misses thin newborn
arcs (width ~3e-4 rad) that open over a finite sub-interval of a
`physical_real` (tangency) cell. The **value** survives
(`∫ √(R−R_e) dR ~ w^{3/2}`) but the **derivative** does not
(`∫ R_e' / √(R−R_e) dR ~ √w`), giving `∂F0/∂P` errors of 8–11 %.

Fix: `arc_intervals` takes the arc endpoints as the real roots of
`boundary_quartic(R)` (`np.roots`), resolving an arc as soon as its
half-gap in `t` exceeds ~1e-9. On the benign config `∂F0/∂X` moved from
`−3.28e−01` (8.5 % low) to `−3.59e−01`; all five components now agree with
a finite difference of the independent `image_plane_flux` (with
`event_radii`):

| component | jet | `image_plane_flux` FD | rel |
|---|---|---|---|
| `∂F0/∂X`   | −3.593e−01 | −3.591e−01 | 5.5e−4 |
| `∂F0/∂Y`   | −1.039e−01 | −1.040e−01 | 8.8e−4 |
| `∂F0/∂ρ`   | +1.610e+00 | +1.610e+00 | 3.6e−5 |
| `∂F0/∂m0`  | −3.077e−01 | −3.075e−01 | 5.3e−4 |
| `∂F0/∂a`   | +3.883e−01 | +3.882e−01 | 2.7e−4 |

### 3.2 `grad_mu` vs the independent M4 value path (`epoch_flux`)

Central difference of `epoch_flux(...).mu_linear_ld(u)` (adaptive QUADPACK,
a wholly different construction from the fixed Gauss–Chebyshev jet):

| case | u | worst component | worst rel |
|---|---|---|---|
| plan15   | 0.0 | `∂μ/∂a`  | 2.8e−4 |
| plan15   | 0.6 | `∂μ/∂a`  | 1.7e−4 |
| resonant | 0.0 | `∂μ/∂ys` | 2.5e−4 |
| resonant | 0.6 | `∂μ/∂a`  | 9.7e−4 |
| benign   | 0.0 | `∂μ/∂a`  | 2.7e−2 † |
| benign   | 0.6 | `∂μ/∂ρ`  | 1.3e−2 ‡ |

† `∂μ/∂a ≈ −0.025` on the benign config — a near-zero derivative; the
absolute error is ~7e−4 and the `quad` oracle's own value accuracy
(~1e−4) limits the finite difference there.
‡ `∂μ/∂ρ = ((1−u)∂F0/∂ρ + u∂F_1/2/∂ρ)/D − 2μ/ρ` is a difference of two
terms each ~150 that cancel to ~1 for small ρ; the absolute error is
~1e−2 and converges `O(1/n_r²)`.

### 3.3 `grad_mu` vs `image_plane_flux(+event_radii)` Richardson CD (fully independent)

| case | u | `∂μ/∂xs` | `∂μ/∂ys` | `∂μ/∂ρ` | `∂μ/∂q` | `∂μ/∂a` |
|---|---|---|---|---|---|---|
| plan15   | 0.0 | 2.0e−5 | 1.5e−4 | 1.6e−4 | 1.6e−5 | 1.4e−5 |
| resonant | 0.0 | 1.7e−4 | 5.6e−3 | 1.4e−3 | 9.0e−5 | 1.8e−5 |
| benign   | 0.0 | 1.2e−3 | 1.6e−3 | 2.6e−1 * | 8.0e−4 | 9.4e−4 |

\* the nested-quadrature oracle cannot resolve the benign `∂μ/∂ρ ≈ 0.17`
near-cancellation for small ρ — this is an oracle limitation, not a jet
error (the `epoch_flux` CD in §3.2 gives 4.9e−3 there, and both converge to
the same limit under `n_r` refinement).

### 3.4 Central-difference convergence in `n_r`

`max_j |grad_mu_j − CD(own μ)|`, resonant config, `u = 0`:

| `n_r` | 48 | 96 | 192 |
|---|---|---|---|
| max abs err | 6.6e−3 | 1.7e−3 | 4.5e−4 |

Clean `O(1/n_r²)` — the value and every derivative share the rule, so they
share the rate. `epoch_jacobian` default `n_r = 64`.

### 3.5 Chart invariance, `dmu_du`, symmetry

* endpoint `∂θ*/∂P` — IFT-in-θ vs IFT-in-t: **2.8e−14** (rel) on every
  arc endpoint of plan15 / resonant / benign away from `θ = π`.
* `dmu_du` sign = `sign(F_1/2 − 2 F0/3)` on both non-trivial cases
  (plan15 `+1.68e−1`, resonant `+2.45e−1`); magnitude vs a central
  difference of μ in `u` to `< 1e−6`.
* on-axis `ys = 0`, `xs = 0.30` (μ even in `ys`): `∂μ/∂ys = 5e−12` while
  the other four components are `O(2–23)`.

### 3.6 `solve_epoch(with_jacobian=True)`

`.jacobian` is `{grad_mu, dmu_du, param_order=('xs','ys','rho','q','a'),
status, notes}`. The returned `res.mu` equals `epoch_jacobian(...).mu`
exactly (one reconstruction), and that reconstruction agrees with the
default adaptive-quad `solve_epoch` value to `2e−3`. Default path
(`with_jacobian=False`) returns `jacobian=None` and is unchanged.

---

## 4. Fail-closed behaviour (plan: status, not silent approximation)

`epoch_jacobian` returns `GRADIENT_UNRELIABLE` (never a bare `OK` with a
guessed number) when:

1. **on-axis-origin singular patch** (`near_origin_source`, `|ζ| < 1e-3`):
   the Gauss–Manin connection is genuinely undefined here
   (`checkpoint_M3.md` §4), so no consistent period derivative exists.
2. **degenerate quartic node** — at a `chart_p4` event the boundary
   quartic loses its leading coefficient; `arc_intervals` returns
   `"degenerate"`, `radius_terms` falls back to the `arcs_at` grid with
   `reliable = False`.
3. **full `φ > 0` circle** inside `R_max` — the endpoint rule degrades to
   a periodic trapezoid with no error control; `_full_circle_terms`
   returns `reliable = False` and `df0 = 0`.
4. **near-tangency node** — `|∂φ/∂θ| < tan_rel · ρ/R` at an arc endpoint
   (`tan_rel = 1e-4`); the endpoint derivative `−(∂φ/∂P)/(∂φ/∂θ)` is then
   ill-conditioned.
5. upstream `classify_cells` status `≠ OK` (propagated).

`solve_epoch` folds this into the epoch status via `_worst_status`.

---

## 5. Open risks / notes for M7

1. **`∂μ/∂ρ` catastrophic cancellation.** `((1−u)dF0 + u dF_1/2)/D` and
   `2μ/ρ` are each `O(1/ρ)` and cancel to `O(1)` for small ρ. Absolute
   accuracy is ~1e−2 at `n_r = 64`, ~3e−3 at `n_r = 128`. The C++ jet
   should assemble `∂μ/∂ρ` from `∂(F/ρ²)/∂ρ` directly rather than
   subtracting two large terms.
2. **jax A/B path unavailable.** `binary_ray_shooting(..., jax=True)`
   raises (`lcbinint` built without the polar epoch FFI). The A/B
   gradient check falls back to a finite difference of native
   `binary_ray_shooting` (rel ~1e-2, coarse). M7's speed/accuracy gate
   will need the FFI build.
3. **`n_r` is fixed, not adaptive.** `epoch_jacobian(n_r=64)` is a
   compromise; a per-cell node count keyed to the cell's condition number
   (already recorded in M4) would tighten `∂μ/∂ρ` without a global cost.
4. **near-tangency trigger is theoretical.** The quartic-root arc
   enumeration resolves newborn arcs so cleanly that the `tan_rel` check
   in §4.4 did not fire on any tested config even ~1e-7 past a
   `physical_real` event; the reliably-exercised `GRADIENT_UNRELIABLE`
   paths are the singular patch and the degenerate/full-circle nodes.
5. **endpoint derivatives are exact; F_1/2 quadrature is not.** The
   `∫ (∂φ/∂P)/(2√φ) dθ` integrand still has the `1/√φ` endpoint
   singularity; the per-arc Chebyshev–Gauss (1st kind) rule handles it but
   at fixed order. Same `O(1/n²)` story as the value.

---

## 6. Design decisions recorded this milestone

(12) endpoint derivatives use the **chart-free IFT-in-θ** form
`∂θ*/∂P = −(∂φ/∂P)/(∂φ/∂θ)`; the `t`-chart form is algebraically
identical but has a removable pole at `θ* = π` and is kept only as a
cross-check.

(13) arc endpoints at each radial node are the **real roots of
`boundary_quartic(R)`**, not `topology.arcs_at`'s angular grid scan — the
grid misses thin newborn arcs near tangency cell edges, giving ~10 %
error in `∂F0/∂P` (negligible, ~1e-4, in the value).

(14) the jax A/B gradient path is unavailable (no polar epoch FFI) →
native `binary_ray_shooting` finite differences are the A/B relation,
loose (~1e-2).

(15) `∂μ/∂u = π ρ² (F_1/2 − 2 F0/3)/D²` has a **`u`-independent
numerator** — one flux pair gives `∂μ/∂u` at every `u`.

(16) `solve_epoch(with_jacobian=True)` **adopts the jet reconstruction's
`F0`/`F_1/2`/`μ`** for value/JVP consistency (plan §11); this reconstruction
agrees with the adaptive-quad `epoch_flux` value to ~1e-4 (2e-3 on the
worst small-ρ config).

---

Next: **M7** — C++ production implementation (`ForwardJet<5>` replay,
versioned FFI) with the end-to-end median ≥ 2× / p95 ≤ +25 % speed gate.
Blocked on the polar epoch FFI build (design note 2 / risk 2).
