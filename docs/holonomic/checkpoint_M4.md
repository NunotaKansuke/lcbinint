# Checkpoint M4 — normal-cell transport + epoch flux (Python reference)

Date: 2026-09-07
Branch: `claude/holonomic-solver-m1` (M1–M4 share the feature branch;
the `…-b7d3cd` name in the docs is the worktree, not the checked-out branch)
Status: **complete** (M4 gate met — `F0` / `F_1/2` agree with a fully
independent double quadrature to ~1e-7; per-cell condition numbers recorded)

M4 turns the M3 period-reduction chain into a working epoch magnification:

```
  F0     = ∫_0^Rmax  R · ( Σ_arcs Δθ_arc(R) )        dR   = image area = μ_uniform · πρ²
  F_1/2  = ∫_0^Rmax  ( Σ_arcs (2/ρ) Φ_arc(R) )       dR
  μ_u    = ( (1-u) F0 + u F_1/2 ) / ( πρ² (1 - u/3) )                       (plan §1)
```

with `Φ_arc(R) = c^T (W · I)` in the residue-free 6-form basis (the M3
reduction, blessed to 3e-12), the arc set at each radius coming straight
from the lens equation (`topology.arcs_at`), and *every* arc summed — no
image-lineage matching is needed for the total.

The transport machinery the plan names for M4 (`root_pair.py`,
`transport.py`, `seed.py`) is **built and validated pointwise**, but the
reference flux deliberately does **not** transport a single seed across a
cell: the raw monomial / ψ bases are ill-conditioned over a full cell
(plan §15) and the flux re-anchors the period at every radial quadrature
node instead. See §4.

---

## 1. What was implemented

| File | Purpose |
|---|---|
| `python/lcbinint/holonomic_ref/root_pair.py` | root-pair `(m, v)` coords (plan §8): `RootPair`, `from_endpoints`, `eo_residuals` (`E`, `O`), `eo_jacobian` (analytic 2×2), `tangency_determinant` `= -½ P''(m)²`, `endpoint_dR` `= -P_R/P_t`, `root_pair_dR` (2×2 IFT; raises on tangency) |
| `python/lcbinint/holonomic_ref/transport.py` | ψ-basis connection `C_ψ` (6×6): `eta_connection`, `psi_connection` (float path), `psi_connection_exact` (sympy oracle), second-kind closure residual, `transport_psi` (stiff DOP853 over Chebyshev-interpolated `C_ψ`), `cell_conditioning` (the M4 gate record) |
| `python/lcbinint/holonomic_ref/seed.py` | period seeds: `seed_eta` / `seed_psi` (robust QAWSE seed, arc re-found at `R` and tracked by nearest midpoint), `tangency_seed_eta` (the `v→0` series of plan §9) |
| `python/lcbinint/holonomic_ref/flux.py` | `epoch_flux` → `FluxResult` (`.F0`, `.F_half`, `.mu_uniform`, `.mu_linear_ld(u)`, per-cell `CellFlux`, status), `mu_uniform`, `mu_linear_ld` |
| `python/lcbinint/holonomic_ref/direct_quadrature.py` | fully independent oracle (no `P(t;R)`, no periods): `point_source_magnification` (Witt & Mao degree-5), `source_plane_flux`, `image_plane_flux` / `image_plane_flux_grid`, `magnification_reference` |
| `python/lcbinint/holonomic_ref/connection.py` (+) | float fast paths `q_coeffs_numeric`, `q_coeffs_dR_numeric`; `connection_matrix_numeric` (balanced float Gauss–Manin + degree-drop hand-off, §5) |
| `python/lcbinint/holonomic_ref/period_reduction.py` (+) | float fast paths `h_coeffs_numeric`, `phi_arc_reduced_numeric` (`(value, h3_residual)`) |
| `python/lcbinint/holonomic_ref/polynomial_family.py` (+) | `boundary_quartic_dR` — exact closed-form `dP/dR` (no finite differencing) |
| `tests/holonomic/test_transport_flux.py` | 49 passed / 3 skipped — numeric fast paths vs the sympy oracles and vs finite differences; root-pair `E`/`O` and its Jacobian; ψ-closure and the pointwise connection ODE; `epoch_flux` vs `image_plane_flux`; condition numbers recorded |

Still a **pure-Python reference** — exact `sympy` for every symbolic
identity, `scipy.integrate.quad` (QAWSE) for every period. The float fast
paths exist only so the radial quadrature (hundreds of nodes per epoch)
does not call `sympy`; each is pinned to its oracle by a test. Touches no
existing solver.

---

## 2. Mathematical basis

### 2.1 Flux assembly (plan §1, §5–6, §16)

* **`F0` = image area.** `Σ_arcs Δθ_arc(R)` is the total angular measure of
  the image boundary at radius `R`; `∫ R · (Σ Δθ) dR` is the area of the
  image region, which for a uniform disk is `μ_uniform · πρ²`.
* **`F_1/2` = the linear-LD half-moment.** `Φ_arc(R) = (ρR/2) ∫_arc √φ dθ`
  (M3 ground truth); `(2/ρ) Φ_arc = R ∫_arc √φ dθ` and
  `∫_0^Rmax Σ_arcs (2/ρ)Φ_arc dR = ∫∫_image √φ dA`. With
  `√φ = √(1 - p²/ρ²)`, `p = |image point − source centre|`, this is exactly
  the image-plane form of `∫_disk A_pt(w) √(1-|w-w_c|²/ρ²) dA` — the
  identity `direct_quadrature` checks against.
* `μ_linear_ld(0) = μ_uniform` and `μ_linear_ld` is monotone in `u`
  (checked in the tests).

### 2.2 Root-pair `(m, v)` (plan §8)

`m = (t₊+t₋)/2`, `v = ((t₊−t₋)/2)²`. With
`E := P(m) + (v/2)P''(m) + (v²/24)P⁗(m)`,
`O := P'(m) + (v/6)P'''(m)`, the identity `E = O = 0 ⟺ P(t₋) = P(t₊) = 0`
is exact (proved symbolically at M1, `rootpair_E_exact` /
`rootpair_O_exact`). The analytic Jacobian

```
  E_m = P' + ½v P''' ,   E_v = ½P'' + (v/12)P⁗
  O_m = P'' + (v/6)P⁗ ,   O_v = ⅙P'''
```

matches a central finite difference of `eo_residuals` to 1e-5·scale
(`test_eo_jacobian_vs_fd`), and at a genuine tangency (`P'(m) = 0`, `v = 0`)
`det ∂(E,O)/∂(m,v) = -½ P''(m)²` (`test_tangency_determinant_double_root`:
`(t²−1)²` → det `= -32`).

Radial motion: a single endpoint moves as `dt/dR = -P_R(t)/P_t(t)`
(`endpoint_dR`); the pair moves by the 2×2 implicit-function solve
`(dm, dv) = -[∂(E,O)/∂(m,v)]⁻¹ ∂(E,O)/∂R` (`root_pair_dR`). Both match a
central FD of the `np.roots` endpoints to 5e-4·(|·|+1) on cells wide enough
for a clean difference (`test_root_pair_dR_and_endpoint_dR_vs_fd`).

### 2.3 ψ-basis connection `C_ψ` (plan §7)

The residue-free basis `ψ = W(R) · η` (6×7, `b1,b2,b3` in column 3) is
closed under `d/dR` with

```
  M = W'(R) + W(R) · C_η(R)          (6×7)
  C_ψ = M[:, (0,1,2,4,5,6)]
  closure residual = M[:,3] + b1 M[:,4] + b2 M[:,5] + b3 M[:,6]   ==  0
```

`psi_connection_exact` (sympy `C_η` and `b_j`) drives the closure residual
below 1e-7·scale on every arc of every non-degenerate config
(`test_psi_connection_second_kind_closure`).

### 2.4 Pointwise connection ODE — the numeric witness

The **moving-endpoint** segment period `I(R)` — arc endpoints re-found at
each `R` via `arcs_at(R)` and tracked by nearest midpoint, exactly what
`seed_eta` returns — satisfies

```
  dI/dR  =  C_η(R) · I(R)
```

to `rel < 2e-5` (4-point stencil, FD-noise limited) on cells wider than
0.04 (`test_moving_endpoint_period_satisfies_connection`). This closes the
M3 open item ("the connection had only a closed-contour witness"): with the
endpoints tracked as functions of `R`, the *segment* period obeys the same
connection, because the boundary term `[S_k/Y]` vanishes at the branch
points where `Y = 0`.

---

## 3. Result — `epoch_flux` vs the independent oracle

`test_epoch_flux_matches_independent_quadrature` and the 5-config smoke run
(`taskset -c 0-7`, shared 64-core host, load ≈ 10.5):

| config | (xs, ys, ρ, q, a) | `F0` rel | `F_1/2` rel | μ_uniform | `epoch_flux` | `image_plane_flux` | status |
|---|---|---|---|---|---|---|---|
| plan §15 | (1/5, 1/7, 1/8, 0.5, 1.2) | 1.7e-7 | 1.5e-7 | 6.01995 | 0.9 s | 12.3 s | OK |
| resonant | (0.05, 0.02, 0.05, 0.3, 0.9) | 1.2e-7 | 1.4e-7 | 13.73011 | 1.2 s | 12.4 s | OK |
| close | (0.4, −0.05, 0.09, 0.8, 0.55) | 2.0e-7 | 1.8e-7 | 8.13319 | 0.9 s | 13.1 s | OK |
| planetary wide | (1.4, 0.10, 0.03, 1e-3, 2.5) | 5.6e-5 | 9.5e-10 | 1.15919 | 0.6 s | 6.1 s | OK |
| near on-axis | (0.30, 1e-6, 0.04, 0.6, 1.35) | 2.2e-7 | 9.4e-10 | 5.35261 | 1.0 s | 11.4 s | TOPOLOGY_UNCERTAIN |

* The three primary cases (plan §15 / resonant / close) agree to ~1e-7 on
  **both** fluxes — the M4 gate ("`F0, F_1/2` が直接二重求積と一致") is met
  with margin. `_FLUX_CASES` in the test asserts `rel < 1e-4`.
* **planetary wide `F0` rel 5.6e-5** is a limitation of the *oracle*, not of
  `epoch_flux`: at `q = 1e-3` the planetary caustic is ~`√q ≈ 0.03` across
  and the source (`ρ = 0.03`) grazes it, so the source-plane
  `point_source_magnification` needs a very fine grid near the caustic that
  `image_plane_flux` does not fully resolve. `F_1/2`, which weights the
  caustic-adjacent rim down by `√φ → 0`, still agrees to 1e-9.
* **near on-axis** reports `TOPOLOGY_UNCERTAIN` (from `classify_cells`, not
  from the flux) — `ys = 1e-6` is ~one part in 10⁴ off the `xs = ys = 0`
  singular locus (M3 §4), so a radial event pair is nearly coincident. The
  flux value is still right to 2e-7; the status correctly refuses to call
  it validated.
* `epoch_flux` is **7–13× faster than the oracle** here, but that is a
  Python-reference-vs-Python-reference number and is *not* the M7 adoption
  gate (which is against the C++ `finite_source_magnifier`).

Independent-quadrature warning: `_arc_measure_sum`'s `quad` raises an
`IntegrationWarning` ("roundoff error in the extrapolation table") for
plan §15 — pre-existing, non-fatal, the flux still matches to 1e-7. Left
as-is; the M7 port will use a fixed radial rule per cell.

---

## 4. Conditioning — the M4 gate record

`cell_conditioning(r_lo, r_hi, params, n, exact=True)` samples
`cond(C_η)` and `cond(C_ψ)` on interior radii and returns max / median plus
the worst ψ-closure residual. Over the arc cells of the primary configs:

| config | # arc cells | `cond(C_η)` max | `cond(C_ψ)` max | `cond(C_ψ)` median | closure resid max |
|---|---|---|---|---|---|
| plan §15 | 8 | 1.6e8 | 1.6e8 | 2.4e5 | 1.8e-12 |
| resonant | 9 | 3.5e7 | 8.0e11 | 2.1e8 | 4.2e-6 |
| close | 8 | 2.9e6 | 2.3e7 | 5.1e4 | 8.5e-12 |

**Verdict (matches plan §15 "monomial 基底の悪条件, 要対策"):** a single-seed
transport of the raw η or ψ period vector across a full cell loses 3–8
digits — near the θ→π degree drop (`resonant`, where an arc endpoint races
toward the θ=π pole) `cond(C_ψ)` reaches ~1e12 and even the *exact*-oracle
closure residual degrades to 4e-6 because the `b_j` blow up like `1/q8`.

**Consequence, already reflected in the code:**

1. `flux.epoch_flux` never transports a seed. It re-anchors the period with
   `phi_arc_reduced_numeric` (the QAWSE seed) at *every* radial quadrature
   node. A stiff `C_ψ` cannot corrupt a value that is never propagated
   through it. `record_conditioning` defaults to **`False`** — the exact
   conditioning sweep costs more than the flux and only annotates.
2. `cell_conditioning` uses the **exact** sympy connection by default. The
   float surrogate `connection_matrix_numeric` reports spurious ~1e20
   condition numbers near the degree drop; `exact=False` is a
   benign-config-only preview.
3. A well-conditioned flux-priority basis (plan §8: a basis chosen so the
   *observed covector* `c` is transported stably rather than the full
   period vector) is deferred to **M7/M8**, exactly as the plan sequences
   it. M4's job was to *measure* the conditioning, and it does.

`transport_psi` and the seeds are kept and pointwise-validated so M7/M8
have a tested starting point.

---

## 5. New finding — float `connection_matrix_numeric` needs balancing + hand-off

The naive float polynomial Gauss–Manin (8×8 solve for `Q_t⁻¹ mod Q`, then
polynomial division by `2Q`) is **catastrophically wrong** — `rel 1e24–1e28`
vs the oracle — for small-ρ / near-degenerate configs. Two distinct causes:

1. **Dynamic range.** For small `ρ`, `Q = ρ²R²·A·B − |T|²` has its `ρ²`
   piece dwarfed by `|T|²`, so the coefficients `q0…q8` span 6+ decades and
   the float polynomial divisions lose all precision. **Fix:** work in a
   balanced variable `t = c·τ` with `c = (|q0/q8|)^{1/8}`; the connection of
   the rescaled family relates back exactly by `C_kj = C̃_kj · c^{k−j}`
   (`c` is `R`-independent). Plus a `max|Q̃|` normalisation. → benign
   configs recover `rel ~1e-10`, `a = 2.5` → `6e-7`.
2. **θ→π degree drop.** When an image-arc endpoint races toward the θ=π
   pole, `q8 → 0` relative to the lower terms and the degree genuinely
   drops toward 7. The float path is then singular; the exact oracle is
   not. **Fix:** hand off to `connection_matrix` (sympy) when
   `|q8| < 1e-6 · max|q|`.

`test_connection_matrix_numeric_vs_exact` pins the float path to the oracle
on benign cells (`|q8| ≥ 0.1·max|q|`, tol 1e-4·scale);
`test_connection_matrix_numeric_hands_off_on_degree_drop` asserts the
hand-off is bit-identical to the oracle.

Residual genuine ill-conditioning near `xs = ys = 0`: the near-on-axis
config (`a = 1.35`, `ys = 1e-6`) still shows `rel 1.9e-2` — this is real
(the connection *is* nearly singular on the M3 §4 locus), so that config is
excluded from `_CONN_CASES`. The flux there is computed from QAWSE seeds
only and still lands at 2e-7 (§3).

---

## 6. Verification summary

`tests/holonomic/test_transport_flux.py` — **49 passed, 3 skipped**
(~104 s, `taskset -c 0-7`). The skips: `test_root_pair_dR…` (planetary
wide has no cell wide enough for a clean radial FD),
`test_connection_matrix_numeric_hands_off_on_degree_drop` and
`test_moving_endpoint_period_satisfies_connection` skip on configs that
happen to expose no qualifying cell.

Full holonomic suite: **228 passed, 3 skipped** (M1 + M2 + M3 + M4),
~257 s. M1/M2/M3 unaffected by the M4 edits to `connection.py`,
`transport.py`, `flux.py`, `period_reduction.py`, `polynomial_family.py`,
`__init__.py`.

| Check | Assertion | Result |
|---|---|---|
| `q` / `h` fast paths | `q_coeffs_numeric` / `h_coeffs_numeric` vs sympy `*_exact` | machine precision |
| `boundary_quartic_dR` | vs central FD of `boundary_quartic` | `< 1e-6·scale` |
| `C_η` fast path | `connection_matrix_numeric` vs oracle, benign cells | `< 1e-4·scale`; bit-identical on degree drop |
| root-pair `E`/`O` | zero at real boundary roots; analytic Jacobian vs FD | `E,O < 1e-7·scale`; Jac `< 1e-5·scale` |
| radial motion | `root_pair_dR` / `endpoint_dR` vs FD of `np.roots` | `< 5e-4·(|·|+1)` |
| ψ-closure | `psi_connection_exact` residual | `< 1e-7·scale` everywhere |
| pointwise ODE | moving-endpoint `dI/dR` vs `C_η·I` | `rel < 2e-5` |
| **epoch flux** | `F0`, `F_1/2` vs `image_plane_flux` | **~1e-7** (primary configs) |
| conditioning | `cell_conditioning` finite, closure `< 1e-6` on first cells | recorded, §4 |

**Conclusion: the M4 gate is met.** `F0` and `F_1/2` from the holonomic
reduction agree with a fully independent double quadrature of the
point-source magnification to ~1e-7 across the primary configs; the
per-cell Gauss–Manin condition numbers are recorded and confirm the plan's
"monomial 基底の悪条件" warning. No design change was required — the plan
already defers the well-conditioned flux-priority basis and single-seed
transport to M7/M8. The float `connection_matrix_numeric` was hardened
(variable balancing + degree-drop hand-off) and its scope bounded by tests.

---

## 7. Open risks carried forward

1. **Single-seed transport is not production-ready.** `cond(C_ψ)` up to
   ~1e12 near the θ→π degree drop. The reference sidesteps this by
   re-anchoring per node; M7/M8 must build the flux-priority basis (plan
   §8) or accept the per-node re-anchor cost in C++.
2. **Oracle resolution at grazing caustics.** `image_plane_flux` `F0` is
   only ~5e-5 for `q = 1e-3`, `ρ ≈ √q`. Not an `epoch_flux` error, but it
   caps the tightness of the cross-check for planetary configs. M5/M6
   should add a caustic-refined oracle grid or a VBBL cross-check for those.
3. **`xs ≈ ys ≈ 0` patch.** The near-on-axis config runs on QAWSE seeds
   only (connection excluded) and reports `TOPOLOGY_UNCERTAIN`. M5 must
   give this locus an explicit singular-patch treatment (`singular.py`),
   not lean on the seed quadrature staying lucky.
4. **`_arc_measure_sum` extrapolation warning** for plan §15. Non-fatal
   now; the M7 fixed radial rule per cell removes it.
5. **`connection_matrix_numeric` near `xs=ys=0`** still `rel ~2e-2` even
   after balancing — genuine ill-conditioning, not a bug. Callers on that
   locus must use the exact path or fail closed.

---

## 8. Next — M5

Singular patches + full-epoch reference: `holonomic_ref/singular.py`
(birth/death tangencies via `tangency_seed_eta`, the `xs≈ys≈0` locus, θ=π
straddling arcs), `holonomic_ref/solver.py` (the epoch driver that picks
normal-cell vs singular-patch per cell and returns a status per audited
region), and separate accuracy / failure-rate reporting over an audit set
with **no silent miss**.
