# Differentiable algebraic-boundary finite-source kernel — design

*Branch: `feature/algebraic-boundary-autodiff` (off `algebraic-boundary-cpp`).
Do not merge to `master`; do not push to origin.*

## Goal

A parameter-differentiable version of the experimental polar algebraic-boundary
finite-source binary kernel (`experimental_algebraic_boundary_integral`), for
fast + accurate HMC (numpyro NUTS).

- **With limb darkening, beat VBM decisively** — the primal already does
  (linear LD 1.57x @ 1e-3, 2.78x @ 1e-4; 15/17 wins) — *and* return an exact
  analytic Jacobian `d(mag)/d(source_x, source_y, rho, q, separation)`.
- **Do not lose primal speed.** The primal `double` path is run unchanged; the
  derivative is a single extra forward-mode `Jet<5>` sweep over the *frozen*
  geometry. `O(1)` in root solves, no tangency Newton, no VBM.

## Reference philosophy (mainstream lcbinint inverse-ray, `bind_jax_ir.cpp`)

1. Run the discrete machinery in `double`; **freeze** every discrete decision
   (`jax.lax.stop_gradient` / `.value`): component/band count, interval
   classification, real-quartic-vs-Skowron path, adaptive panel partition, all
   epsilon-threshold branches, Boolean bisections.
2. Model the smooth quantities analytically at the converged geometry and run
   **one** 5-wide forward `JetBase<5>` pass that carries the parameter
   derivatives. First order only (`_config.py` rejects 2nd+).
3. Measure-zero transition set (support appears/disappears) → honest `NaN`
   Jacobian row, gated by a two-form disagreement estimator (`grad_tol`).

This mirrors microlux `native/vbm_ffi.cpp::finite_contour_magnification`.

## Notation (from `algebraic_boundary.cpp`)

Native binary frame: lenses at `0` and `a` (`= AlgebraicBoundaryLens::
lens_position`), masses `m1, m2` (`m1 + m2 = 1`). HMC parameters
`p = (s_x, s_y, ρ, q, s)`; `a, m1, m2` are explicit functions of `(s, q)` via
`make_binary_lens_mapper` (**TODO confirm exact map + its p-derivatives — feeds
every ∂/∂p**).

Fixed image radius `R`, `z = R e^{iθ}`. Shifted source
`ζ = (s_x + a·m1) + i s_y`. The primal builds (`algebraic_boundary_quartic_fixed`):

```
n0 = -ζ R²                       (complex)
n1 = R(R² - 1 + a ζ)             (complex)
n2 = a(m2 - R²)                  (real)
E  = ρ² a R³                     (real, "radial_denominator_edge")
p0 = n0 n2
p1 = n0 conj(n1) + n1 n2 + E
p2 = |n0|² + |n1|² + n2² - ρ²(R⁴ + a² R²)     (real)
```

self-inversive quartic coeffs `{p0, p1, p2, conj(p1), conj(p0)}`, then
(`self_inversive_trig_quartic`, reading coeff[4],[3],[2]):

```
A = Re(p0),  B = -Im(p0),  C = Re(p1),  D = -Im(p1),  Epsilon = p2
```

The boundary membership function on the circle (`evaluate_trig_quartic`):

```
F  (θ) =  Epsilon + 2(C cosθ - D sinθ) + 2(A cos2θ - B sin2θ)
F' (θ) =            2(-C sinθ - D cosθ) + 4(-A sin2θ - B cos2θ)
F''(θ) =            2(-C cosθ + D sinθ) + 8(-A cos2θ + B sin2θ)
```

`F(θ_k) = 0` ⟺ image-plane ray at `(R, θ_k)` lands on the source edge.
Inside intervals `[θ_k, θ_{k+1}]` are where `phi_value > 0`.

Radial integrand (scalarised `combined_flux` of the angular `Moments`):

```
J(R; p) = (1-c-d)·m0(R) + c·mhalf(R) + d·mquarter(R)
m0     (R) = R · Σ_intervals (θ_{k+1} - θ_k)                         [exact]
mhalf  (R) = Σ_intervals R·hw·Σ_i w_i·√φ(angle_i)                    [GK 10/21, sine-mapped]
mquarter(R)= Σ_intervals R·hw·Σ_i w_i·√√φ(angle_i)
angle_i = mid + hw·sinemap(s_i),  mid = ½(θ_k+θ_{k+1}), hw = ½(θ_{k+1}-θ_k)

mag(p) = [ Σ_bands ∫_{R_lo(p)}^{R_hi(p)} J(R;p) dR ] / N,   N = πρ²(1 - c/3 - d/5)
```

## Analytic derivatives

### (A) Crossing angles `θ_k` — quartic IFT

`(A,B,C,D,Epsilon)` are explicit polynomials in `(ζ, a, m1, m2, ρ², R)`.
Build them with the **templated** coefficient builder on `JetBase<5>` inputs
`(s_x, s_y, ρ, q, s)` ⟹ `(A..Epsilon)` come out as `Jet`s.

At each frozen crossing (primal `θ_k` value held as a `double`):

```
θ_k.value      = primal
θ_k.derivative = - F.derivative / F'.value            (F built with Jet coeffs, double θ_k)
```

`F.derivative` already contains the full `dF/dp` (all channels, incl. any
`R(p)` channel when `R` is fed as a Jet, see C). Zero extra solves.

Gate: `|F'(θ_k)| ≤ τ · scale` ⟹ two angular crossings merging ⟹ NaN row.

### (B) Radial band endpoints `(R*, θ*)` — 2×2 tangency IFT (replaces the FD)

`solve_radial_tangency` (`algebraic_boundary.cpp:1054-1070`) currently takes the
radial derivative by **centered finite difference**. Replace with analytic:

Tangency system `G(R,θ;p) = ( F(θ), F'(θ) ) = 0`. All R-dependence of `F` is
through `(A..Epsilon)(R)`, so

```
∂F/∂R   = Σ_κ (∂F/∂κ)·(dκ/dR),   κ ∈ {A,B,C,D,Epsilon}
∂F'/∂R  = Σ_κ (∂F'/∂κ)·(dκ/dR)
```

with the closed-form `dκ/dR` from `dn0/dR, dn1/dR, dn2/dR, dE/dR,
d[ρ²(R⁴+a²R²)]/dR` (all polynomial), and

```
∂F/∂A = 2cos2θ   ∂F/∂B = -2sin2θ   ∂F/∂C = 2cosθ   ∂F/∂D = -2sinθ   ∂F/∂Epsilon = 1
∂F'/∂A = -4sin2θ ∂F'/∂B = -4cos2θ  ∂F'/∂C = -2sinθ ∂F'/∂D = -2cosθ  ∂F'/∂Epsilon = 0
```

Implementation trick: build `(A..Epsilon)` on `JetBase<6>` where lane 5 is `R`.
Then `dκ/dR = κ.derivative[5]` and `∂κ/∂p = κ.derivative[0..4]` from the *same*
builder. IFT:

```
[ ∂F/∂R    F'(θ*)  ] [ dR*/dp ]   = - [ ∂F/∂p  ]
[ ∂F'/∂R   F''(θ*) ] [ dθ*/dp ]       [ ∂F'/∂p ]
```

solved once per band endpoint, 5 params at once.

Gate: `det = ∂F/∂R·F''(θ*) - F'(θ*)·∂F'/∂R`; `|det| ≤ τ·scale` or
`|F''(θ*)| ≤ τ` ⟹ band about to appear/vanish ⟹ NaN row.

### (C) Integral — Leibniz via reference-space change of variables

Map each band's outer quadrature to a fixed reference interval:

```
R(ξ; p) = mid_R(p) + hw_R(p)·sinemap(ξ),  mid_R = ½(R_lo+R_hi), hw_R = ½(R_hi-R_lo)
∫_{R_lo}^{R_hi} J(R;p) dR = ∫_ref J(R(ξ;p); p) · hw_R(p)·sinejac(ξ) dξ
```

Feed `R(ξ;p)` as a `Jet` (from B's `dR_lo/dp, dR_hi/dp`) at the **frozen** GK
node positions `ξ_i` (adaptive panel partition frozen by primal). Then
`d/dp` of the LHS = `∫_ref d/dp[ J(R(ξ;p);p)·hw_R·sinejac ] dξ` — the moving
limits `J(R_hi)dR_hi/dp - J(R_lo)dR_lo/dp` are **automatically included** by the
change of variables; no separate boundary flux term.

Inner angular moments in `Jet`:

- **uniform** `m0`: `R · Σ (θ_{k+1} - θ_k)` with `θ_k` Jets from (A) — exact,
  no angular quadrature, fully differentiable.
- **LD** `mhalf, mquarter`: run `angular_moments_rule` in `Jet` — Jet limits
  `(θ_k, θ_{k+1})` ⟹ Jet `mid, hw` ⟹ Jet `angle_i` ⟹ Jet `φ(angle_i)` (Jet
  lens map) ⟹ `scalar_sqrt` / `scalar_pow(·,0.25)`. Frozen: `w_i, s_i`, sine
  map of `s_i`. This is the full Leibniz result (integrand + moving angular
  endpoints) in one pass. The `√φ→0` endpoint is tamed by the sine-map
  Jacobian exactly as in the primal.

Then `combined_flux`, `/N(p)` (N a Jet in ρ,c,d) — all Jet. Output
`mag.value` (== primal) and `mag.derivative[0..4]` (the Jacobian row).

### (D) Measure-zero transition handling (`grad_tol`)

Same contract as microlux Phase 2:

- Any (A)/(B) gate trips ⟹ whole-epoch Jacobian row = `NaN` (honest refusal;
  HMC recovers from neighbours — "confidently wrong gradient" is worse).
- Two-form estimator per epoch: `est[p] = |∂/∂p of (kronrod - gauss) combined
  flux| / N`, reduced `max_p est[p] / max(1, |dmag/dp|)`. Over `grad_tol` ⟹
  NaN row. `grad_tol` FFI scalar; `None` → `10·tol` clamped `[1e-4, 1e-1]`.
- Emit `grad_error[epoch]` output (like microlux `gradient_error`).

## Speed argument

| stage | primal | with autodiff |
|---|---|---|
| root solves / tangency Newton | N / M | **N / M (unchanged)** |
| extra work | — | 1 templated `Jet<5>` sweep of frozen nodes + one 2×2 per band endpoint |

Primal `double` path byte-identical ⟹ primal speed preserved. Derivative cost
is the `Jet` arithmetic constant (~5-8x) on the *cheap* post-pass only.
More accurate than mainstream inverse-ray's per-cell affine level-set model —
here the boundary is exact (quartic roots).

## Scope

- **In:** polar path (`experimental_algebraic_boundary_integral`), binary
  (5 params), uniform + linear + two-coefficient LD.
- **Out (later):** Cartesian sextic, triple lens, 2nd-order derivatives.
- Seeds / `augmented_image_seeds` / caustic branches / band discovery: all
  frozen (`stop_gradient`), decided by primal `.value`.

## Implementation plan

1. **Extract a templated core** from `algebraic_boundary.cpp` into
   `algebraic_boundary_detail.hpp` (coeff builder, `evaluate_trig_quartic`,
   `angular_moments_rule`, `combined_flux`, sine map). Primal instantiates
   `<double>` and stays byte-identical (threshold branches, `std::sort`,
   caches remain `double`-only control flow, outside the template).
2. `algebraic_boundary_autodiff.{hpp,cpp}` — `AlgebraicBoundaryJacobian`
   driver: run primal `double` → capture frozen geometry (bands, panels,
   `θ_k`, classification, path choices) → (B) 2×2 per endpoint → (C) one
   `Jet<5>` sweep → (D) gates + estimator.
3. `python/bind_jax_algebraic.cpp` — FFI targets
   `lcbinint_algebraic_boundary_autodiff_f64_v1` (+ `_jacobian_f64_v1`),
   batch signature: `rho`, `q` = BufferR0 batch scalars; `separation`,
   `source_x`, `source_y` = BufferR1 per-epoch; attrs `moment_mode`,
   `grad_tol`. Outputs `mag[n]`, `jac[n,5]`, `grad_error[n]`.
4. `python/lcbinint_jax/algebraic.py` — `jax.custom_jvp` wrapper.
5. Tests: value parity vs primal (`~1e-13`), grad vs tight FD
   (`1e-6..1e-8`) off caustic, `NaN` row on caustic grazing, uniform +
   linear LD.

## Open questions / risks

- `make_binary_lens_mapper`: exact `(a, m1, m2) ↔ (s, q)` and derivatives.
- Refactor must not perturb primal numerics — verify with the existing
  `tests/unit/test_algebraic_boundary.cpp` corpus (byte-identical primal).
- `scalar_pow(·, 0.25)` / `scalar_sqrt` derivative guards at `φ = 0`.
- Confirm primal never adaptively subdivides the *angular* rule (single
  10/21 pass observed — if so, frozen-θ-value + Jet-limit is exact).
