# Checkpoint M7 — C++ production port of the value+Jacobian path

Date: 2026-09-07 (in progress)
Branch: `claude/holonomic-solver-m1` (worktree `…-b7d3cd`)
Status: **in progress**

M7 ports the M6 `epoch_jacobian` path — value + 5-component Jacobian from
one per-cell Gauss–Chebyshev radial reconstruction — to C++, **and drives
it below the incumbent cost in the ordinary binary-scale regime**, not
only in the extreme-`q` tail.

## Adoption gate (design decision 20 — binding)

The re-framing proposed in `checkpoint_M0.md` §4 was **retracted by the
user**. The M7 goal is end-to-end speedup across the *normal* binary-scale
regime as well as bounded-tail behaviour. Adoption criteria:

1. **median ≥ 2× faster** than the incumbent (`binary_ray_shooting` value
   + central-FD 5-Jacobian, `evidence/holonomic/baseline_M0.json`). This
   is the original plan §14 condition and is **kept**.
2. **p90 / p95 / p99 non-regressing**, ideally substantially faster
   (the incumbent's slow tail is the 9.3 % extreme-`q` band; the holonomic
   cost is bounded there). All four percentiles are always reported.
3. **Same accuracy / failure coverage** as M5/M6 — audit median rel
   ≤ 1e-3, zero silent miss, fail-closed on the flagged statuses.
4. **Analytic Jacobian quality** — grad_mu agrees with an independent
   central FD of the reference to the M6 bars (plan15/resonant ≤ 1e-3;
   small-ρ ∂μ/∂ρ, ∂μ/∂a oracle-limited ~1e-2).

The Python reference is 0.4 s and a naive C++ port is *estimated* at
3–12 ms; **that estimate is a starting point to cut further**, not a
target. The normal binary-scale case (incumbent ~2 ms) must end up clearly
faster. If median 2× is not reached, the response is **not** to move the
goal — it is to decompose the profile, name the dominant term, and put a
concrete optimization on the M8 list.

### Mandatory optimization targets (design decision 20)

* D14 event solve / certificate **amortization** (one exact enumeration per
  trajectory, reused across epochs / re-used structure across nearby `R`).
* **Seed / re-anchor reduction** — the M6 jet re-anchors the period at
  every radial node; cut redundant re-seeding, warm-start from the
  previous node / previous epoch.
* Gauss–Manin **connection matrix precompute / factor reuse** (LU / QR kept
  and re-applied, not refactored per call) where the transport path is
  brought in.
* **Full-cell transport stabilization** (the `θ→π` degree drop,
  cond(C_ψ) ~ 1e12 — `checkpoint_M4.md`).
* **Fixed-size SIMD** — `Jet<double,5>` and the GC-1(64) angular pass are
  fixed-width; vectorize.
* **No heap allocation / no dynamic dispatch** on the hot path — all
  buffers stack / fixed arrays, no `std::function`, no virtuals.
* **Branchless hot path** — arc accumulation, sign tests, weight application.
* **Geometry reuse / warm-start within a trajectory** — cell bounds,
  quartic coefficients, root locations move slowly between epochs.
* **Fused value + Jacobian evaluation** — one pass produces μ, ∇μ, ∂μ/∂u
  (already the M6 contract; keep it fused in C++, no second sweep).

---

## 1. Isolation (design decision 19, refined)

The shared `site-packages/_lcbinint*.so` is imported by every other session
and the user's sweeps; a `pip install` rebuild overwrites it globally
(`project_editable_install_serves_stale_so.md`). M7 therefore:

* builds into `build-holonomic-m7/` in the worktree via **direct CMake**
  (or a standalone `g++` for the pure-C++ harness), never `pip install`;
* guards every M7 CMake target behind `option(LCBININT_BUILD_HOLONOMIC_M7
  OFF)` so a normal configure of this tree is byte-unchanged;
* ships the Python binding as a **separately named** module
  `_lcbinint_holonomic_m7` (not `_lcbinint`), imported by explicit path in
  tests — no `sys.modules` collision, the shared `.so` is never touched;
* keeps a pure-C++ `test_holonomic_m7` executable (no Python at all) as the
  primary correctness harness, comparing against Python-reference values
  dumped by `tests/holonomic_cpp/dump_reference.py` into a flat TSV
  (`evidence/holonomic/m7_reference.tsv` — no JSON dependency is vendored).

## 2. Module layout — `src/lcbinint/magnification/holonomic/`

| file | ports (Python `holonomic_ref/…`) | status |
|---|---|---|
| `lens_frame.hpp` | `polynomial_family.LensParams` | **done** — `LensParams`, `PrimaryFrame::from`, `m0=1/(1+q)` |
| `boundary_polynomial.hpp` | `polynomial_family.boundary_quartic{,_dp}` | **done + validated** (1e-12, 245 samples) |
| `phi.hpp` | `jacobian.phi_grad`, `topology._phi` | **done + validated** (1e-11, 245 samples) |
| `poly_roots.hpp` | `np.roots` (quartic), `sympy` real-root isolation (D14/p4/L) | **done + validated** — templated Aberth–Ehrlich; quartic at `double`, D14/p4/L at `__float128` |
| `radial_events.hpp` | `radial_events.{_P_coeffs_in_R_exact,_d14_poly,_p4_poly,radial_events,_double_root_is_real}` | **done + validated** — `__float128` poly arithmetic build of D14(v)/p4(R)/L(v), AE roots, exact-oracle match on all 7 cases |
| `cells.hpp` | `topology.classify_cells` (boundary use only) | pending |
| `radius_terms.hpp` | `jacobian.{arc_intervals,radius_terms,polish_endpoint,_real_root_thetas}` | pending |
| `epoch_jacobian.{hpp,cpp}` | `jacobian.{flux_jacobian,epoch_jacobian,_internal_to_user_jac}` | pending |
| `status.hpp` | plan §12.2 enum | pending |

Binding: `python/bind_holonomic.cpp` → `_lcbinint_holonomic_m7.epoch_jacobian(xs,ys,rho,q,a,u,barycentric,n_r) -> {mu, grad_mu[5], dmu_du, F0, F_half, status, notes}`.

## 3. Correctness plan (oracle = Python `holonomic_ref`)

1. `dump_reference.py` writes `evidence/holonomic/m7_reference.tsv`: for
   the 7 CASES (benign, close, resonant, plan15, wide-planet, cusp,
   tiny-rho) — `PF` (a,m0,X,Y,rho,r_max); `EV` radial events
   (radius/kind/physically_real); `SAMPLE` 35 (R,θ) points with
   `phi_grad` + `boundary_quartic{,_dp}`; `RT` 11 radii of `radius_terms`;
   `EPOCH` u∈{0,0.5} `epoch_jacobian`.
2. `test_holonomic_m7.cpp` reproduces each and asserts:
   * `boundary_quartic{,_dp}`: exact to 1e-12 rel — **PASS** (245 samples);
   * `phi_grad`: 1e-11 rel — **PASS** (245 samples);
   * `radial_events`: every reference radius matched within 1e-7, kinds and
     `physically_real` flags identical, `r_max` 1e-9 — **PASS** (7 cases,
     94 events); D14 built in `__float128` with no catastrophic
     cancellation, roots via AE, matches sympy exact real-root isolation
     including the near-double clusters (`resonant`, `wide-planet`,
     `tiny-rho`) that a `double` companion silently merges;
   * `radius_terms`: `f0,fh,df0,dfh` to 1e-9 rel, `reliable` identical —
     pending;
   * `epoch_jacobian`: `mu` 1e-9, `grad_mu` to the M6 bars, `status` equal
     — pending.
3. Only then wire the benchmark and compare distributions to
   `baseline_M0.json` against the four criteria above.

## 4. Progress log

* 2026-09-07a — `lens_frame.hpp`, `boundary_polynomial.hpp`, `phi.hpp`
  written; `dump_reference.py` emits `m7_reference.tsv` (7 cases, 245
  samples, 435 lines); `test_holonomic_m7.cpp` compiles standalone
  (`g++ -std=c++17 -O2 -I src … -lquadmath`) and passes 9065 checks —
  `boundary_quartic`/`boundary_quartic_dp` to 1e-12, `phi_grad` to 1e-11.
* 2026-09-07b — `poly_roots.hpp` (templated Aberth–Ehrlich, `Cplx<R>`, no
  heap on the double path) + `radial_events.hpp` (`__float128` build of
  D14(v), p4(R), L(v); `double_root_is_real` via the quartic's cubic
  derivative). `test_holonomic_m7.cpp` now also checks `radial_events` +
  `r_max`: **9256 checks, 0 failures** — every event radius within 1e-7 of
  the sympy-exact oracle, kinds/flags identical, on all 7 cases.
* (next) `radius_terms.hpp` (arc intervals, Newton polish, GC-1(64)
  angular pass, IFT-θ endpoint derivs), then the fused `epoch_jacobian`
  (per-cell GC-1(n_r) radial rule, `_internal_to_user_jac`); `status.hpp`;
  CMake guard + `_lcbinint_holonomic_m7` binding; benchmark vs
  `baseline_M0.json` against the four decision-20 criteria.
