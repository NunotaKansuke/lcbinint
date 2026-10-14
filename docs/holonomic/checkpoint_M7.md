# Checkpoint M7 — C++ production port of the value+Jacobian path

Date: 2026-09-08
Branch: `claude/holonomic-solver-m1` (M1–M7 share the feature branch; the
`…-b7d3cd` in the milestones header is the worktree, not the branch)
Status: **complete** — all four decision-20 adoption criteria met.

M7 ports the M6 `epoch_jacobian` path — value + 5-component Jacobian from
one per-cell Gauss–Chebyshev radial reconstruction — to C++, **and drives
it below the incumbent cost in the ordinary binary-scale regime**, not
only in the extreme-`q` tail.

## Adoption gate (design decision 20 — binding)

The re-framing proposed in `checkpoint_M0.md` §4 was **retracted by the
user**. Adoption criteria and the M7 result:

| # | criterion | target | M7 result | verdict |
|---|---|---|---|---|
| 1 | median ≥ 2× faster than incumbent | C++ median ≤ 7.44 ms | **4.77 ms** (3.1× median/median, 3.9× median of per-case ratios) | **PASS** |
| 2 | p90/p95/p99 non-regressing, ideally ≪ | ≤ incumbent 145 / 4392 / 4628 ms | **8.15 / 8.37 / 8.44 ms** (18× / 525× / 548×) | **PASS** |
| 3 | same accuracy / failure coverage | audit median rel ≤ 1e-3, 0 silent miss, fail-closed matches M5/M6 | μ **bit-identical** to M6 (n=104); status **0 mismatches** vs M6 (n=108); μ vs incumbent median 2.7e-5 / p99 7.6e-3 | **PASS** |
| 4 | analytic Jacobian quality | grad_mu to the M6 bars | vs M6: median 5.4e-10, p90 6.0e-7, max 2.1e-2 (caustic-cross, M6-oracle-limited); dmu_du max 9.8e-8 | **PASS** |

Benchmark: `bench_holonomic_m7`, best-of-200 per case, `taskset -c
0-7`, host load ≈ 13 (logged before/after each run), 108 `(config, u)`
points = the M0 baseline set. Incumbent numbers from
`evidence/holonomic/baseline_M0.json` (`binary_ray_shooting` value +
central-FD ×11 5-Jacobian): median 14.89 ms, p90 145.3 ms, p95 4392 ms,
p99 4628 ms.

Evidence: `evidence/holonomic/m7_accuracy_coverage.txt` (criteria 3 & 4,
full tables + the fail-closed case list + the worst-disagreement rows).

The "min speedup 0.0×" line in the bench output is an artefact of four
points whose recorded incumbent `t_jac` is a sub-0.1 ms probe stub (the
incumbent bailed immediately) — not a real regression; those same four
points are the two fail-closed cases below (each at u=0 and u=0.5).

---

## 1. Isolation (design decision 19)

The shared `site-packages/_lcbinint*.so` is imported by every other session
and the user's sweeps; a `pip install` rebuild overwrites it globally
(`project_editable_install_serves_stale_so.md`). M7 therefore:

* is a **standalone CMake project** at `tests/holonomic_cpp/CMakeLists.txt`
  with its own `project()`, **not** `add_subdirectory`'d from the top-level
  `CMakeLists.txt` — a normal configure of this tree is byte-unchanged.
  Build:
  ```
  cmake -S tests/holonomic_cpp -B build-holonomic-m7 -DCMAKE_BUILD_TYPE=Release
  cmake --build build-holonomic-m7
  ```
  (`/build*/` is git-ignored.) The header-only solver in
  `src/lcbinint/magnification/holonomic/` is compiled directly; nothing
  links or rebuilds `liblcbinint`.
* ships the Python binding as a **separately named** module
  `_lcbinint_holonomic_m7` (`python/bind_holonomic.cpp`) — not `_lcbinint`,
  no `sys.modules` collision, the shared `.so` is never touched. Confirmed
  untouched (mtime unchanged) after a full M7 configure + build + import.
* keeps a pure-C++ `test_holonomic_m7` executable (no Python) as the
  primary correctness harness, comparing against Python-reference values in
  a flat TSV (`evidence/holonomic/m7_reference.tsv`, no JSON dependency).
  Wired as the `holonomic_m7_reference` ctest.

## 2. Module layout — `src/lcbinint/magnification/holonomic/` (header-only)

| file | ports (Python `holonomic_ref/…`) | status |
|---|---|---|
| `lens_frame.hpp` | `polynomial_family.LensParams` | **done + validated** |
| `boundary_polynomial.hpp` | `polynomial_family.boundary_quartic{,_dp}` | **done + validated** (1e-12, 245 samples) |
| `phi.hpp` | `jacobian.phi_grad`, `topology._phi` | **done + validated** (1e-11, 245 samples); hand-rolled complex, `__builtin_sincos` |
| `poly_roots.hpp` | `np.roots` (quartic), sympy real-root isolation (D14/p4/L) | **done + validated** — templated Aberth–Ehrlich; quartic + p4 at `double`, D14 two-stage |
| `radial_events.hpp` | `radial_events.*` | **done + validated** — `__float128` D14(v)/p4(R)/L(v) build; two-stage D14 solve (balanced `double` search → 113-bit polish → residual/finiteness-gated cold fallback) |
| `angular_rule.hpp` | GC-1st-kind `_cheb1(n)` | **done + validated** — fixed `Cheb1<64>` singleton + `Cheb1Dyn` |
| `cells.hpp` | `topology.classify_cells` (boundary use) | **done + validated** — 3-probe `arcs_at(3072)` uniformity check, `TOPOLOGY_UNCERTAIN` propagation |
| `radius_terms.hpp` | `jacobian.{arc_intervals,radius_terms,polish_endpoint,_real_root_thetas,_grid_intervals,_full_circle_terms}` | **done + validated** — 77 RT checks, f0/fh/df0/dfh + `reliable` |
| `epoch_jacobian.hpp` | `jacobian.{flux_jacobian,epoch_jacobian,_internal_to_user_jac}`, `singular.near_origin_source` | **done + validated** — 14 epoch checks; fused μ, ∇μ, ∂μ/∂u in one radial pass |
| `status.hpp` | plan §12.2 enum | **done** |

Binding: `python/bind_holonomic.cpp` →
`_lcbinint_holonomic_m7.epoch_jacobian(xs,ys,rho,q,a,u=0,barycentric=False,n_r=64)`
→ `{mu, grad_mu[5], dmu_du, F0, F_half, r_max, status, param_order}`.

Correctness harness (`test_holonomic_m7.cpp` vs `m7_reference.tsv`):
**10 397 checks, 0 failures** — 7 CASES × {245 phi/quartic samples, 92
radial events, 77 `radius_terms`, 14 `epoch_jacobian`}.

## 3. Optimizations applied (mapping to the decision-20 mandatory list)

| mandatory target | what was done | effect |
|---|---|---|
| D14 event solve amortization | **two-stage D14 solve**: balance the monomial basis (`v = s·w`, `s = |a_n/a_0|^{1/n}`), locate every root basin with a `double` Aberth–Ehrlich pass (~15 µs), warm-start a 24-iter 113-bit polish from those; residual- **and finiteness**-gated cold `__float128` fallback | radial_events **47 ms → 0.6 ms**; wide-binary p99 fixed (see §4) |
| — | `aberth` gained a `tol_override`; qf default tol 1e-30 → 1e-24; quartic AE 120 → 40 iters; p4 + chart-p4 solved at `double` | search cost down, accuracy unchanged (10 397 checks hold) |
| no heap / no dynamic dispatch on hot path | hand-rolled limited-range complex in `phi.hpp` (no `std::complex` Annex-G inf/nan bookkeeping — ~4× on every divide); fixed `std::array` jets; `Cheb1<64>` static singleton; no `std::function`, no virtuals in the radial/angular loops | `phi_grad` ≈ the single hottest routine, now branch-lean |
| branchless hot path | `__builtin_sincos` in `phi_grad`/`phi_lens`; arc accumulation over fixed 64-node GC-1 weights | — |
| fused value + Jacobian | `flux_jacobian` accumulates `F0, F_half, dF0_internal[5], dF_half_internal[5]` in **one** per-cell GC-1(n_r=64) radial pass; `epoch_jacobian` then does μ, ∇μ, ∂μ/∂u with no second sweep | matches the M6 contract |
| fixed-size SIMD | compiler auto-vec under `-O3 -march=native -funroll-loops -ffp-contract=fast -fno-math-errno`; the 5-wide param-derivative inner loops are contiguous `std::array<double,5>` | (explicit intrinsics deferred — see §6 item 4) |

## 4. Bugs found and fixed autonomously (recorded per the standing instruction)

* **Silent-failure in the D14 residual gate (wide binary).** The first
  two-stage D14 solve did a plain `double` Aberth search. For a wide
  binary (`a = 5`, config `very-wide`) the D14 coefficient spread is
  ~1e11 and `|z|^14` overflows the `double` range, so the search returned
  all `NaN`. The residual gate `if (worst > 1e-12) redo_cold` did **not**
  fire because `NaN > 1e-12` is `false` — the NaN seed was fed to the
  113-bit polish, which then also produced NaN, and `positive_real_roots`
  silently dropped every root. Result: **all D14 events lost →
  `classify_cells` sees only `empty` cells → μ = 0 returned with status
  `OK`.** A silent wrong answer, exactly the failure mode the standing
  "fail closed, never silent approximation" instruction forbids.
  Fix (two layers): (a) **balance the polynomial** before the `double`
  search (`v = s·w`), which keeps the Cauchy bound at O(10) and makes the
  `double` pass reliable for wide binaries — restoring the fast path and
  the p99 (8.4 ms, was 19.6 ms with the cold fallback); (b) **NaN-safe
  gates** — an explicit `std::isfinite` check on the seeds and
  `!(worst <= tol)` (not `worst > tol`) so any non-finite residual forces
  the cold `__float128` solve. After the fix `very-wide` returns
  μ = 1.02752307546, bit-identical to M6.

* No other silent divergence: across the 108-point set the C++ μ is
  bit-for-bit equal to M6 Python on all 104 OK points, and the status is
  identical on all 108.

## 5. The fail-closed points (criterion 3 — coverage matches M6)

Identical set in C++ and M6, both `GRADIENT_UNRELIABLE`:

| case | config | cause (verified) |
|---|---|---|
| `rand008` (u=0, 0.5) | xs −1.042, ys −0.714, ρ 1.42e-3, q 4.39e-3, a 0.195 | genuine near-tangency: a ~6.6e-4-wide cell at R ≈ 0.55 in which a thin arc is born; the 3 probe radii disagree (`arcs`/`empty`) → `TOPOLOGY_UNCERTAIN`. M6 `classify_cells` reports the same "samples disagree" detail. |
| `rand031` (u=0, 0.5) | xs −0.509, ys 0.031, ρ 0.263, q 6.45e-8, a 0.438 | extreme `q`: the planetary caustic is far below the 3072-grid resolution → `TOPOLOGY_UNCERTAIN`. M6 identical. |

Both are correct fail-closed: μ is still returned (weak-lensing regime,
μ ≈ 1.20 and 2.22) but the gradient is flagged untrustworthy. Zero silent
misses.

## 6. Deferred to M8 (with reasons — not goal-moving, per decision 20)

The median-2× gate is **met**, so these are additional headroom, not
required to pass:

1. **`quartic_topology` for `classify_cells`.** Using the boundary
   quartic's real-root **count** as the cell-probe (instead of the
   3072-point `arcs_at` grid) is ~3× cheaper per probe and catches thin
   newborn arcs the grid misses. It was implemented (`quartic_topology`
   in `radius_terms.hpp`, retained) but **reverted from the hot path**:
   on `wide-planet`/`extreme-q` it flips a handful of near-tangency
   cell-edge classifications relative to M6, shifting ∂μ/∂q by ~0.5 %.
   Both the grid and the quartic classifications are internally
   FD-consistent; deciding which is *more* correct needs an independent
   gradient oracle (a high-order Richardson FD of `binary_ray_shooting`,
   or the M4 `epoch_flux` double-quadrature differentiated). Parity with
   M6 was the safe call for M7.
2. **Aberth warm-start across radial nodes.** Seeding the per-node quartic
   AE from the previous node's roots (decision-20 "seed/re-anchor
   reduction") was tried and **fully reverted**: when the real-root
   structure changes across a cell interior (near-tangency), the warm seed
   has the wrong root count/config and AE converges to a wrong set
   (tiny-rho ∂μ/∂q went −25 vs +583). A safe version needs a root-count
   guard and a cold re-seed on mismatch — an M8 task.
3. **Gauss–Manin connection precompute / factor reuse**, **full-cell
   transport stabilization** (θ→π degree drop, cond(C_ψ) ~ 1e12). Not
   reached — the M7 reconstruction re-anchors per node (the M6 contract)
   and does not yet use the M4 transport path. This is the largest
   remaining structural optimization.
4. **Explicit fixed-width SIMD** of the 5-wide param-derivative loops and
   the GC-1(64) angular pass (currently compiler auto-vec only).
5. **In-trajectory geometry reuse / warm-start** — cell bounds and quartic
   coefficients move slowly between epochs of one light curve; the C++
   path currently recomputes `classify_cells` per epoch. A trajectory-level
   cache is an M8 item.
6. **∂μ/∂ρ direct assembly** — still built as `(large)/D − 2μ/ρ`
   (catastrophic cancellation for small ρ, `checkpoint_M6.md` §5-1). The
   C++ jet should form `∂(F/ρ²)/∂ρ` directly. M8.

## 7. Profile decomposition (for the M8 optimization plan)

At the 4.8 ms median the dominant terms (granular timing, benign config):

* per-cell GC-1(n_r=64) radial pass × arcs: **~60 %** — for each of ~64
  radial nodes per arc-bearing cell, `radius_terms` does a degree-4 AE
  (`real_root_thetas`), 2 Newton polishes, and a 64-node angular
  `phi_grad` sweep with the 5-wide derivative accumulation. **This is the
  M8 SIMD + warm-start target.**
* `classify_cells` (radial events + 3 `arcs_at(3072)` probes per cell):
  **~30 %** — `arcs_at` is 3072 `phi_lens` evals + bisections per probe.
  The M8 `quartic_topology` switch (§6 item 1) removes most of this.
* D14 two-stage solve: **~6 %** (was ~90 % before the two-stage work).
* chain rule + assembly: **< 4 %**.

M8 priority order: (1) `quartic_topology` classify with an independent
gradient cross-check; (2) SIMD the radial×angular derivative loops;
(3) trajectory-level `classify_cells` cache; (4) the M4 transport path for
full-cell states.

---

## 8. Progress log

* 2026-09-07a — `lens_frame.hpp`, `boundary_polynomial.hpp`, `phi.hpp` +
  `dump_reference.py` + `test_holonomic_m7.cpp` (9065 checks: quartic 1e-12,
  phi 1e-11).
* 2026-09-07b — `poly_roots.hpp` + `radial_events.hpp` (`__float128` D14/p4/L,
  AE roots); 9256 checks.
* 2026-09-07c — commit `34b45dd` (boundary/phi/radial-event port; decision-20
  gate reversal recorded in `milestones.md`).
* 2026-09-08 — remaining modules ported and validated (`angular_rule`,
  `status`, `cells`, `radius_terms`, `epoch_jacobian`): **10 397 checks, 0
  failures**. Two-stage D14 solve (double→qf, then balanced + NaN-safe).
  Silent-failure bug in the D14 residual gate found and fixed (§4).
  Benchmark **PASSES all four decision-20 criteria** — median 4.77 ms
  (3.1×), p90 8.15 ms, p95 8.37 ms (525×), p99 8.44 ms. Standalone `tests/holonomic_cpp/CMakeLists.txt`
  + isolated `_lcbinint_holonomic_m7` binding; shared `.so` untouched.
  Evidence: `evidence/holonomic/m7_accuracy_coverage.txt`.
