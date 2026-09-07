# Checkpoint — forced full-solve benchmark methodology + D14-into-algebraic experiment

Status: **IN PROGRESS** (autonomous). Supersedes the priority order in
`checkpoint_M8.md` step >= 3: the `classify_cells / arcs_at(3072)` deep
optimisation is **on hold** until this comparison decides M8's real priority.

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
| 3 | algebraic + D14 + cont | D14 + D14 cells seed `(m,v)` root-pair continuation | reduced cold re-solve |
| 4 | holonomic M7/M8       | `radial_events` | holonomic transport + fused LD/Jac |

Frame map (algebraic native -> holonomic PrimaryFrame), exact — the two
fixed-radius boundary quartics are the same polynomial:

    pf.a   = lens.lens_position
    pf.m0  = lens.m2                              (mass at the origin)
    pf.X   = source.x + lens.lens_position * lens.m1
    pf.Y   = source.y
    pf.rho = source_radius

Variant 2 changes **only** band discovery; `integrate_radial_interval`,
`evaluate_radial_integrand`, the quartic solves, the `(m,v)` transport, the LD
moment series and the ForwardJet<5> Jacobian are byte-for-byte the current
backend.

Variant 3 additionally uses the D14 cell boundaries (exact radii where an
angular root pair is born/dies) to guarantee `(m,v)` continuation is valid
across each cell interior, cold-solving at most once per cell instead of
whenever the transported pair fails its residual gate.

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
_(pending — harness build in progress)_
