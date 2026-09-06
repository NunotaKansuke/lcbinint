# Checkpoint M2 — radial-cell topology classification

Date: 2026-09-07
Branch: `claude/holonomic-solver-m1` (M1 + M2 share the feature branch;
the `…-b7d3cd` name in earlier docs is the worktree, not the checked-out branch)
Status: **complete** (adoption gate for M2 met)

M2 turns the radial-event list from M1 into a **cell decomposition** of
`(0, R_max)` and classifies the image topology of each cell, producing the
`CellPlan` list and the arc-incidence lineage that the period-transport stage
(M4) walks.

---

## 1. What was implemented

| File | Purpose |
|---|---|
| `python/lcbinint/holonomic_ref/topology.py` | cell decomposition, per-cell topology (`empty` / `full` / `arcs` with 2 or 4 crossings), arc extraction, birth/continue/death lineage |
| `tests/holonomic/test_topology.py` | 61 cases: per-cell consistency, arc-endpoint correctness, brute-force agreement, lineage conservation, full-circle detection, 30 random configs |
| `tests/holonomic/conftest.py` | puts the source-tree `holonomic_ref` on `sys.path` (installed wheel is the main checkout) |

Still a **pure-Python reference** — dense sign-scan of `phi` straight from the
lens equation plus Brent refinement. No algebraic short-cuts; touches no
existing solver.

### Public API (`holonomic_ref`)

* `arcs_at(R, a, m0, xs, ys, rho) -> (kind, n_crossings, arcs)` — topology of a
  single circle `|z| = R`.
* `classify_cells(LensParams, *, samples_per_cell=3) -> TopologyResult` — the
  full decomposition.
* dataclasses `Arc` (oriented `phi >= 0` interval), `CellPlan`, `ArcTransition`,
  `TopologyResult`.

---

## 2. Cell construction

1. `radial_events` gives the sorted event radii and `R_max`.
2. Events sharing a radius (to `max(1e-9, 1e-7·R)`) are **merged into one
   boundary** carrying the joined kind string (e.g. `chart_p4+physical_complex`).
   *Reason:* on the axis (`ys = 0`) a `chart_p4` and a `physical_complex` event
   land at exactly the same radius; without merging the zero-width cell between
   them is dropped and the two neighbours then disagree on the shared-boundary
   label. Autonomous fix, per "fix it where safely judgeable and record why".
3. Cells are the open intervals between consecutive boundaries
   (`0`, merged events…, `R_max`); intervals narrower than `1e-11` are skipped.
4. Each cell is probed at `samples_per_cell` radii spread over the central
   64 % of its width. If the samples disagree on `(kind, n_crossings)`, or the
   representative crossing count is odd, the cell is flagged
   `status = "TOPOLOGY_UNCERTAIN"` and the whole result inherits it
   (**fail-closed**, no silent guess).

`arcs_at` classifies one circle by sampling `phi` on a 3072-point `theta` grid:
all-positive → `full`, all-negative → `empty`, otherwise every sign change is
Brent-refined to a boundary point, crossings are paired into oriented `Arc`s by
their rising/falling sense, and an odd crossing count returns
`("arcs", n, ())` — reported, never patched.

`deg_t P = 4` bounds a circle at 4 real boundary crossings, i.e. **at most 2
arcs**; the tests assert this.

---

## 3. Arc incidence lineage

`_match_arcs` pairs the arcs of adjacent cells by nearest arc **midpoint**
(circular distance, threshold `0.6 rad`) and emits `ArcTransition` records:

* `continue` — arc present on both sides;
* `birth` — arc only in the upper cell;
* `death` — arc only in the lower cell.

This is the graph M4 needs: which arc of cell *k* feeds which arc of cell
*k+1*, and where the period basis must be re-seeded (births) or retired
(deaths).

---

## 4. Verification results

`tests/holonomic/test_topology.py` — 61 cases, all pass (~42 s). 6 physical
configs (plan §15, resonant `a=0.9`, planetary `a=2.5`, close `a=0.55`,
on-axis `a=1.35`, big source on centre `a=1.1 ρ=0.25`) plus 30 random.

| Check | Assertion |
|---|---|
| per-cell stability | `(kind, n_crossings)` from `arcs_at` is identical at 4 extra radii spanning each cell |
| crossing count | every cell has `n_crossings ∈ {0,2,4}` and `len(arcs) ≤ 2` |
| arc endpoints | `|phi(R, θ)| < 1e-7` at each endpoint; `phi > 0` at the arc midpoint; `phi < 0` just outside both ends |
| brute-force agreement | `n_crossings` equals the sign-change count of `phi` on a 4096-point θ grid at `r_mid`; `empty`/`full` cells have 0 crossings |
| lineage conservation | for each cell, `#{continue,death}` out the top = `#arcs` and `#{continue,birth}` in the bottom = `#arcs` |
| adjacent-cell changes | any change in `(kind, n_crossings)` between neighbours is backed by a real brute-force crossing-count change straddling the boundary (or a genuine `empty↔full` flip) |
| full-circle detection | large source on the primary produces a `full` cell; `phi ≥ 0` at 200 random angles there |
| random robustness | 30 configs (`q∈[1e-5,1]`, `a∈[10^±0.7]`, `ρ∈[1e-3,0.25]`): contiguous cells covering `[0, R_max]`, **zero `TOPOLOGY_UNCERTAIN`** |

Full holonomic suite: **154 passed** (M1 + M2), ~84 s.

### Cross-checks against M1 findings

* Plan §15 case → 14 cells. The two-arc region is cell 7,
  `R ∈ (0.89133, 0.91110)` (`n_crossings = 4`), exactly between the two
  `physical_real` events M1 identified as the second-arc birth/death pair.
* On-axis `a=1.35` → the symmetric `±t*` band pair appears as one cell,
  `R ∈ (0.86984, 0.92359)` with `n_crossings = 4` / 2 arcs — the 0→4→0 change
  M1's exact enumerator recovered (and float64 missed).

---

## 5. Open risks carried forward

1. **`physical_complex` as a cell boundary (from M1 risk 2).** M2 data: these
   events *do* sometimes separate cells with different arc counts (plan §15
   cell 2↔3 boundary at `R≈0.72464` is `physical_real`, but the
   `physical_complex` at `0.77033` sits inside a constant 2-arc stretch in
   §15 yet coincides with real topology change in the on-axis case). Keep them
   for now; revisit when M4 shows whether the extra (often zero-width) cells
   cost anything.
2. **Midpoint arc matching.** `_match_arcs` uses angular proximity with a fixed
   `0.6 rad` gate. Fast-moving arcs near a caustic cusp could be mismatched
   across a wide cell. M4 transport will move endpoints analytically; the
   lineage here is only the combinatorial skeleton and is re-checked by the
   conservation test. Flagged for a tighter (transport-based) match at M4.
3. **Grid resolution.** `arcs_at` uses 3072 θ samples. A pair of boundary
   crossings closer than `2π/3072 ≈ 2 mrad` on the *same* circle would be
   missed. Not observed in any test config; M4/M5 should replace the scan with
   the algebraic root count of `P(t; R)` once the period machinery exists.

---

## 6. Next — M3

Period reduction reference: 7-form monomial basis `η_k = t^k dt / Y` →
residue-free 6-form observed basis; the Gauss–Manin connection matrices; verify
coefficient identities, residue conditions, and 7D/6D period values against
direct angular integration at `rtol ~ 1e-10`. Files
`holonomic_ref/period_reduction.py`, `holonomic_ref/connection.py`.
