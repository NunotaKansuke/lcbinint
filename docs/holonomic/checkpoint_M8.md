# Checkpoint M8 — cutting the dominant profile terms

**Branch:** `claude/holonomic-solver-m1`
**Goal (user, 2026-09-08):** beat the algebraic-boundary **ordinary-binary
deep-solve median** while keeping the p95/p99 bounded-tail and Jacobian-
availability advantages. Attack, in order: (1) `classify_cells` /
`arcs_at(3072)` probes, (2) per-cell radial × angular pass. Do **not**
over-invest in the D14 solve.

Baseline entering M8 (checkpoint_algebraic_vs_holonomic.md), best-of, taskset -c 0-7:
- holonomic fused value+Jac: p50 4.80 ms, p90 8.13, p95 8.31, p99 8.44, max 8.47
- vs inverse-ray baseline (14.89 ms) = 3.1x
- vs algebraic ordinary-binary deep-solve (val+Jac p50 5.2 ms): holonomic slower at median, faster in tail

Fresh profile (holoprof2, per-epoch):

| case | full | classify_cells | (radial_events) | radial×angular |
|---|--:|--:|--:|--:|
| plan15 | 4.56 | 2.93 | 0.58 | 1.63 |
| resonant | 8.13 | 4.69 | 2.01 | 3.43 |
| caustic-cross | 7.32 | 6.07 | 3.78 | 1.25 |
| close-binary | 8.24 | 4.64 | 1.97 | 3.60 |
| wide-planet | 3.69 | 2.76 | 0.71 | 0.93 |

`classify_cells` beyond `radial_events` = the 3× `arcs_at(3072)` probes per
cell: ~0.0505 ms each × 3 × (13–16 cells) ≈ 2.0–2.4 ms. `quartic_topology`
(same boundary quartic the integrator root-finds) delivers the same
(kind, crossing-count) signature in ~0.0006 ms — **~85× cheaper**.

---

## Step 1 — hybrid `classify_cells` (quartic topology + coarse-grid cross-check)

`src/lcbinint/magnification/holonomic/cells.hpp`.

Per cell, per fraction {0.18, 0.50, 0.82}:
1. topology signature from `quartic_topology` (boundary-quartic real roots);
2. cross-check the mid fraction against an independent **512-point**
   `arcs_at` grid;
3. on ANY disagreement — crossing count, kind, non-uniformity across the
   three fractions, odd mid crossing count — **escalate the whole cell to
   the exact M7 path**: `arcs_at(…, 3072)` at all three fractions;
4. a cell whose two independent methods still disagree after escalation is
   `TOPOLOGY_UNCERTAIN` (fail closed, unchanged).

So the adopted classification is always either (a) confirmed by two
independent methods (quartic + 512-grid), (b) exactly the M7 3072-grid
result, or (c) fail-closed. Never a lone unchecked quartic guess.

### Validation (3-way harness, 108 (config,u) points, reps=90)

- **holonomic μ: bit-identical to M7 on all 108 cases** (per-case rel diff
  0.0 everywhere) — the hybrid never adopted a classification the 3072-grid
  would not have.
- **status: 0 changes** — same 104/108 OK, same fail-closed set
  (rand008, rand031 near-tangency).
- `test_holonomic_m7` correctness harness: 10397 checks / 0 failures.

### Latency (best-of-90, taskset -c 0-7, load ~11–12)

| slice | metric | M7 | M8 step 1 |
|---|---|--:|--:|
| ALL 108 | p50 | 4.80 | **3.03** |
| | p90 | 8.13 | 5.82 |
| | p95 | 8.31 | 6.07 |
| | p99 | 8.44 | 6.13 |
| | max | 8.47 | 6.15 |
| ordinary-binary deep-solve (n=14) | holo p50 | 7.21 | **5.11** |
| | holo p95 | 8.46 | **6.08** |
| | algebraic val+Jac p50 | 5.18 | 5.18 |
| | algebraic val+Jac p95 | 12.51 | 12.51 |

- **vs inverse-ray baseline: 14.89 / 3.03 = 4.9×** (was 3.1×).
- **vs algebraic ordinary-binary deep-solve: median now at parity**
  (holo 5.11 vs algebraic 5.18 ms) and **p95 2.1× better** (6.08 vs 12.51).
- Profile after step 1: `classify_cells` beyond `radial_events` drops from
  ~2.1 ms to ~0.2 ms. `radial_events` (0.58–3.78 ms) is now the dominant
  term on hard cases; within it the D14 two-stage solve is 0.44 ms (plan15)
  to 2.83 ms (caustic-cross).

Evidence: `evidence/holonomic/m8_step1_three_way.txt`.
