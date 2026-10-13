# Hot-path bulk pass v16

Existing experimental checkout; master seed provider retained. Three reductions accepted together:

1. Reuse each radial node's square-free Sturm root-count certificate for transport discriminant sign (0/4 real roots: positive, 2: negative). Unknown counts retain the old calculation.
2. For a complete certified set of simple angular roots, evaluate the lens-map sign once in the widest interval and alternate it across roots. Unknown/degenerate cases retain the full interval scan. The audit binary recomputes every legacy interval sign and aborts on disagreement.
3. Count-only frontier probes build quartic coefficients directly in long double, rather than constructing unused derivative jets. Reciprocal-chart coefficient transformation is retained.

The trial to relax transport corrector stopping from 1e-13 to 1e-10 gave no measured gain and was reverted. No convergence gate or integration tolerance was changed in this pass.

## Matched normal comparison

87 conditions, relative tolerance 1e-4, same inputs and master seed provider. Each input has five repetitions; reported numbers are means of per-input medians. One thread; all compilation completed before serial measurements. New workspaces per repetition. Total below includes master seed preparation, boundary preparation, and integration, and excludes caustic construction. It is not total application wall time.

| Stage (ms) | v15 | v16 | Reduction |
|---|---:|---:|---:|
| Seed | 0.10057 | 0.10052 | unchanged |
| Boundary preparation | 0.12618 | 0.11080 | 12.2% |
| Integration | 0.30436 | 0.29915 | 1.7% |
| Total excluding caustic construction | 0.53432 | 0.51361 | 3.9% |

Reverse order repeats: preparation 12.5%, integration 0.6%, total 3.4% shorter. Integration improvements are small; the consistent bulk gain is preparation. Per-stage medians need not sum exactly to total medians. `sign.tsv` isolates changes 1+2; adding coefficient-only probes changes preparation chiefly. Do not claim a statistically established standalone benefit for each small integration reduction.

## Validation

- Shared adaptive tests: 1608 checks, zero failures.
- Frontier tests: 23 checks, zero failures.
- Audited normal/fold/deeper runs: 896 conditions, no interval-sign or nearest-anchor mismatch.
- Full matched before/after comparison: 896 conditions, unchanged node counts, maximum relative magnification change 1.1945e-13.
- D14 resolved 754 conditions: zero candidate convergence failures, every relative discrepancy below requested tolerance (maximum discrepancy/tolerance 0.249).
- D14 unresolved 142 conditions: excluded from reference acceptance. Before/after consistency does not establish independent physical correctness for these conditions or universal component completeness.

## Reproduction and artifacts

Run scripts from `/rogue1_8/nunota/lcbinint/experiments/seeded-boundary-events`. Build the current candidate with `benchmarks/holonomic/build_seeded_radial_master.sh`. Compile the same source with `HOLO_DISABLE_COUNT_REUSE`, `HOLO_DISABLE_ARC_SIGN_REUSE`, and `HOLO_FRONTIER_LEGACY_PROBE_COEFFICIENTS` to reconstruct the pre-v16 paths; saved baseline binary `build-seeded-events/bench_seeded_before_v16` was copied before the trials. `build_audit.sh` builds the audit variant. Keep compilation and accepted timing runs separate.

Benchmark arguments: INPUT OUTPUT REPS 0; use `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`. Inputs are `evidence/holonomic/caustic_seeded_connection_v4/{initial,fold,deeper}_minimal_seeds.tsv` (stored seeds ignored; live master seeds generated). Normal repetitions 5, holdout 3, audit 1. Reverse order uses the same normal input.

`summarize.py` creates `summary.json`, source snapshots, and a provenance/hash manifest from raw TSV files. `trials.md` records the rejected trial. No commit, publication, or change to the shared master checkout.
