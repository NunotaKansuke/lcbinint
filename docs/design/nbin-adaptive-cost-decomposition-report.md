# Adaptive work and error-driver decomposition

## Conclusion

On the current default experimental adaptive route (interior grouping disabled), tightening the error estimator alone does not explain most of the remaining cost at absolute target 1e-4. Reference-informed extra refinement accounts for roughly 15–18% of refinement time, or 10–15% of the whole scalar call. The dominant estimator direction differs strongly by brightness profile: uniform is driven by vertical row integration; pure linear LD is driven by horizontal Bennett endpoint residuals. The current code refines both lattice directions and reprocesses a whole component for either driver.

These are measured **estimator drivers and work associations**, not an independent localization of the actual physical error. In particular, the existing model is not a complete certified LD error model.

## Main measurements

74 geometry/profile combinations, both absolute targets, initial nbin16, max512, default map budget. Profile counts are 27 uniform, 25 linear-LD c=0.5, 22 pure-LD c=1; the profile sets are not identical, so cross-profile comparisons are descriptive. All 148 final calls met their estimated targets and their independent saved VBM/radial-LD reference errors. Every instrumented call matched the uninstrumented magnification, error indicator, lattice counts, history, components, and acceptance result exactly.

At target 1e-4:

| Profile | Refinement share of full call | Extra refinement time after stable truth pass | Same extra time / full call | Dominant estimator direction by trial time |
|---|---:|---:|---:|---|
| uniform | 83.4% | 18.1% | 15.1% | vertical signed residual: 99.1% |
| LD c=0.5 | 69.1% | 14.6% | 10.1% | vertical: 69.9%; horizontal: 27.8% |
| LD c=1 | 74.1% | 18.2% | 13.5% | horizontal signed residual: 95.7% |

Additional adopted steps after the stable truth pass occurred in 8/27 uniform, 6/25 c=0.5, and 5/22 c=1 cases. Their additional new maps were 18.6%, 14.1%, and 14.1% of all new maps, respectively. Tiny-component safety guards caused many steps (27, 27, 22) but only approximately 0.3%, 0.8%, 0.8% of refinement time at this target. Removing those guards is therefore not the main speed opportunity.

The retained run cells at adopted finer stages were approximately 94.6%, 92.4%, 92.6% farther than two horizontal cells from a run endpoint. This is a **geometric scan-volume proxy**. It does not mean those percentages of wall time are wasted, and LD interior brightness cannot simply be ignored. It does show why whole-component refinement covers much more space than the endpoint stencil.

At target 1e-3:

| Profile | Refinement share of full call | Extra refinement time after stable truth pass |
|---|---:|---:|
| uniform | 19.3% | 3.4% |
| LD c=0.5 | 10.6% | 56.0% |
| LD c=1 | 21.5% | 38.3% |

Even large relative excess within refinement contributes only about 6–8% of total LD call time at the looser target. Initial seed/fill and other call work dominate the remainder; this audit does not separate those stages. The short-component guard dominates summed indicator magnitudes at some coarse truth-passing stages but its steps are cheap; large numerical indicator magnitude alone is a poor optimization priority.

## Definition of reference-informed excess

Save the magnification at each actually adopted stage, then compare it to independent truth **after the adaptive calculation**. The earliest stage whose error and all subsequent adopted-stage errors meet tolerance is called the stable truth-passing stage. Refinement after this stage measures retrospective excess relative to the reference. This is not an operational stopping criterion or a guaranteed removable cost.

A first truth-passing stage can be transient: at 1e-4 this happened in 12/27 uniform, 10/25 c=0.5, and 7/22 c=1 cases. Taking the first accidental pass would exaggerate the removable work. No magnification-difference convergence rule or truth-assisted acceptance was added to the solver.

## Instrumentation and timing

`LCBININT_DIAGNOSTIC_ADAPTIVE_COST_TRACE=1` adds snapshots of raw component flux and separated inner/outer signed residuals, inner/outer remainders, the extra short-component guard, row counts, run-cell counts, and endpoint-stencil cell counts. Each refinement trial records elapsed time, map/cache counts, source/target nbin, and adoption status. The history gains per-stage magnification.

A driver's label is the largest magnitude among those five terms before within-component cancellation. Trial time is associated with that label; it is not a measured time spent evaluating the term. Signed magnitudes do not add to the accepted error indicator because cancellation within a component is retained.

Trial timing starts immediately before recursive fill and includes trial aggregation and footprint checking. It excludes seed lifting and initial seed/fill work. Full-call timing includes native-to-Python result construction. Timed calls change geometry every time: adjacent repeated geometry count is zero. The baseline and trace matrices run separately, avoiding immediate repeated-geometry warming. Five alternating full matrices measured trace overhead at 1.00139 times baseline (about 0.14%). v1 was retained but its immediate baseline/trace repetition warmed the seed cache; use v2 for full-call fractions.

`unit_core` and `unit_boundary_support` passed 2/2 on the instrumented build. No acceptance thresholds, estimator coefficients, seed routing or refinement policies were changed. General certification remains false.

## Next design justified by the measurement

Prioritize **directional local refinement of boundary/run-width information**, rather than another broad attempt to interpolate LD interior blocks:

1. Uniform and much of c=0.5: add vertical boundary/row-width samples without simultaneously doubling horizontal sampling across the image interior.
2. Pure LD: improve/refine the horizontal limb endpoint contribution without increasing vertical resolution of the whole component.
3. Retain current contributions and stencils. Add only the selected direction/patch, keeping a minimal local fallback for unresolved geometry.

This requires coherent unequal spacings and Bennett weights; the existing kernel assumes a single spacing. This audit implements the measurement, not that new integration scheme. Additional speed from error-model improvements remains possible, but the measured retrospective excess cannot by itself close the previous roughly threefold gap to reference-informed fixed nbin selection.

## Artifacts and reproduction

- `tests/diagnostics/recal2026/adaptive_cost_decomposition.py`
- `tests/diagnostics/results/recal2026/adaptive-cost-decomposition-v2/traces.json`: all snapshots and independent-reference stage errors.
- `summary.json`: work/time aggregates and instrumentation overhead.
- `drivers.json`: trial driver labels and geometric scan-volume aggregates.
- `adaptive-cost-decomposition-v1/`: build and 2/2 ctest logs; exploratory immediate-repeat timing retained.

Run after building with project-local TMPDIR:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LCBININT_PROBE_BUILD=$PWD/build-boundary-support /home/nunota/.miniconda3/envs/myenv/bin/python -m tests.diagnostics.recal2026.adaptive_cost_decomposition
```

Source and native hashes are recorded in v2 provenance. The fixed output directory is for this bounded audit; preserve it or choose a new directory before another run.
