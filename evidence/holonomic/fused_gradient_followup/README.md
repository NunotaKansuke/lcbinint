# Bounded derivative diagnosis, LD sharing, optional compiled API

This follow-up starts at commit 81366763. The requested scope is modest:
diagnose obvious gradient discrepancies, remove redundant LD work, and
provide an optional API without changing the normal integration route.
Gradient accuracy certification remains out of scope.

## Findings

The existing three-step finite-difference probes separate at least two
effects. Some small derivative discrepancies shrink approximately with
the squared difference step (e.g. case 38, uniform, q), so the smallest
saved difference is still a biased reference. In case 155, uniform, the
stable finite-difference result and the previous strict quadrature agree
much better than the value-mesh result. An eight-input resolution probe
using only the existing mu_rtol setting confirms the latter discrepancy
shrinks when the value mesh is refined. This supports quadrature resolution
as the main cause there; it does not prove every gradient formula correct.

Case 155 uniform, y derivative: the old default is -1.1462385, a diagnostic
h-squared extrapolation of converged finite differences gives -1.1386220,
and mu_rtol=1e-6 gives -1.1386485. The lens-separation derivative similarly
changes from -2.0414988 to -2.0650644, versus approximately -2.0651288 from
the finite differences. No new solver rule, default tolerance change, or
derivative refinement was introduced from this probe.

Case 9 linear LD remains inconclusive: the base value and/or perturbed
reference values do not converge. Its derivatives must not be treated as
accurate merely because numbers are finite. Several other LD references
also fail their value tolerance. These cases were retained in the timing
and regression tables; no accuracy threshold was relaxed.

## Implementation and checks

LD parameter derivatives used five scalar solves sharing the same root-pair
2x2 Jacobian. The new helper builds that Jacobian/determinant once and solves
the same five right-hand sides with the same singularity guard. A 45-case
algebra check covers separated/coalescing roots, matches scalar solves,
checks their implicit equations, and checks singular-pair rejection.

The public `epoch_fused_gradient` entry point is declared in
`src/lcbinint/magnification/holonomic/fused_gradient.hpp` and linked through
an optional static archive built by the existing experiment script. It
accepts LensParams, the LD coefficient, prepared primary-frame image seeds,
and the two existing value tolerances. It returns magnification, five
derivatives, value status, component validity and the existing coverage
flag. `gradient_precision_checked` is false. Component validity is not a
derivative error bound.

The compiled implementation has a private namespace to avoid mixing two
header definitions in one program. Normal value and gradient API are tested
in one translation unit. Independent single-lens-limit position/radius
checks pass for uniform and LD; empty-seed failure flags pass. No existing
src header, normal CMake target, primal settings or acceptance rule changed.

## Matched timings and regression

`api_bench` ran all 7,216 previous inputs with three repetitions, fresh
seed/index preparation included and caustic prebuild excluded. Value/API
order alternates; they share seeds and use independent workspaces. All
normal value results exactly match the previous build. The fused primal,
stopping status and node count also match the previous fused implementation
exactly for every input.

| Inputs | Median preparation-inclusive ratio | Ratio of summed medians |
|---|---:|---:|
| All | 1.250 | 1.230 |
| Uniform | 1.192 | 1.154 |
| Linear LD | 1.328 | 1.297 |

This is a small measured improvement from the previous aggregate ratios
(all 1.239, LD 1.314), and run-to-run timing variation contributes. Do not
attribute the whole difference solely to Jacobian sharing. Excluding seed
preparation, the aggregate solver ratio is 1.343 (LD 1.432). 2,623 inputs
still exceed 1.3x including preparation. The goal is not met universally.

Tangents are not bitwise identical to the previous separately compiled
experiment. Maximum |difference|/max(1,|old gradient|) is 2.475e-4, found
in weak/sensitive LD separation derivatives. The primal is unchanged;
algebra and analytic-limit checks pass. This comparison is a regression
observation, not an independent accuracy bound.

Reproduction from this checkout:

```bash
bash benchmarks/holonomic/fused_gradient/build.sh
build-seeded-events/fused-gradient/api_check
build-seeded-events/fused-gradient/batch_check
build-seeded-events/fused-gradient/api_bench evidence/holonomic/seeded_paper_qrho_20261014/input_snapshot.tsv evidence/holonomic/fused_gradient_followup/full_api.tsv 3 1
python benchmarks/holonomic/fused_gradient/followup_audit.py
```

`resolution_inputs.tsv` has only eight selected difficult geometries/profiles.
The two resolution tables use one repetition, so their timings are not a
performance claim. Full tables, checks, audit, and implementation/backend
hashes are retained here. Prior finite-difference evidence remains in
`fused_gradient_20261014`.
