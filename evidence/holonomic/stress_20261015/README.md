# Broad stress exploration — 2026-10-15

## Scope and reproduction

Existing experiment/algebraic_boundary checkout, HEAD 1d6984f7501066f22ff0143d69c1cd1c45b04d06 plus preserved dirty angular-alias fix and per-probe continuation speed improvement. No notebook edits or new worktrees. Native hashes and input seed recorded in provenance.json. Use the project Python environment with numpy, scipy, VBMicrolensing 5.5. All artifacts remain here.

Run `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python evidence/holonomic/stress_20261015/sweep.py` from this checkout. Existing result JSONs are skipped. Each worker is sequential, pinned to one allowed CPU, limited to 20 seconds. Failed workers are preserved, not silently discarded. sweep.py was subsequently amended to save native values before a potentially slow VBM call; original runs only saved complete workers. This is a cold-call discovery screen, not a controlled repeated speed benchmark.

120 deterministic configurations: s roughly .1–10, q 1e-7–1, rho 1e-5–2, random angles, zero/near-zero/nonzero impact parameters, exact/near-resonant separations, LD coefficients 0/.5/.9/.999999. Each uses 33 uniformly spaced epochs and tol=reltol=1e-3. Includes reverse-order value checks, gradients, and VBM comparisons. cases.json contains exact inputs.

## Observations

119 complete workers, 3927/3927 finite and value-converged native magnifications. 27398/27489 gradient components finite: 13 epochs in 13 configurations each have all seven components NaN. Reverse-order maximum relative value difference 4.006e-11. Thirty additional same-object parameter-change checks exactly match newly constructed curves.

The remaining whole-worker timeout (case_054) is isolated to VBM: native value 33 epochs ~4 ms, gradient ~18 ms in a diagnostic rerun; VBM alone exceeds a separate 5-second limit. These single timings are diagnostic, not headline performance comparisons.

### Native gradient counterexamples (unresolved)

1. case_033: s=1.099687863945755, q=1.555471033863785e-6, rho=1.648889061484618e-5, u0=0, alpha=-1.2719072845402482, LD=.5, tE=10. At t=-.15, tol=reltol=1e-3 gives dmu/drho=-150.420 while strict native central differences at relative parameter steps .01/.003/.001 give approximately +1.1235. Strict value convergence flags are true and the last two finite differences change by ~2.2e-5 relatively. At positive t the derivative is +110.932 versus approximately +1.1242. Tightening the gradient value tolerance to 1e-7 reduces the discrepancy greatly, but does not eliminate it. Thus finite output is not sufficient accuracy evidence.
2. NaN examples include case_004 (equal masses, s=1.000001, u0=0), case_049 and very small-q resonant configurations. The value result reports Converged while derivatives are invalid. This status refers to value convergence and must not be interpreted as gradient acceptance. nan_probe.py reproduces near-zero-time behavior. Disabling the fast geometry screen does not cure these examples, ruling out that shortcut alone. The exact downstream geometry/degeneracy failure is not yet isolated.
3. fd.py contains 12 selected configurations × 4 parameters × 3 epochs, each with three central-difference steps and strict value tolerances. fd.json records all steps and convergence flags. Large relative differences when the true derivative is almost zero, and nonconverged finite differences, are not classified as proven large absolute errors. case_089 strict references do not converge; its gradient discrepancy remains inconclusive.

Hypothesis for case_033: the value-selected mesh and moving narrow-image derivatives leave large uncancelled quadrature contributions. This remains a hypothesis; neither precision-only nor unconditional replay fixed it reliably.

### VBM disagreement is not a native-error metric

- case_033 central LD: native 128251, VBM built-in LD 1278147. Uniform outer-disk results are about 120178 versus 120226 at tol1e-3. VBM uniform-based annuli vary 128287→128454→129568 for 32/64/128 nodes, so this reference is NOT converged. Tightening built-in VBM LD caused a long-running call, which was explicitly terminated. No certified exact answer is assigned.
- case_064 (uniform, tiny q, source edge): native 51.34879 versus VBM 27.21355. Independent single-lens polar integration gives 51.34703.
- case_089 (LD, tiny q, source edge): native 19.13460 versus VBM 16.25333; single-lens integration gives 19.13366.
- case_117: native 7.46217 versus VBM 7.40994; single-lens integration 7.46171.
- case_013 and case_118: VBM-uniform annulus integration supports native LD values, with the numerical sequences retained in annulus.json.

Single-lens comparisons are independent limiting checks, not exact binary references: q remains nonzero, and its effect can be amplified near caustics. They support suspicion of VBM problems but do not certify the native solver. Integral support flags are preserved; no certification was promoted.

## Rejected trials and final source state

- Unconditionally replay every active derivative panel with Gaussian nodes, deferring initial tangents too: some improvements but unstable case_033 derivatives (e.g. +84.19 rather than +1.1235 at tol1e-3). Rejected, sources restored.
- Evaluate only the root-pair implicit derivative 2x2 system in long double: case_033 remains about -150.420. Rejected, source restored.
- Sources, rejected trial code and ladder outputs are saved under before_* / rejected_* / *_ladder.log. Default gradient native module restored byte-for-byte from before_gradient.so; generated headers regenerated from restored sources. Previous speed improvement remains active. No new production optimization or gradient fix is claimed in this exploration.

## Next work

Prioritize case_033: inspect span-local uniform/LD derivative contributions and moving-boundary terms against a smooth single-lens/point-source limit before changing mesh rules. Separately isolate which geometry-completeness guard invalidates symmetric configurations. Keep tol/reltol as the only error controls; do not introduce gradient-driven adaptivity. Use these counterexamples as regression fixtures for future fixes.
