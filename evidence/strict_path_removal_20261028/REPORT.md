# Remove derivative-error contracts from algebraic binary and triple integration

Baseline: branch `experiment/algebraic_boundary`, HEAD `59be25417ba30a2277e40dfadb32adf8fd06f670`, plus the preceding source-disk removal changes. Baseline triple/binary C ABI libraries are `../source_disk_removal_20261028/triple-final.so` and `binary-final.so`; default-path results and hashes from that step are preserved. Existing unrelated JAX/backend work remains untouched.

## Production contract

Only value tolerances select/refine quadrature. Analytic derivatives travel with values on the retained integration packets. Finite derivative checks and physical support/topology validity remain. Tight value requests still receive the existing local geometry/arithmetic precision repair.

Removed triple policy selector, strict derivative convergence and component scheduling, derivative-error packet fields, normalized derivative-error propagation, per-component error estimators, and precision/reliability output flags. The public triple `gradient_policy` keyword is rejected. Informational `gradient_policy` output remains `value_mesh` for gradients and `none` for scalar calls. `gradient_valid` expresses finite computed derivatives with complete support and converged values.

Removed binary internal strict policy, derivative tolerances, derivative-error budgets, post-value derivative refinement/adaptive replay, error/contract diagnostic counters, derivative error estimates, tolerance-met quality labels and the always-false public precision flag. Both binary error ledgers and optional angular-cache error storage have one scalar component; derivatives retain their five value components. Fixed replay and geometry/topology repairs remain. Internal quality is now finite-valid/invalid/not-requested, without a derivative accuracy claim.

The independent scientific test comparisons still assess derivative correctness offline. They do not introduce runtime derivative error estimation or a derivative tolerance API.

## Verification

- Final Release build completed, with at most two compilers and project-local temporary files. `ctest.log`: 30/30 pass. Public scalar helper regression: 1/1 pass through the rebuilt native module.
- Triple: 240/240 independent canonical derivative components and pure-LD identities pass the unchanged offline gate (`triple-independent.json`); 88/88 physical public-API components pass (`native-physical.json`).
- Binary: 100/100 smooth-source FD components, 20/20 contact components, 20/20 former source-repair components pass (`binary-fd.json` excluding its initially unresolved contact-reference group, `binary-contact-fd.json`, `binary-used-fd.json`). These are independent five-point differences with two-step reference checks; the offline gate remains `1e-4 + 2e-4*abs(reference)`, including recorded step uncertainty.
- Binary before/after: 203 configurations, scalar and gradient (406 calls). All 405 previously value-converged outputs, including validity flags and all derivatives, remain bitwise identical. The only changed call is the negative extreme-contact gradient at relative value tolerance `1e-7`: it now converges instead of failing. Its value differs from the existing scalar route by only about `7.9e-13` relative.
- The newly converged negative contact also passes five independent derivative comparisons with converged scalar references (`negative-contact-resolved-fd.json`). An initial `1e-9` reference did not converge; that inconclusive run remains saved in `negative-contact-fd.json`. The resolved reference uses converged `1e-7` scalar values and wider two-step stencils, retaining the unchanged derivative comparison criterion. No production tolerance was relaxed.
- Triple before/after on 48 naturally integrated conditions, scalar and default gradient (96 calls): 88 outputs are bitwise identical, excluding timing fields. The remaining eight keep exactly the same success flags, quadrature work and refinement counts; maximum flux difference is `1.78e-15`, maximum derivative difference `7.24e-14`, maximum scalar error-estimate difference `3.69e-13`. Independent accuracy checks pass. Initial bitwise-only comparison failures are retained in `parity.log`; `parity-complete.log` checks numerical parity and explicitly checks the one improved binary failure.
- Static scan of production `src`/C++ bindings finds no derivative error fields, derivative tolerance fields, strict derivative policy or precision-check outputs. Public tests reject the removed selector and check absence of the removed fields.

Initial test failures are retained. The radius/panel tests formerly assumed identical scalar/gradient quadrature; they now require numerical agreement at approximately `1e-12`, while keeping their independent finite-difference checks unchanged. Scalar endpoint promotion is tested separately from actual fused derivative packet retention, using fresh-versus-incremental work counts and exact repeated-packet identity. Production value tolerances and the offline derivative acceptance criterion are unchanged.

## Artifacts and scope

`changes.patch` and `artifact-manifest.json` capture task-owned sources and final library hashes. Baseline helper sources and native module are preserved locally. `check_parity.py`, `triple-kernel.cpp`, `check_negative_resolved_fd.py` and raw JSON/logs record the checks. The preceding source-disk-removal evidence supplies the six representative binary fixtures, the 64 random geometries and archived contact configurations; all matched baseline inputs are retained.

No new speed claim is made for this deletion. Historical source-disk helpers, strict-policy evidence and independent FD references remain as archived evidence, not callable production routes. The working-tree changes are not committed.
