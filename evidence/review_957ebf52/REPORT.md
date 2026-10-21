# Review and validation of 957ebf52

## Conclusion

The implementation is consistent with the inspected COM translation identity, physical angular-root recovery, fold-boundary handoff and candidate-export contract. No additional blocking implementation defect was established in this review. This is bounded implementation/regression acceptance, not a global gradient-accuracy or HMC/posterior certificate. Production solver code was not modified. Two regression-test files were corrected/extended.

## Review scope

- COM transport: x_primary=x_COM+(1-m0)*a gives D_m0=partial_m0-a*partial_image_x and D_a=partial_a+(1-m0)*partial_image_x. The signs in tangent/phi.hpp agree. The integral translation identity requires complete supported integration; it is not a coverage certificate. Endpoint and angular-node derivatives use the transported chart, and radial material motion is disabled while it is active. Thread-local scopes restore caller state.
- Physical roots: extended coefficients and binary128 physical crossing counts avoid treating rounded-coefficient Sturm counts as physical certification. Root recovery retains residual/bracket checks. Radial witnesses lie strictly inside isolated cells and avoid the lens pole.
- Recovery: detected fold pairs are passed through radial retries and outer reseeding. All active panels contribute to the retained candidate; a failed panel is not silently a zero domain. Extra replay node work is counted. Fixed replay keeps unbounded derivative error estimates; it is not derivative-tolerance adaptation.
- Metadata: unresolved frontier/candidate paths retain invalid/uncertified reasons; a missing positive finite value or finite derivative candidate raises instead of exporting initialized zeros. gradient_valid is not an independent derivative error bound.

## Test-contract finding and correction

The original complete run had 576 passed, 3 skipped and 1 failure. The binary smooth-route test reached s=1.0969276945030313, q=0.00010573716122179744, rho=0.015630246022285084, x=0.00037639405005819387, y=0.0002088807679604192. The verified previous b38a51a6 native returned mu=0 and five zero derivatives with node_evaluations=0. Its old test passed because non-smooth integral results were never checked. 957ebf52 correctly throws when no candidate exists. This was not evidence that the new COM derivative formula broke this value.
The smooth-route property now distinguishes that specific explicit refusal from acceptance of a smooth approximation, still requires >500 smooth samples, and asserts every returned magnification is finite and positive. All other errors still fail the test. Two captured-case tests prohibit initialized-zero export in value and tangent calls while allowing a future genuinely computed candidate. The new tests fail against the pinned previous native (2 expected negative-control failures) and pass on the reviewed native. The caustic-clearance criterion, tolerances and solver exceptions were not weakened.

## Final checks

- C++ unit_core passed. Updated public/JAX and smooth-route modules:64 passed. Full final regression:579 passed, 3 skipped.
- Independent VBM uniform disks plus repeated-precision scipy LD annulus integrals:349 conditions; max normalized value error 1.991346e-05 <1e-3. Tight lcbinint values:max 2.7904028e-11 <5e-7.
- Physical14-parameter directional finite differences:153 epochs,4 LD settings,3 directions and3 steps; max best-step relative difference 6.5751489e-05 <2e-3.
- Supplemental q/s checks at3 thin-image epochs and2 LD settings:8 comparisons have converged same-engine references and pass2e-3. At the remaining epoch the stricter same-engine value references did not converge, so their noisy stencils were not accepted. Independent VBM uniform-source references at that epoch pass for q and s with repeated reference precision.

## Remaining limits

The supplemental extreme-epoch LD independent derivative reference did not finish within its bounded final180-second attempt and emitted roundoff warnings. Its central value was stable across reference precision, but no unfinished LD derivative comparison was accepted. Raw failed/unconverged attempts and timeout logs are retained. This does not establish a new gradient defect; it also does not certify that additional LD q/s derivative condition. Existing captured thin-image/LD regressions pass, but no all-parameter-domain accuracy claim follows.
No MCMC, posterior convergence, sampler acceptance or new timing/speed claim was tested. The historical0.55-second result must not be presented as a measurement of this commit. Build and installed/default packages were not switched. Unrelated dirty Python/JAX changes were excluded by exporting the committed source; they were not reviewed here.

## Provenance and review mode

Source:git archive of957ebf52 build/test dependencies, with only the two documented test changes. The native hash is adf5413ca8e3e158552e6a65bce41fc6eb2c5b1c9a635ca7202c8f04c1eb32ac. All non-test source blobs were checked against the commit after validation. Source hashes, build config/logs, test logs, numerical JSONs and the exact test patch are stored here. No new Git worktree was created.
The requested Claude companion review was rejected by automatic approval because it would send branch source to an external Claude API without explicit disclosure authorization. It was not retried by another route. The code review above is local; numerical independent checks use VBM.

## Raw evidence

- full-tests.log:original full-run contract failure; baseline-smooth.log and old-point.log:previous false acceptance of initialized zeros.
- contract-negative-old.log:new tests reject previous behavior; targeted-fixed.log and full-final.log:updated tests.
- accuracy-gate.json,gradient-fd.json,thin-qs-fd.json,thin-vbm-fd-progress.json:all accepted and inconclusive numerical comparisons.
- thin-reference-probe.log,thin-vbm-single-fd.log,thin-annulus-fd.log,thin-annulus-scaled.log:reference failures/roundoff and bounded incomplete attempts.
- test-contract-fix.patch:only original-repository changes made by this review.
