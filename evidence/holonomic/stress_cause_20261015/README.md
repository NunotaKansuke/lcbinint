# Follow-up cause investigation — 2026-10-15

Existing experiment/algebraic_boundary at HEAD 1d6984f7501066f22ff0143d69c1cd1c45b04d06 plus preserved prior changes. Production source/native modules unchanged in this investigation. All new instrumentation is in the isolated project-local build-algebraic-boundary/stress-cause build. No notebooks edited. All diagnostics were monitored to completion.

## Small-source LD gradient error: localization

case_033, t=-.15, inputs from ../stress_20261015. Baseline diagnostic reproduces the public API gradient exactly: LD=.5 dmu/drho=-150.42007418691975 at tol=reltol=1e-3.

The accepted mesh has three depth-zero radial panels, each level 3. Their rho derivatives are approximately +2.94e-7, -150.48708, +0.0670056. Thus one ordinary unsplit image span, R in [1.2528025255063888,1.2546590966293123], dominates the error. This is not a close-fold replay case.

Uniform brightness under the same settings gives dmu/drho=1.22158, close to a source-plane reference. LD=.9 gives -337.40368; the error grows with the LD contribution. Tightening value tolerances substantially reduces the LD derivative error.

### Controlled exclusions

- Force every LD angular integral through 31 nodes: dmu/drho=-150.42007418682223. The result barely changes. Increasing this angular quadrature order does not address this counterexample.
- HOLO_HOLONOMIC_TRANSPORT=0/1: identical output in this example. The transport switch does not affect the reproducer; this is not evidence that all root initialization mechanisms are excluded.
- Previous root-pair long-double-only experiment did not remove the error.
- Initial radial level 6 (63 nodes) does change the derivative, but some diagnostic runs do not complete the value refinement and leave deferred derivative components unevaluated. Those outputs must not be accepted as improved gradients. This experiment is diagnostic only and not promoted.

### Independent reference

source_plane.py integrates the source plane directly using only VBM BinaryMag0, not any finite-source function. Disk coordinate z=sqrt(1-r^2) makes the linear brightness polynomial. Uses radial GL8/GL16, angular 32/64 nodes, and rho differences of 5%, 2%, 1%. VBM point coordinates are the negatives of the native source coordinates; correcting this is essential. Center point magnification is 66.66893117057924, consistent with the native small-source limit.

For LD=.5 the resulting rho derivative is about +1.1235. The exact sequences are in source_plane.json and agree with the previous strict same-engine finite differences. Numerical stability is observed across node counts and steps; this is not a formal interval certificate.

### Interpretation and next discriminating check

Evidence localizes the failure to the LD derivative on a coarse, value-accepted radial panel. It does not yet isolate a unique erroneous arithmetic operation. The rho derivative combines terms on the scale 2*mu/rho ~= 8.09e6 to obtain a physical answer near 1.12. A small residual integration error can therefore be hugely amplified. Simply raising precision in the root-pair 2x2 solve or raising angular order is insufficient.

Next check: evaluate fixed radial nodes with independently differentiated moving endpoints and compare node-level material derivatives. Distinguish an inaccurate derivative integrand from an inadequate radial quadrature/control variate. Do not introduce derivative-error adaptivity or a new tolerance API. A source-coordinate or analytically subtracted leading image contribution is a candidate for reducing cancellation, not an implemented fix.

## NaN: exact failure path localized

For case_004 and case_049 at t=0, value integration finishes, then the close-root detector sets geometry_complete=false. The post-integration guard intentionally invalidates all gradients. It is not a scalar-value NaN or a failed value integral.

Independent detector-only reproduction for case_004:

- Primary-frame X=0.50000049999999996, Y=0.
- Occupied band [0.99797790808638764,1.0020275584743472].
- Bernstein isolation finds uncertain coefficient signs.
- Quad-interval Sturm chain construction returns false.
- The detector reports incomplete; derivative guard returns NaNs.

case_049 similarly fails in [0.99984035281159112,1.0001596825225434], with Y=0. Disabling the fast screen does not cure either example. isolate.cpp reproduces the detector independently of flux integration. A general failure of all symmetric sources is NOT claimed.

The axis symmetry is algebraically relevant: at Y=0 the angular quartic has no odd coefficients and its discriminant contains a squared factor. The structured D14 expression also simplifies to (C^2-4*v*G)*G^2 when its Y-dependent Z vanishes. Interval root isolation of the expanded derivative can lose signs under these clustered/symmetric factors. symmetry.py records exact symbolic factorization; the precise numerical reason for each Sturm-chain pivot failure remains to be isolated.

A safe fix should exploit the factor structure and known endpoint information or strengthen certified isolation. Clearing the incomplete flag or returning unchecked finite derivatives would hide the failure.

## Artifacts

baseline/ contains untouched pre-intervention panel logs. angular31.log records the forced-angular trial. radial63.log is diagnostic and includes incomplete refinements; do not use it as an accuracy result. material_fd.json records a rejected coarse-mesh finite-difference check: changing mesh and coarse primal errors make those differences unstable. Its per-panel differences are not meaningful when the panel lists differ, so they are not evidence for a derivative formula error. source_plane.json contains the independent reference. isolate4.log contains detector failures. prepare.py/build.sh/probe.cpp/run.py reproduce the baseline diagnostic. For the radial trial use build63.sh. Default builds and the prior speed improvement remain intact.
