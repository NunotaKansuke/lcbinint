# LD cost reduction, 2026-10-28

Baseline is the prior validated working-tree build, SHA256 c1c904ecb28693c2815228301a39101152896f75ef8b16fd6b38e81fbad88354, on experiment/algebraic_boundary at HEAD 2afd056a9fad679380a5a7f1b832732e39558039. The baseline includes earlier uncommitted optimization work; this comparison is not against pristine HEAD.

Accepted algorithms:

- Scalar LD keeps the existing arithmetic and endpoint-map uncertainty bounds. A point no longer needs its own relative error below the whole requested tolerance. For loose local requests (sample_relative_tolerance >= 1e-6), when e/phi <= 1/4, sqrt(phi)*e/phi conservatively bounds the square-root arithmetic error; sum the existing positive-weight absolute bounds into the unchanged global value controller. Locally ill-conditioned points retain quad fallback.
- Gradient LD shares sine/cosine across mirrored nested angular nodes, retaining both physical units in the node packets. Promoted points retain the incumbent precise map. All eight RHS use a single set of scalar contraction factors; affine endpoint motion is formed once per arc.
- Before solving the coordinate-hint stationary cubic, a padded Taylor bound excludes a quartic maximum when its curvature is positive throughout the interval or its derivative cannot vanish. Existing pole-distance and physical consistency guards remain. This does not alter physical support/completeness proofs.

Rejected trials:

- Eager scalar deflation: construction overhead makes main LD conditions slower.
- Late nonconcave ordinary-moment fallback: median gradient times around 4% worse across the twelve disks.
- Mirrored trigonometry in scalar LD: no meaningful speedup; retained only in the gradient-owned route.
- Cached scalar reciprocal: no robust measured benefit.
- Removing node-level angular-map uncertainty: scalar 168-condition checks and coarse-boundary regression passed, but a complete error argument for mixed ordinary/promoted coordinate maps was not established. Keep incumbent conservative map bound. Domain-budget binaries/results remain as rejected evidence.

Initial paired-matrix trials are exploratory. The concurrently launched budget-paired/hint-paired timings are not used for any performance claim; isolated repeats and final paired measurements are the evidence.

No derivative tolerances are relaxed. Acceptance remains value-only; offline derivative gate is 1e-4+2e-4*abs(reference), including independent reference uncertainty. The 30% derivative overhead target is not asserted achieved. Saved VBM tight references use the identical installed VBM 5.5 binary (hash verified), and are cross-engine diagnostics rather than certified truth.

## Tight-request regression

The unrestricted absolute-budget trial stalled at rho=.01, rtol=1e-9 for the source obtained by mapping z=.9+.1i through the lens equation (the baseline converged). The final code therefore preserves the incumbent early local promotion for sample_relative_tolerance < 1e-6. Three previously converged small-source/tight requests again converge, with the same quad counts as baseline (168,250,192). test_regular.cpp now includes the two smallest/strictest converged requests. The older rho=.001, rtol=1e-9 request already failed in baseline and is retained in tight.json; it is outside this performance population and is not claimed solved.

The unrestricted-* timings/binaries are retained only as intermediate evidence. Final performance must use the final guarded source and binary hashes.
