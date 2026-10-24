# Physical topology and transverse-gradient recovery review

Reviewed baseline: 7cb4e0fbd9b092dc4149cd987f5f6c4faf0b2cec, branch experiment/algebraic_boundary.
The eight scoped source/test files exactly match the existing validated/source.patch:
SHA256 b7a222253318361208bafc39ba723bdb08a70f867f21883dea3e3692d4afabee.

Existing validation artifacts (not rerun during this review):
/extern/moao38_7/nunota/binfit/work/MB23169/runs/transverse_gradient_repair/validated/
Its public.log records 214 passed. The parent README documents C++ success,
five all-component finite-difference checks, retained uncertified diagnostics,
and matched ordinary-target timing/parity. Same-engine finite differences are
not an independent solver reference or a global derivative-accuracy certificate.
The saved timing comparison is against the preceding physical-topology build,
not a fresh measurement of this commit. No sampler or default package changed.

Local inspection found no blocking defect in the scoped changes. Physical arc
reconstruction is limited to rejected samples and retries only once. Value-only
reconstruction retains the value template; parameter derivatives are not enabled.
Primary-frame edge handling follows the existing chart predicate. Gaussian
panels with distinct real pairs at both ends split once; node work is counted.
Failed edge replay retains finite incumbent output and invalid gradient flags.
Completed edge replay avoids the older redundant contact retry.

This review ran no tests, builds, benchmarks or sampling. Unrelated preexisting
Python/JAX changes are excluded. Scientific acceptance remains false; no universal
accuracy or zero-overhead claim is made.
