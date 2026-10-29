# LD radial resolution and critical-arc transport

Base: experiment/algebraic_boundary, 53335b9a. This is a continuation of the complete-component and rho-tangent closeout, not a replacement of its VBM/independent-reference evidence.

Two separate effects were reproduced at value rtol=1e-3, atol=0:

1. Case 940 central alpha/ang gradients: a scalar-flux convergence test could accept a radially underresolved LD chart. An unweighted LD fixed Patterson floor of 31 nodes fixes the tighter-scalar-FD counterexample. Uniform and weighted LD floors retain 7 and 15 nodes. No derivative estimate controls adaptation.
2. Case 1070 small ang gradients: replacing ordinary moments does not fix their periodic/epoch-shift inconsistency, while bypassing critical-arc coefficient transport does. At the SAME critical sinh-chart nodes, computing the physical Phi parameter jet instead of contracting coefficient/pair sensitivities removes the discrepancy. This isolates numerical cancellation in critical polynomial tangent transport; a fully physical angular oracle corroborates the corrected result. This is observed on the saved counterexamples, not a universal proof of all numerical conditioning.

The adopted candidate retains the factorized scalar Phi evaluation to avoid subtracting nearly equal physical source/radius terms at angular ends. It evaluates the physical parameter jet at each quadrature node and carries all eight canonical tangents, including radial chart motion, together with the factorized flux. When the scalar sample roundoff exceeds the existing local arithmetic fraction 1e-5 of the factorized Phi, local binary128 jet evaluation is used. This is an arithmetic guard, not a derivative-accuracy controller. The physical fixed-angle derivative of the vanishing LD intensity needs no physical-end boundary term; internal chart boundaries cancel.

The former coefficient contraction remains for the optional material-map compile modes. The production fixed-physical-angle mode skips it. No image, parameter RHS, or existing root validation is omitted. No accepted value integration is replayed, and there is no source-disk production shortcut.

The all-physical oracle is diagnostic: some raw canonical integrations exhaust the refinement budget, and it must not be promoted to an unconditionally accepted backend. The public API's normal frame retries remain active. All production validation must use the frozen public LightCurve package and record its hash, rather than classifying a raw prototype as a production API result.

The intermediate b0c3b3be candidate tried an additional per-cell refinement floor and a stricter 1e-9 promotion guard. It passed 3711/3712 original native jobs but case 788 exhausted the 60 CPU-second worker cap; it was rejected. The final candidate removes that extra per-cell controller change and reuses the existing 1e-5 local arithmetic guard. Case 788 then returns successfully in approximately 1.8 seconds. The final full validation must be distinguished from the rejected candidate's data.

C++ header-only prototype results after rebuilding the native static libraries may be affected by weak inline/template symbol selection at link time. Treat these probes as diagnostic pointers. Claims and acceptance use the frozen complete native build, public LightCurve workers, independent references and regression tests. Do not substitute header overlays for final build identity.
