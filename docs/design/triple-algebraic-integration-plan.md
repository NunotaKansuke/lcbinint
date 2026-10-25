# Triple algebraic finite-source integration: design and delivery plan

## Scope and baseline

Existing checkout: `experiment/algebraic_boundary`, baseline
`50d5073e138329ea36beb5fcace504605096684a`. Preserve the pre-existing dirty JAX
work. Deliver static triple finite-source values and analytic derivatives,
uniform/linear limb darkening, a standalone physical-coordinate API and explicit
LightCurve routing. Do not change the ordinary triple backend default.

## Required numerical invariants

1. Reuse the optimized binary primitives wherever their assumptions hold:
   floating-point environment/enclosure operations, nested Fejer nodes and
   interpolation matrices, fold maps, error policy and retained-node scheduling.
   Do not copy the complete binary/tangent implementation trees.
2. Binary64 is the normal precision. Promote only an unresolved event/root box,
   event centre or sensitive local endpoint/sample. Do not restart an epoch in
   quad. Count promotions and parameter derivative work separately.
3. Preserve accepted root/exclusion decisions, canonical samples, radial
   workspaces and accepted values. Local refinement adds work to the affected
   region. Cache identities must include source/radius/geometry/settings.
4. Value-only instantiations contain no parameter-gradient storage or loops.
   Share geometry definitions/rules and differentiate the accepted integral
   analytically. Never use finite differences inside the implementation.
5. Keep coverage, flux convergence, and gradient reliability separate. A budget
   failure is explicit, with no silent change of tolerance or integration method.

## Geometry and boundary polynomial

Use the primary frame `(0,a,bx+i by)`, masses `(1-m1-m2,m1,m2)`, source `(X,Y)`.
Internal derivative order is `(X,Y,rho,m1,m2,a,bx,by)`.

At `z=R w`, `|w|=1`, define

- `D=product(R-conj(lens[j])*w)`, degree at most 2 in w;
- `N=(R*w-source)*D-w*sum(m[j]*product_except_j)`, degree at most 3;
- `H=rho^2*|D|^2-|N|^2`.

Autocorrelation constructs the real trigonometric coefficients of H. Under
`t=tan(theta/2)`, `P=(1+t^2)^3 H` has degree at most 6 in t and 8 in R.
The same scalar-generic construction supplies binary64 values, interval
coefficients and local geometric jets. Lens poles and zero-radius limits are
not physical boundary roots introduced by clearing denominators.

## Final architecture and implementation sequence

1. `triple/geometry.hpp`: scalar-generic sextic, physical phi, optional eight
   parameter derivatives, radial/angular derivatives and reciprocal chart.
2. `triple/angular_topology.hpp`: bounded two-chart Bernstein root isolation.
   Binary64 coefficient enclosures are reconstructed from original inputs;
   ambiguous boxes alone use quad. Keep accepted roots and excluded boxes.
   Construct physical arcs, join chart-seam arcs and distinguish full circles.
3. `triple/event_surface.hpp` and `radial_events.hpp`: construct an 8x6
   bivariate Bernstein boundary tensor once per chart. Subdivide/restrict its
   retained control coefficients to exclude or isolate `P=P_t=0` events with
   Krawczyk proofs. Local direct/quad refinements resolve conditioning. Use
   overlapping chart-edge proof neighbourhoods and a nonzero dyadic split to
   handle symmetry-axis roots. No D14 polynomial or sampled seed grid is used.
4. `triple/radius_terms.hpp`: uniform arc lengths and weighted linear-limb
   angular terms. Analytically differentiate angular endpoint motion and the
   moving angular map. Reuse nested rules; local angular h-repair retains its
   samples. Exact collinear/axial reflection identities avoid chasing transverse
   cancellation noise. `local_precision.hpp` repairs only sensitive geometry.
5. `triple/radial_panel.hpp` and `integrator.hpp`: bounded full-support radial
   partition, incumbent fold maps, p-refinement, local h-refinement, epoch-local
   radius registries and retained parents. Use both embedded differences and
   interpolation-detail floors, sharing `QuadratureErrorPolicy` with binary
   value and tangent controllers. Include source-area/limb normalization in the
   derivative. Internal moving radial-boundary terms cancel between adjacent
   panels because total flux terms are continuous; the exterior bounds have
   zero flux. A fixed partial panel retains its own fixed-endpoint contract.
6. `triple/frame.hpp`, native adapters and Python bindings: translate/rotate the
   existing geometry, apply the analytic 8x11 physical Jacobian and existing
   lens/trajectory parameter pullbacks. Export `triple_algebraic` and explicit
   `LightCurve(..., finite_source_backend="algebraic_boundary")` values and ten
   analytic parameter derivatives. Native calls release the GIL.

### Support-plan design adjustment

The first plan proposed seeding a local radial frontier with existing triple
image/caustic probes. During implementation the complete bounded interval-event
planner proved practical after retaining Bernstein tensors: the representative
contact case fell from about 80,000 direct interval boxes to 1,177 boxes; a
planetary case formerly exhausting 1,000,000 boxes completed in 2,737 boxes.
This avoids paying for centre-image and caustic meshes in the standalone deep
integrator and gives support authority independent of a finite probe set.
Native identical epochs cache the completed support plan. The deep integral and
its error/precision contracts remain the requested algebraic integration.
These box counts are recorded observations, not universal speedup claims.

## Acceptance gates

- Polynomial versus direct lens mapping; incumbent binary limit; frame,
  reflection and coincident-lens identities; local analytic finite differences.
- Root/arc versus independent angular membership; close roots; full circles;
  reciprocal charts; environment, budget and disabled-promotion failures.
- Bivariate tensor/restriction enclosures against independent quad point values;
  physical event residuals and two-root transitions across simple events.
- Retained angular/radial samples and value-only zero parameter-derivative work.
- Complete integrals for regular, contact, large-source, asymmetric, axial,
  symmetric and small-planet-mass cases. Independently adaptive GK flux checks,
  plus external VBM uniform-source contour checks; keep unresolved VBM dark
  comparisons separate from solver acceptance.
- Public ten-component analytic derivatives in both `vbm` and `lcbinint`
  conventions versus two finite-difference steps of the same value engine.
- Main CMake/CTest registration, native public API and existing core regression;
  assertion-enabled Debug kernel tests; source/build hashes and commands.

Implementation is present for all six stages. Final observed checks and their
precise limits are in `evidence/triple_algebraic_kernel/REPORT.md` and
`manifest.json`; usage and supported-domain details are in
`docs/python/triple-algebraic.md`. JAX tracing, dynamic triples, quadratic limb
darkening, and full flux/gradient enclosure certification are outside this
native static implementation; bounded failures are reported explicitly.

## Relative-value performance revision (2026-10-24)

The complete bivariate planner remains the reference and fallback. Its original
per-source cost did not meet the requested VBM comparison, even after reducing
quad work. The explicit one-point Python adapter now also tries
`PreparedRegularSupport` for value requests with `rtol >= 1e-4`. Gradients and
stricter requests retain the complete planner. This does not change the default
LightCurve backend or add a point/hexadecapole shortcut.

### Geometry cache and complete image count

Adapt the incumbent binary caustic-tree strategy to six critical roots:
`sum(m_j/(z-l_j)^2) = exp(i*phase)`. Native polynomial roots are proposals only.
Disjoint contraction disks enclose all six roots over each continuous phase
interval, avoid the poles, and map to source-plane caustic disks. Refine only
intervals which cannot clear the requested source disk. Geometry identity is
`(m1,m2,a,bx,by)`; source positions, source radii, fluxes and radial plans are not
cached. Initial geometry construction is included in cold timing.

Matched endpoint disks give the winding of the continuous mapped critical
curves, with a bound on endpoint-angle uncertainty. In the orientation with the
positive-Jacobian region on the left, the triple image count is `N=4+2*W`.
This specializes the harmonic-map argument principle of
[Sète and Zur (2021)](https://arxiv.org/abs/1908.08759): `N_+=1+W` and
`N_+-N_-=-2`. Refuse ambiguous winding, uncovered critical phases, budget
exhaustion, or disagreement with the native point-image proposals. A finite
caustic polyline alone never authorizes completeness.

### Source-boundary events without global bivariate subdivision

For a source disk clear of all caustics, enclose all image branches of each
source-boundary arc with real implicit-map contraction disks. The certified image
count plus disjoint branch disks accounts for every image. After this initial
cover, refine only the unresolved branch, retaining its siblings. Child branch
identity is checked by a certified midpoint image shared with the parent.

The stationary numerator has an exact cancellation which is particularly useful
for planetary mass ratios. Write `w = source + rho*exp(i*phi)`,
`kappa=sum(m_j/conj(z-l_j)^2)`,
`U=sum(m_j*conj(l_j)/conj(z-l_j)^2)` and
`T=sum(m_j*conj(l_j)/conj(z-l_j)^3)`. Then

```
H = det(J) * Re(conj(z)*z') = Re(conj(source-U)*w')
H' = Re(conj(source-U)*w'' + 2*conj(T*conj(z'))*w')
z' = J^-1 w'
```

The primary at `l_0=0` cancels exactly from `U` and `T`; it is not neglected.
Forward bounds and an interval fallback enclose these two expressions. Sign
exclusion, a nonzero derivative enclosure and certified endpoint signs isolate
all stationary events. Small source-phase brackets can enclose an event directly
through the implicit image disk; charge the entire radial annulus as uncertainty.
If this enclosure is insufficient, refine only that event with the incumbent
physical/quad proof. A refused regular plan uses the complete bivariate planner.
Caustic-crossing source disks currently use that fallback.

### Bounded components and value-only work

When a full-source implicit disk certifies an image branch, bounds on `|det(J)|`
give lower and upper bounds on its point magnification everywhere in the disk.
Their midpoint approximates the component, with half-width charged to the final
error budget. The same interval contains its normalized finite-source contribution
for both uniform and nonnegative linear limb-darkened brightness. This is based
on a computed enclosure, not a mass-ratio cutoff. The component's physical roots
are excluded from the remaining integral, and all components remain accounted
for. Derivative evaluation refuses such a reduced support packet.

For loose value requests, begin angular integration with an embedded one/three
node pair. Retain every sample when extending it. Separate radial and inner
angular error contributions when choosing the next refinement, and retain the
parent workspaces after a split. Strict reference requests and gradients keep
the incumbent refinement schedule: the first unrestricted schedule experiment
exhausted its budget in the strict contact reference and was not adopted there.
The binary difference-safety and interpolation-detail floors are unchanged.

Uniform values allocate endpoint-bracket accuracy from the occupied radial
measure and requested flux tolerance. Actual bracket uncertainty still enters
the flux controller. Tighten an unresolved radius or overlapping root pair
locally. Overlapping certified radial-event intervals may share one value cut;
expand the retained enclosures to contain that cut and charge the full annular
area. This is not an assertion that nearby events are mathematically identical.
Gradient event centres keep their strict independent refinement.

Validation and timings are recorded under
`evidence/triple_relative_optimization_20261024/`. Prepared geometry, cold setup,
full per-source support planning and integration must remain distinguishable.
The performance objective remains uniform-source comparability and a large
limb-darkened advantage at `rtol=1e-3,1e-4`, `atol=0`; successes on regular points
do not establish that objective for the caustic-crossing fallback.

### Contact-search optimization (2026-10-24 evidence)

The complete two-dimensional fallback now favors radial subdivision of thin
image bands, uses midpoint angular subdivision except at the t=0 symmetry
axis, and bounds the stationary numerator after exact primary-lens
cancellation: with g=w-source and U=sum m*conj(l)/conj(z-l)^2,
Re(conj(g)*g_theta)=Re(conj(g)*i(source-U)). All coverage tests, unresolved
siblings, local precision fallback and explicit event budgets remain active.
This does not rely on source flux caching or a sampled caustic polyline.

The bounded contact study in
`evidence/triple_contact_optimization_20261024/REPORT.md` records a median
1.55x gain over final67 across 64 conditions, including near-grazing points
on both sides. Some conditions regress; uniform-source parity with VBM and
exact degenerate tangency certification are not established. Earlier h
splitting was tested and rejected rather than changing tolerance/controller
semantics to obtain a favorable timing.

### Further direct-kernel work reduction

`evidence/triple_work_reduction_20261024/REPORT.md` records the next measured
optimization: reuse accepted Newton evaluations; combine the primary
polar deflection exactly as `(R-m0/R)*unit` in physical jets and angular
proofs; tighten guarded radial-domain bounds; defer broad-box Jacobian tests;
and apply the actual `(1-u),u` profile coefficients to value quadrature
errors. The factor-two safety and detail floor are retained. Gradient and
strict-reference error policies remain separate. No source flux cache or
production default is added.

All 36 conditions improve in a balanced persistent-process before/after
experiment (median speed ratio 1.25). Full Singlepoly/Multipoly/Nopoly tables,
reference discrepancies, budget behavior and the reproduced Nopoly symmetric
LD nonfinite outputs are reported together. Uniform-source contact parity
and regular-position LD performance versus Singlepoly remain unresolved.

### Certified image domains for regular LD

The next stage is recorded in
`evidence/triple_seed_ld_optimization_20261024/REPORT.md`. Prepared image seeds
are continued to new source centres and certified with disjoint point-source
IFT disks and the independent caustic count, avoiding a repeated global
polynomial solve when continuation succeeds. Geometry changes invalidate the
seed bank. Prepared radial rays reuse constant terms in endpoint proofs.

For regular LD plans with two certified events per image and no bounded
component omissions, existing full-source contraction balls classify event
enclosures into disjoint image domains. Strict classification uses certified
rectangle bounds and guarded ball separation. The image-domain boundary adds
no physical boundary because a contraction ball maps every source-disk
solution strictly into its interior. Remaining unsupported or ambiguous
configurations keep the incumbent whole-domain path. All component panels
stay in one adaptive controller and retain their radius workspaces and shared
source coefficient cache. Each domain uses the existing isolated weighted
panel rule. Error safety and requested tolerances are unchanged; there is no
one-annulus LD approximation. The value-only optimization is separate from
the gradient and strict-reference controllers.

The regular LD fixture improves by about 3.14x in paired timing and reaches
VBM Singlepoly's range in the measured point workload. General contact
performance does not materially improve at this stage. Full all-method
comparisons, actual errors, seed/proof counters, held-out cases and Nopoly
nonfinite failures are retained together.

### Uniform domains and contact-proof arithmetic

The certified image-domain decomposition also applies to uniform-source values. Their angular-length primitive remains analytic, and each component starts with the caller's requested angular endpoint tolerance before its automatic budget adjustment. The flux controller remains common across all domains.

Global event discovery dispatches Bernstein split weights outside the inner triangle loops, computes first-difference hulls without materializing derivative tensors, and preserves the correlation of general subdivision weights. Guarded binary64 midpoint endpoints may use SSE2 lanes with the scalar outward-neighbour fallback on other targets. Existing extreme-range interval arithmetic and local quad fallback remain available. Speculative expansion of a certified event neighbourhood was removed; completeness still follows from exploring every uncovered global box. The numerical and performance evidence is in evidence/triple_contact_uniform_optimization_20261025/.

### Contact proof residuals and deferred diagnostics

Bernstein sign exclusions precede the physical disk screen; unknown and unique-root candidates retain the physical guard. Positive derivative scales are deferred until a local Krawczyk test needs them. A physical proof encloses only the residual pair at its centre and reuses the full-box Jacobian to choose a preconditioner, avoiding a second full jet. Dependent interval squares and exact square-jet product rules preserve enclosures without treating identical factors as independent. Frontier diagnostic counts are aggregated on return, preserving cached numerical work across refinements. Error tolerances, safety factors and local precision fallback are unchanged. The accepted scope and rejected contact-domain/contraction experiments are recorded in evidence/triple_contact_proof_optimization_20261025/.


### Binary-inspired work reuse and early event exclusion

Broad event boxes now check only strict common coefficient signs, stopping when neither sign can exclude a zero; local inclusion proofs retain the full hull/Jacobian path. Empty nodes are excluded before tensor restriction around certified regions. Physical exclusion prepares lens/source distances once per event plan and reuses box-centre distances between pole and stationary-boundary screens. Angular continuation keeps an accepted centre when its next Newton correction is already below the existing endpoint budget; interval proofs remain the sole acceptance gate.

Exact sextic and rational even/odd root-pair proposals were tested against the binary m,v formulation but rejected because seed arithmetic outweighed saved scalar iterations. They are archived only under evidence/triple_pair_transport_20261025/. Final contact/grazing paired speedup is 1.032x median over 64 conditions; ordinary-case improvements are not uniform, and uniform contact remains slower than VBM Singlepoly. Tests, held-out and derivative validation, all three VBM methods, failures, and source hashes are retained in that evidence directory. The remaining major bottleneck is globally certified event coverage, not the scalar pair representation alone.


### Partial seed coverage and contained-caustic witnesses

Contact values can now retain individually certified full-source IFT components without first proving completeness of the centre-image seed list. The source-boundary planner matches each retained branch to its disk, and the global stationary-event search proves the complement. Integration separates these components even when the remainder has more than two events, while retaining one controller/error budget and the existing value/gradient separation. Initial phase arcs reuse the full-source disk rather than rebuilding its proof.

The geometry tree retains mapped critical-point disks wholly contained in a source disk. Their inclusion witnesses cheaply reject the regular route for subsequent contained queries. Exact repeated non-budget refusals are eligibility metadata; a budget failure must not suppress progress of the retained tree. No magnification is cached. IFT self-mapping uses rational quadratic remainders, with a separately proved contraction condition and a dominant-pole inverse-disk fallback. Physical event jets defer higher derivatives until the value/first-derivative exclusion test fails.

Final evidence is in evidence/triple_seeded_coverage_20261025/: contact/grazing paired median speed ratio 1.133x (64 conditions), symmetric LD 2.639x at rtol=1e-4, fresh prepared-state whole-matrix median essentially unchanged. Large-source uniform-contact parity with Singlepoly remains unmet. All three VBM methods, cold/persistent timing boundaries, failures, rejected trials, native/debug tests, independent promoted checks, held-out values, and public derivative audits are recorded there.
