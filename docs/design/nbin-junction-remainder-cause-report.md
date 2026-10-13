# Junction remainder diagnosis and rejected ablations

## Observed cause

The previous maximum 2.431413 (fresh s=1.3106493282,q=.00012191597,c=1,bins128) decomposes into:

- actual absolute error:7.3539932881e-6
- signed inner plus outer residual:8.5781690092e-6
- inner remainder:3.5856445723e-8
- outer remainder:9.2665696976e-6

Outer remainder parts in magnification units:

| term | value |
|---|---:|
| width expansion |0|
| LD endpoint held-out row |8.1608754080e-10|
| width held-out row |0|
| junction second differences |9.2577953546e-6|
| smooth fourth differences |7.9582555131e-9|
| internal endpoint model |0|

Thus 99.9% of the outer remainder comes from junction rows. The other large cases show the same concentration. Brightness-table uncertainty is zero. This attribution is measured, not inferred from the overall ratio.

The current estimator flags a junction whenever run multiplicity changes anywhere in a four-row stencil, subtracts second differences predicted by successfully fitted internal singular models, then adds abs(unexplained second difference)/24 separately for each junction row. It can overestimate, but those differences cannot simply be ignored.

## Experiments: all 82 matched numerical-reference conditions

1. Subtract an interpolated smooth curvature background from flanking retained rows, keeping their disagreement and the original local fallback. Ten indicators changed but the maximum remained2.431413. Rejected as ineffective.
2. Sum signed unexplained second differences within each contiguous junction patch before taking abs. Maximum fell1.435475, >=1.1 fell15→5, but minimum ratio fell.667736 (33.2% underestimate). Rejected. Other underestimates included1.3,c=.5,bins256 and several bins512 conditions. Cancellation inside a junction does not establish a small quadrature defect.
3. Retain the total variation of unexplained second differences, including zero-halo patch endpoints, as additional uncertainty and cap at the original local estimate. All meaningful results were unchanged. Rejected as ineffective; one floating-point-last-bit difference is not an improvement.

The original estimator was restored. Replay confirms all82 magnifications and indicators bit equal the prior inline analytic-LD build. Unit core/boundary support2/2 pass. No numerical speed improvement is claimed; expensive diagnostic arrays and accumulation were removed. Only inner/outer remainder scalar fields remain exposed from already computed counters.

## Broader fixed-resolution check

27 extra combinations from the same three independent-reference geometries: c=0,.5,1; bins96,192,384. Both variation candidate and restored original give minimum ratio.993090 and maximum3.874914, with no deficit exceeding1.1%. Worst is the same fresh geometry,c=1,bins96. These are additional resolutions, not independent new geometries. The82-condition maximum2.43 is therefore not a universal maximum. No runtime resolution-doubling/convergence ladder was introduced.

## Mathematical next model, not yet established on the lens case

Near a saddle, take normalized source depth D(x,t)=x²-t. A linear-LD row component is proportional to the integral of sqrt(max(D,0)) over [-L,L]. For L=1:

F(t)=sqrt(1-t)-t*log((1+sqrt(1-t))/sqrt(abs(t))), t!=0.

Consequently F(t)=1-(1/2+log2)*t+(1/2)*t*log(abs(t))+t²/8+O(t³).

Independent SciPy integration at eight positive/negative t values agrees with the exact formula within7.8e-16. The remainder/t² approaches1/8. This establishes a t log|t| row singularity for the toy saddle, distinct from a regular sqrt(t) tip or the regular-cap t+t² LD basis.

The internal gap model scales its uniform-width singular contribution by limb brightness (1-c), which vanishes atc=1. That cannot describe this LD saddle term. This is a structurally motivated candidate for the missing junction model, not proof that the measured worst lens case is dominated by this saddle normal form.

Next implementation should first verify the local saddle signature and phase against retained actual lens rows, then fit the regular background plus t log|t| and evaluate its quadrature defect. Keep the current local fallback where the fit is unresolved. Reuse rows and crossing fractions; do not add global ray shooting or a fitted global safety multiplier. The failed signed-sum ablation must not be promoted.

## Reproducible artifacts

Under tests/diagnostics/results/recal2026:

- shared-flux-remainder-cause-v1: inner/outer decomposition.
- shared-flux-remainder-parts-v1: six outer parts.
- shared-flux-junction-background-v1: ineffective background subtraction.
- shared-flux-junction-grouped-v1: rejected unsafe cancellation.
- shared-flux-junction-variation-v1: ineffective variation remainder, source snapshot, restore logs.
- shared-flux-junction-restored-v1: restored82-condition replay.
- shared-flux-junction-resolution-v1 and shared-flux-junction-resolution-restored-v1: extra27 fixed-resolution combinations.
- ld-saddle-row-model-v1: independent toy analytic/integral check.

Use the instrumented build via LCBININT_PROBE_BUILD=$PWD/build-boundary-support, OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, Python /home/nunota/.miniconda3/envs/myenv/bin/python. Replay module tests.diagnostics.recal2026.shared_flux_exact_ld_audit accepts --output; extra-resolution module shared_flux_junction_resolution_audit accepts --output; toy module ld_saddle_row_model writes an exclusive output. Preserve rejected artifacts.
