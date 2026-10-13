# LD radial product integration v43

## Mathematical mechanism

For an isolated occupied radial span with ordinary physical folds at both ends,
let R=a+(b-a)(1+x)/2 and w=1-x*x. The existing angular kernels have endpoint
forms f0=sqrt(w)*A(x), fh=w*B(x). Thus the two contributions use different
product measures: sqrt(w) dx for uniform brightness and w dx for linear LD.
Both kernels are evaluated together at the same nested Fejer-II nodes.

The uniform rule remains Gauss-Chebyshev-II. The LD rule is an interpolatory
product rule (not Gaussian) on those same nodes. For k odd,

    integral[-1,1] (1-x*x) U_(k-1)(x) dx = 4/[k*(4-k*k)].

For k even the moment is zero. Discrete sine orthogonality gives

    W_i = (2/m) sin(theta_i) sum_{k odd<m}
          4 sin(k theta_i)/[k*(4-k*k)], theta_i=pi*i/m.

It integrates polynomials through degree m-2 exactly. Cached dyadic nodes and
interpolation matrices are shared; no second root/angle evaluation is needed.
Each quotient gets its own full interpolation-detail estimate and contraction
check; radial errors are added, with no cancellation credit between channels.
Odd remainders integrate to zero under either symmetric measure.

## Numerical scope and ledger

Enabled only in the experimental frontier value-only route, for whole-input
isolated two-crossing double-fold eligibility already used by v42 uniform.
Other spans retain the incumbent controller. Weighted parents that split use
ordinary one-sided fold children, preserving interval coverage and atomic
parent replacement. Gradient routes retain their prior rules. Tolerances,
node budgets, local count/correction gates and event decisions are unchanged.

Angular, geometry and existing roundoff error at each stored sample use the
maximum of the uniform rule coefficient and LD rule coefficient divided by
sqrt(w). Thus changing the integration measure does not silently reduce those
error ledgers. Additional weighted-sum roundoff is accounted for. Event
remainders use the combined physical integrand. coverage_certified remains
false; the error estimator remains a contraction model, not a proven bound.

## Final observed results

All 14,432 rows: new reference tolerance violations 0, new failures 0;
existing violations 3 remain; nonconvergences improve 6 -> 4. Low-A<1.1 LD
at rtol=1e-3 has median nodes 53 -> 29 (uniform remains 22).

| workload | forward v42 -> v43 | reverse v42 -> v43 |
|---|---:|---:|
| low LD integration, 1e-3, median us | 41.612 -> 26.712 | 41.784 -> 27.998 |
| low LD integration, 1e-4, median us | 47.030 -> 39.432 | 47.249 -> 39.687 |
| low LD full, 1e-3, median us | 70.205 -> 69.228 | 70.050 -> 71.537 |
| low LD full, 1e-4, median us | 74.362 -> 68.260 | 73.110 -> 69.088 |
| normal 87 full, mean pair-median ms | .237793 -> .227871 | .239700 -> .229389 |

Low LD paired median reductions: integration 30.5--30.8% (1e-3) and
25.8--26.8% (1e-4); full 4.6--5.3% and 4.3--5.1%, respectively.
Population medians and medians of matched ratios differ; the low-LD full
population median at 1e-3 worsens in the reverse order. Do not claim a
uniform improvement for every input. Normal 87 improves 4.2--4.3% in both
orders, with caustic construction excluded. Low uniform full paired ratios
worsen 1.2--1.7%, and integration worsens 2.3--2.8%; its quadrature and node
counts are unchanged, but the shared code/storage change has a small cost.
All measured inputs, including regressions, are retained.

## Validation and timing

See audit.json and summary.json for the final measured results. Independent
polynomial moment and mixed beta-integral checks are in test_split.log; existing
weighted, adaptive and frontier checks retain their direct transport, fold-map,
arc sign, anchor and count-witness audits. Full comparison uses the saved
VBM RelTol=1e-6 references and all 14,432 input/tolerance rows, including failures.

Timing compares preserved v42 binaries against v43: fresh seed included,
caustic preparation excluded, one thread, 5 repetitions per input, both serial
orders. Low-magnification inputs and the normal 87-pair matrix are reported
separately. Saved VBM timings in summary.json were not rerun.

## Reproduction

Run from /rogue1_8/nunota/lcbinint/experiments/seeded-boundary-events:

    bash evidence/holonomic/ld_split_radial_v43/build.sh
    bash evidence/holonomic/ld_split_radial_v43/build_normal.sh
    bash evidence/holonomic/ld_split_radial_v43/build_tests.sh
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 build-seeded-events/bench_ld_split_v43 evidence/holonomic/seeded_paper_qrho_20261013/input_snapshot.tsv evidence/holonomic/ld_split_radial_v43/full_after.tsv 1
    python evidence/holonomic/ld_split_radial_v43/audit.py
    bash evidence/holonomic/ld_split_radial_v43/run_timing.sh
    python evidence/holonomic/ld_split_radial_v43/summarize.py

Source snapshots and SHA256SUMS preserve the final dirty source identity and
baseline v42 headers. No commit, push, or production/default promotion.
