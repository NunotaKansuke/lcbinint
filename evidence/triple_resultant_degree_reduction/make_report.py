from pathlib import Path
import json,statistics as st
P=Path(__file__).resolve().parent
load=lambda name:json.loads((P/name).read_text())
paired=load('paired-stream.json');startup=load('startup-ready.json');vbm=load('matched-vbm.json');gap=load('gap-summary.json');normal=load('repeat-normal.json');audit=load('audit.json');work=load('work-counters.json')
for name in ['paired-stream.json','startup-ready.json','matched-vbm.json','gap-stages.json','repeat-normal.json','saved-heldout.json','gradient-vbm.json','gradient-lcbinint.json','work-counters.json']:
 assert load(name)['status']=='PASS',name
assert len(audit)==1620
assert all(not r['nonfinite'] and not r['above_requested'] for r in vbm['rows'])
maximum=max(r['relative_error']/r['rtol'] for r in audit)
checks=dict(status='PASS',native_cpp_tests=20,assertions_tests=2,exact_generated_matrix_coefficients=4950,matrix_basis_overlap_diagnostics=86400,exact_direct_coefficient_parity=1856,warm_coefficient_overlap_diagnostics=1856,unique_exact_rational_resultant_coefficients=87,cold_warm_and_lens_invalidation_reference_checks=2262,magnitude_independent_quad_comparisons=24022,cached_motion_enclosure_endpoints_equal=46416,direct_point_conditions=324,repeats_per_condition=5,direct_point_evaluations=1620,heldout_conditions=120,ld_extreme_conditions=64,gradient_finite_difference_comparisons=40,max_requested_error_fraction=maximum,vbm_methods=['Singlepoly','Multipoly','Nopoly'],vbm_nonfinite=0,vbm_above_requested=0,startup_fresh_process_trials=8,startup_ready_barrier=True,target_30_percent_universal_reduction_achieved=False)
(P/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
text='''# Triple algebraic: reduce preparation, image proofs and LD value cost

The final implementation improves several independent parts without changing requested tolerances, final acceptance criteria, default backend selection or parameter-derivative arithmetic. Compared with the previous accepted triple_resultant_sparse build, the paired nearby-source tests show roughly 4–7% less time in the regular fixture, 12–16% in far, 5–6% for contact LD and 6–7% for large LD. Contact uniform improves only about 1–3%. These are measured fixture results, not a universal speedup or full-fit certification. The 30% overall target remains open.

## Adopted changes

1. **Share fixed-lens polynomial products.** The old matrix-basis constructor repeatedly formed derivative products and then applied the real congruence. The generated exact contraction has 96 integer terms and only 24 distinct sextic coefficient products. Each product is formed once per source-basis pair and reused across all real matrix entries. Exact zero tails are inspected once on the fixed bases; uncertain zeros are retained. The isolated balanced constructor probe fell from 0.6764 to 0.4622 ms (about 32% less time). Only the existing hardware extended-significand coefficient tier is used.
2. **Use the root-count certificate immediately.** One Bernstein variation already establishes exactly one discriminant root in a radial cell. The planner now attempts current physical inclusion there, instead of splitting unconditionally to depth six. A failed proposal retains its Newton work for child cells. No event is accepted without the same current physical inclusion, and unresolved coverage still goes to the incumbent physical planner.
3. **Defer component construction until it is needed.** The projected planner first constructs global coverage. If unresolved bands remain, a lazy factory builds the partial image components and the physical fallback receives the already completed events and radial exclusions. A complete one-dimensional projection does not call the center-image solver. Tests assert both zero factory calls for complete projection and exactly one call for fallback. No completed projection work is discarded and rebuilt.
4. **Reuse normal-range magnitudes in the value kernel.** The double LD kernel already needs squared norms. Their square roots now supply the magnitudes used solely in roundoff bounds, with four-epsilon relative padding. Tiny/overflowing squared norms retain scaled hypot. The physical phi calculation is unchanged. The helper is independently checked against quad precision over 24022 cases, including extreme/subnormal inputs. Gradient-only work is not introduced into the value path.
5. **Reuse disjoint full-source image disks and their motion model.** A successful full-source contraction already encloses an entire image branch. When all retained disks are proven and disjoint from each other and omitted disks, the planner starts from those disks instead of proving eight new source-arc disks per image. It prepares the disk-invariant kappa/U/third-derivative bounds once. The motion enclosure arithmetic matches the old implementation at 46416 tested interval endpoints. If the existing cheap-motion validity test fails, the old initial-cover route remains. If a particular arc is unresolved, only that arc receives a narrower disk, validated by containment or a common certified image witness. Other accepted branches remain intact. The already computed motion enclosure is also passed to event refinement.

The full-source reuse gate is based on actual disjointness and the existing motion-bound validity criterion, not a benchmark-case name or fitted source-size threshold. There is no cached flux and no previous-query completeness decision. The existing caustic count/cache and seed validation still establish image coverage. Motion models live only within one source query; the fixed-lens coefficient cache keeps its exact lens identity and four-query activation policy.

## Numerical checks

- 20 native C++ tests passed, including event-count changes across source-limb contact, independent full 2D event coverage, cache invalidation, budget failures and component masks.
- Assertions builds of resultant and magnitude tests passed. The 87 unique exact rational resultant reference coefficients are tested on cold, warm and lens-invalidation paths (2262 enclosure checks).
- The term generator independently compares the integer table with the unsimplified Bezout/congruence construction on generic Gaussian-integer polynomials: 4950 exact coefficient equalities. Run `python tests/triple_algebraic/generate_resultant_matrix_terms.py --check`.
- 324 source/profile/tolerance conditions were checked for five consecutive calls each, including the fourth-call matrix-basis construction: 1620 direct point evaluations. All converged and passed requested relative error and reported-error-plus-reference checks. Maximum observed error was '''+f'{maximum:.4f}'+''' of the requested allowance. These checks use saved tight references; they are not universal precision certification.
- Separate saved heldout matrix: 120 checks passed. LD extremes including grazing cases: 64 passed. Public native API, translation/rotation invariance and gradient smoke tests passed. Both public parameter conventions passed 40 two-step finite-difference gradient comparisons.
- All three VBM methods were finite and within requested relative error against the saved reference matrix. This does not establish that any engine is universally correct or that a future discrepancy must belong to another engine.

## Direct point A/B benchmark

CPU0, one active numerical worker, no concurrent compilation. Same persistent lens geometry, source coordinates, rho, limb and tolerances; absolute tolerance zero. Nine original-source warmups, then eight distinct nearby sources over four cycles, with balanced before/after order: 32 pairs per condition. C++ timers include support preparation and integration; engine construction, frame conversion, Python dispatch and I/O are excluded. All timings below are median milliseconds. Reductions use the median paired ratio, so they need not equal ratios of separate medians.

| Case | rtol | u | Before ms | After ms | Paired time reduction |
|---|---:|---:|---:|---:|---:|
'''
for r in paired['rows']:
 text+=f"| {r['case']} | {r['rtol']:g} | {r['limb']:g} | {r['before_ms']:.4f} | {r['after_ms']:.4f} | {1-1/r['median_pair_speedup']:.1%} |\n"
text+='''
## Ordinary-case repeat check

The prior change had a small ordinary-case slowdown. The final regular/planet fixtures were repeated for 128 balanced pairs per condition. This run shows improvement or near-neutral behavior; the full table is retained rather than selecting only favorable tolerances. The separate symmetric fixture is close to neutral at rtol=1e-3, so no universal ordinary-case improvement is claimed.

| Case | rtol | u | Before ms | After ms | Paired time reduction |
|---|---:|---:|---:|---:|---:|
'''
for r in normal['rows']:
 text+=f"| {r['case']} | {r['rtol']:g} | {r['limb']:g} | {r['before_ms']:.4f} | {r['after_ms']:.4f} | {1-1/r['median_pair_speedup']:.1%} |\n"
text+='''
## Initial calls, with an explicit readiness barrier

The final startup experiment uses eight fresh-process pairs per condition, zero numerical warmups and eight distinct source coordinates. Both engines must send READY after library import and object creation before the first query is issued. Thus another worker's initialization cannot overlap the first integration timer. Library import and process creation are excluded; actual seed, caustic-tree and matrix-basis work inside each point call is included. New PreparedRegularSupport objects alone would not reset the TLS matrix cache, so fresh processes are required.

Earlier startup.json files without the readiness barrier are intermediate diagnostics; startup-ready.json is authoritative. The fourth actual resultant call still has a basis-construction expense, but it is smaller. The first-eight-query contact streams now improve about 4–7% relative to the previous accepted build. First-call times remain noisy and not every individual condition wins. This is not a claim that all one-shot calls are faster, or that setup cost has disappeared.

| Case | rtol | u | First query before ms | First query after ms | First-eight paired time reduction |
|---|---:|---:|---:|---:|---:|
'''
for r in startup['rows']:
 q=r['queries'][0]
 text+=f"| {r['case']} | {r['rtol']:g} | {r['limb']:g} | {q['before_ms']:.4f} | {q['after_ms']:.4f} | {1-1/r['first8_pair_speedup']:.1%} |\n"
text+='''
## Matched VBM comparison: Singlepoly, Multipoly, Nopoly

Same five-engine protocol as the prior report: before/current algebraic and the three VBM methods, identical physical geometry, source, rho, u and relative tolerance, with absolute tolerance/Tol zero. Five original-source warmups followed by 16 measured fresh-source evaluations per engine, rotated/reversed engine order. Timers include each direct point integrator, with VBM SetLensGeometry outside the timer as before. Values below are median milliseconds. The workload differs from the two-engine A/B run, so do not combine these absolute times with those medians or historical timings.

VBM samples and annuli are saved in matched-vbm.json. Its reported annuli counter is zero here; that field does not prove that LD used one annulus. Both uniform and LD APIs are invoked directly with the requested profile.

| Case | rtol | u | Before | Algebraic | Singlepoly | Multipoly | Nopoly |
|---|---:|---:|---:|---:|---:|---:|---:|
'''
for key in sorted({(r['case'],r['rtol'],r['limb']) for r in vbm['rows']}):
 row={r['engine']:r['median_ms'] for r in vbm['rows'] if (r['case'],r['rtol'],r['limb'])==key}
 text+=f'| {key[0]} | {key[1]:g} | {key[2]:g} | '+' | '.join(f'{row[e]:.4f}' for e in ['before','algebraic','Singlepoly','Multipoly','Nopoly'])+' |\n'
before=gap['before'];after=gap['algebraic'];single=gap['Singlepoly']['mean_ms']
text+='''
## Remaining contact-uniform gap

A separate 64-evaluation, balanced five-engine run uses additive means for contact/u=0/rtol=1e-3. Only this table should be used for this stage decomposition.

| Mean ms | Before | After |
|---|---:|---:|
'''
for name,a,b in [('Support preparation',before['stages_ms'][0],after['stages_ms'][0]),('Fallback',before['stages_ms'][1],after['stages_ms'][1]),('Flux integration',before['stages_ms'][2],after['stages_ms'][2]),('Total algebraic',before['mean_ms'],after['mean_ms']),('Singlepoly',single,single),('Difference from Singlepoly',before['mean_ms']-single,after['mean_ms']-single)]:text+=f'| {name} | {a:.6f} | {b:.6f} |\n'
text+=f"\nMultipoly: {gap['Multipoly']['mean_ms']:.6f} ms; Nopoly: {gap['Nopoly']['mean_ms']:.6f} ms. Support preparation fell {1-after['stages_ms'][0]/before['stages_ms'][0]:.1%} in this run. Uniform flux time is unchanged, as expected from a value-bound helper used only by LD. Contact uniform at 1e-3 still loses to Singlepoly; the total 30% reduction target remains unmet.\n"
text+='''
## Work counters

Center-source fifth-call counts at rtol=1e-3/u=0, comparing before/current point executables. Work counters only: executable diagnostic I/O timings are not used for speed claims.

| Case | Resultant cells before | After | Physical/branch boxes before | After |
|---|---:|---:|---:|---:|
'''
for r in work['rows']:
 if (r['limb'],r['rtol'],r['repeat'])==(0.,.001,4):
  text+=f"| {r['case']} | {r['before']['resultant_nodes']} | {r['after']['resultant_nodes']} | {r['before']['boxes']} | {r['after']['boxes']} |\n"
text+='''
## Isolated LD check and rejected variants

magnitude-only/paired-profiles.json compares the value-magnitude helper against the immediately preceding global-first prototype, before the ordinary-cover changes. Uniform time is close to unchanged. LD time falls about 4–5% in contact and 7–9% in large, depending on source/profile. The final whole-change comparison is the A/B table above; those isolated percentages must not be added arithmetically to it.

The generic polynomial-tail normalization prototype was slower (degree-probe.json) and was reverted. The adopted fixed-basis code inspects tails once and shares symbolic products instead. A naive full-source-cover prototype made symmetric cases slower; the final implementation adds the existing cheap-model gate, shared disk preparation and per-arc tightening rather than introducing a symmetric-case special rule. Its source snapshot is retained as regular-full-cover-unrefined.hpp; intermediate timing observations are diagnostic, not final speed claims.

No source-neighborhood completeness cache from the earlier 10-ms research prototype is installed. No tolerance was loosened to obtain these gains. The coefficient tier, quad promotion policy, gradient-only arithmetic and failure gates retain their existing roles. Full mixed-trajectory/fit performance is not established by this direct-point benchmark.

## Reproduction and provenance

The canonical checkout, branch/HEAD, compiler, input/reference hashes, native module and baseline/candidate library hashes are recorded in provenance.json. The baseline kernel hash matches the prior accepted triple_resultant_sparse candidate. source.patch contains only this turn's eight source/test files and passes reverse apply-check against the current files. Unrelated dirty/untracked work was preserved; no commit or push was performed.

Build the configured native checkout with TMPDIR set to this directory's compiler-tmp and -j2, then run kernel-build.sh and build-point.sh. The existing scripts reproduce the paired, startup-ready, VBM, heldout, LD-extreme and API checks. Debug/assertion executables and logs are retained. SHA256SUMS covers the local artifacts; input/reference symlink targets have separate hashes in provenance.json. No UBSan pass is claimed because the installed runtime was unavailable in the preceding experiment.
'''
(P/'REPORT.md').write_text(text)
print('report and checks written')
