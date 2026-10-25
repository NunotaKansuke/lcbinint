from pathlib import Path
import json,statistics as st,hashlib,difflib,subprocess
P=Path(__file__).resolve().parent
load=lambda n:json.loads((P/n).read_text())
paired=load('paired-stream.json');vbm=load('matched-vbm.json');startup=load('startup.json');gap=load('gap-summary.json')
for n in ['paired-stream.json','matched-vbm.json','startup.json','gap-stages.json','saved-heldout.json','gradient-vbm.json','gradient-lcbinint.json']:
 assert load(n)['status']=='PASS',n
assert all(not r['nonfinite'] and not r['above_requested'] for r in vbm['rows'])
header=Path('src/lcbinint/magnification/algebraic_boundary/triple/support_resultant.hpp');test=Path('tests/triple_algebraic/test_resultant.cpp')
patch=[]
for before,after in [(P/'support-baseline-original.hpp',header),(P/'test-resultant-before.cpp',test)]:patch+=difflib.unified_diff(before.read_text().splitlines(True),after.read_text().splitlines(True),fromfile='a/'+str(after),tofile='b/'+str(after))
(P/'source.patch').write_text(''.join(patch))
for path in [header,test]:assert all(line.rstrip()==line for line in path.read_text().splitlines()),path
before=gap['before'];after=gap['algebraic'];single=gap['Singlepoly']['mean_ms']
support_gain=1-after['stages_ms'][0]/before['stages_ms'][0];total_gain=1-after['mean_ms']/before['mean_ms']
checks={'status':'PASS_WITH_PERFORMANCE_LIMITATIONS','native_cpp_tests':20,'public_native_api_tests':'PASS','unique_exact_rational_coefficients':87,'cold_warm_and_invalidation_reference_checks':2262,'coefficient_overlap_diagnostics':3480,'paired_conditions':36,'paired_samples_per_condition':32,'vbm_methods':['Singlepoly','Multipoly','Nopoly'],'vbm_conditions':36,'vbm_nonfinite':0,'vbm_above_requested':0,'heldout_conditions':120,'ld_extreme_conditions':64,'gradient_finite_difference_comparisons':40,'startup_fresh_process_trials_per_condition':4,'startup_queries_per_trial':8,'normal_regression':'about 1-2 percent in repeated regular/planet measurements','short_contact_stream_regression':'about 3-5 percent over first 8 queries','target_30_percent_total_reduction_achieved':False}
(P/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
text=f'''# Triple resultant: reuse the fixed lens algebra

Adopted implementation: polarize the sextic's Bezout matrix in the source variables and cache the lens-only coefficients. In the final matched contact/U/rtol=1e-3 stage measurement, support preparation fell **{support_gain:.1%}**, and total time fell **{total_gain:.1%}**. The requested 30% overall reduction is NOT achieved.

The current change has performance tradeoffs: first-eight-query contact streams are about 3–5% slower because they include basis construction; repeated ordinary regular/planet cases show about 1–2% slower timings. These are retained as limitations, not excluded from the results. This is a useful repeated-geometry optimization, not a universal speedup.

## What changed

Before clearing the sextic discriminant, its polynomial can be written

`Q = Q0 + X*Qx + Y*Qy + (rho²-X²-Y²)*Qdisk`.

The four Q polynomials depend only on lens masses and positions. The Bezout operation is quadratic in Q, so its real symmetric 5x5 matrix has ten source monomials. `LensBasis` constructs the four source-independent Q bases; `MatrixBasis` polarizes the Bezout matrix and retains its ten real polynomial coefficient blocks. A source query now combines these blocks, converts the entries into homogeneous Bernstein form, and computes the degree-28 determinant.

The same row/column permutation minimizes weighted convolution work: 5910 coefficient products versus 6714 in the previous determinant ordering. Applying the same permutation to rows and columns preserves determinant sign. The original source construction remains available for one-shot queries. After four actual resultant queries with identical `(m1,m2,a,bx,by)`, a thread-local geometry basis is built. Source position and rho may vary. Any lens-identity change destroys the old basis. Its heap storage is allocated only when needed.

Only coefficient formation uses the existing hardware extended significand. Integration, physical proofs and ordinary numerical work retain their existing precision and fallback rules. There is no new quad route, no tolerance relaxation, no exact-point flux cache and no cached completeness decision: every query still performs current resultant isolation and physical inclusion.

Production changes are limited to `support_resultant.hpp` and `test_resultant.cpp`. Existing unrelated dirty work was preserved. `source.patch` is relative to the source at the beginning of this turn; the baseline header was checked byte-for-byte against the previous accepted final-source archive.

## Main A/B measurement

CPU0, one active numerical worker, no simultaneous compilation. Both engines use persistent geometry, nine center-source warmups, then the same eight distinct nearby sources over four cycles, with balanced before/after order (32 pairs per condition). The C++ timer includes prepare, support and flux integration; frame conversion, Python dispatch, engine construction and I/O are excluded. The one-off basis construction is included in the separate fresh-process startup experiment below.

Final warm coefficient microbenchmark: {load('final-coefficients.json')['median_coefficient_speedup']:.3f}x (about {1-1/load('final-coefficients.json')['median_coefficient_speedup']:.1%} less coefficient time). This is not whole-integrator speedup. 3480 coefficient-interval overlap checks are diagnostics, not an independent accuracy proof.

| Case | rtol | limb u | Before median ms | After median ms | Median paired time reduction |
|---|---:|---:|---:|---:|---:|
'''
for r in paired['rows']:
 text+=f"| {r['case']} | {r['rtol']:g} | {r['limb']:g} | {r['before_ms']:.4f} | {r['after_ms']:.4f} | {1-1/r['median_pair_speedup']:.1%} |\n"
text+='''
Percentages use the median paired ratio, so they need not equal ratios of the separate time medians. Full raw pairs are in paired-stream.json. Ordinary cases were repeated with 128 pairs in repeat-normal.json; the small regression persisted. A noinline/code-layout trial did not consistently remove it and was reverted. Its cause is unresolved; altered code generation is a hypothesis, not a demonstrated diagnosis.

## Matched VBM comparison

Separate five-engine run: before, current algebraic, Singlepoly, Multipoly and Nopoly; identical saved geometry, source coordinates, rho, limb and relative tolerances (atol/Tol zero). Five original-source warmups and 16 measured evaluations per engine, rotated/reversed engine order, one active worker. All numbers below are median milliseconds. All methods had finite outputs and were within the requested relative error against the saved tight references on this matrix. Agreement with that reference is not an independent proof that one engine is universally correct.

This is a different timing sequence from the A/B experiment above: long VBM LD calls change the surrounding workload. Do not mix its absolute times with the A/B medians or historical measurements. Its shorter before/after comparison has mixed results; it does not establish an improvement in every condition. Annuli and sample counters are retained raw; VBM reports annuli=0 here, so this field does not substantiate a claim that LD used one annulus.

| Case | rtol | u | Before | Algebraic | Singlepoly | Multipoly | Nopoly |
|---|---:|---:|---:|---:|---:|---:|---:|
'''
keys=sorted({(r['case'],r['rtol'],r['limb']) for r in vbm['rows']})
for key in keys:
 r={x['engine']:x['median_ms'] for x in vbm['rows'] if (x['case'],x['rtol'],x['limb'])==key}
 text+=f"| {key[0]} | {key[1]:g} | {key[2]:g} | "+' | '.join(f"{r[e]:.4f}" for e in ['before','algebraic','Singlepoly','Multipoly','Nopoly'])+' |\n'
text+=f'''
## Where the Singlepoly gap moved

The separate stage comparison uses 64 nearby-source evaluations per engine and additive means, with the same balanced five-engine sequence (contact, U, rtol=1e-3). Only this table should be used for the additive decomposition.

| Mean ms | Before | After |
|---|---:|---:|
| Support preparation | {before['stages_ms'][0]:.6f} | {after['stages_ms'][0]:.6f} |
| Fallback | {before['stages_ms'][1]:.6f} | {after['stages_ms'][1]:.6f} |
| Flux integration | {before['stages_ms'][2]:.6f} | {after['stages_ms'][2]:.6f} |
| Total algebraic | {before['mean_ms']:.6f} | {after['mean_ms']:.6f} |
| Singlepoly, same sequence | {single:.6f} | {single:.6f} |
| Difference from Singlepoly | {before['mean_ms']-single:.6f} | {after['mean_ms']-single:.6f} |

Multipoly: {gap['Multipoly']['mean_ms']:.6f} ms; Nopoly: {gap['Nopoly']['mean_ms']:.6f} ms. The gap to Singlepoly fell by {1-(after['mean_ms']-single)/(before['mean_ms']-single):.1%} in this measurement; it remains positive. Flux time is essentially unchanged, supporting that the gain came from preparation.

## Startup cost is not free

startup.json starts fresh processes (resetting the geometry-only TLS cache), uses no warmup, and measures the first eight distinct sources. Four trials per condition, balanced before/after order. A new PreparedRegularSupport alone would NOT reset this TLS cache, so the previous cold_paired.py protocol is not used as evidence of cold lens-basis cost.

The fourth actual resultant query constructs the basis. In contact, that query adds approximately 0.6–0.7 ms; the first-eight-query total is approximately 3–5% slower. Initial single-query times are noisy across four trials and show no consistent speedup. Long streams amortize this expense; short streams can lose. No claim is made that a one-shot point evaluation has become 8% faster.

| Contact u | rtol | First-eight paired speed ratio (before/after) |
|---|---:|---:|
'''
for r in startup['rows']:
 if r['case']=='contact':text+=f"| {r['limb']:g} | {r['rtol']:g} | {r['first8_pair_speedup']:.4f} |\n"
text+='''
## Completeness-cache research continued

The previous attempt independently widened the source variables through the degree-28 determinant; its enclosures were unusable. This turn also implemented a different prototype: prove one event tube for every source in a square neighborhood, subtract those tubes (including opposite-chart coverage), then exclude the remaining physical F/F_t domain. A source-aware physical exclusion bound handles poles; parameter-interval Krawczyk exclusions eliminate the remaining overestimation strips. Unresolved boxes and failed tubes remain failures, never acceptance.

At halfwidth 1e-3*rho, the prototype reaches zero unresolved cells in 6/9 geometries, including contact, collinear and asymmetric. Contact proves 12 tubes and completes the complement in 6132 evaluated cells, about 10.5 ms in that run. The simple direct-physical prototype from the preceding turn did not reach this point. Large fails to construct all tubes at that width; broader 1e-2*rho neighborhoods mostly fail. The tube prototype is not installed in PreparedRegularSupport: its construction cost and narrow reuse neighborhood do not meet the direct-point workload, and independent source-neighborhood/degeneracy validation is still required. No speedup is attributed to this research prototype.

The files tube-probe.cpp, tube-probe-krawczyk.jsonl and earlier intermediate outputs preserve this investigation. Reproduction links the existing native geometry object and magnification archive, as shown in the recorded build scripts; the physical source-interval Ray snapshot is in ../triple_source_certificate/physical_box.hpp.

## Validation and rejected variants

- 20 native C++ tests passed; public native API/invariance/gradient smoke tests passed.
- 87 unique exact Gaussian-rational reference coefficients passed in cold, warm and every lens-identity invalidation sequence: 2262 reference enclosure checks. Assertions build passed as well.
- All 36 paired conditions passed requested relative error and reported-error-plus-reference checks at the measured nearby coordinates.
- Saved heldout matrix: 120 conditions passed; LD extremes (u=0 and u=1, including grazing): 64 passed. No reference was recomputed or tolerance loosened.
- Both parameter conventions passed 40 analytic-gradient versus two-step finite-difference comparisons using the final native build. This is a bounded gradient check, not general inference certification.
- Generic sparse complex multiplication: 0.990x coefficient speed ratio, rejected.
- Reordered determinant alone: 1.063x; Q-basis-only prototype: 1.129x. The adopted implementation additionally caches the polarized matrix.
- Exact-zero skipping during matrix preparation did not improve its measured construction time, rejected. The noinline/code-layout trial did not consistently fix ordinary timing, rejected.
- No UBSan result is claimed: the installed compiler's UBSan runtime was already found missing in the preceding source-certificate experiment. This turn uses the successful assertions build and the native numerical checks.

## Remaining work

The total 30% target remains open. The next useful work is to reduce matrix-basis startup cost, avoid the ordinary-case regression, and make a source-neighborhood certificate cheap enough to amortize at realistic source spacing. The current measured improvement is a repeated-lens preparation gain; it does not solve the remaining Singlepoly uniform gap or all LD cost.

Raw inputs are linked to the prior preserved conditions/references. provenance.json identifies the source/build hashes, compiler, branch/HEAD and timing boundary. source.patch contains only this turn's two production-file changes. SHA256SUMS covers the retained local artifacts. No commit or push was made.
'''
(P/'REPORT.md').write_text(text)
print('report written; support reduction',support_gain,'total reduction',total_gain)
