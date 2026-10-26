# Triple primal radial error estimator optimization

This phase optimizes naturally integrated source disks, not point/hex routes. Production changes are limited to radial_panel.hpp and test_panel.cpp relative to this phase's saved before snapshots.

The unweighted primal controller now estimates error on the actual physical profile `(1-u)F0+u*Fh`, avoiding a triangle-bound sum of separate component errors. Fine Fejer levels 3/4/5 reuse existing samples for positive 5/7, 13/15, and 29/31 embedded comparisons. Two omitted interpolation witnesses retain a detail floor. The alternative estimate is used only when successive differences contract by at least a factor of four above the rounding floor. The fine integral, shared safety factor 2, and requested tolerance are unchanged. Weighted panels and gradient controllers are unchanged. This is an error estimator, not a rigorous bound for arbitrary source geometry.

The two LD-specific experiments (local endpoint-error allocation and faster repair arithmetic) were rejected after ablation: they did not provide a stable speed improvement. Their raw data and source snapshots remain here.

Protocol: immutable baseline native library and immutable VBM wrapper, matched inputs/tolerances, CPU0 thread CPU time, separate prepared state per method and condition, rotated/reversed method order, four warmups and eight timed calls. Three complete repetitions retain timing spread. Turbo/frequency are not fixed; CPU-frequency samples are context, not clock normalization. VBM Singlepoly, Multipoly, and Nopoly are always reported separately.

Final numerical summaries and hashes are in checks.json and provenance.json. Raw repeats are final-matched-{0,1,2}.json. Validation uses tighter native and independent VBM comparisons; reference agreement is evidence, not an absolute proof of correctness. Unit tests: 8/8 pass (tests-final.log). Full gradient precision certification is outside this phase.

## Final matched results

Each time is the median across three repeats per condition, then across twelve source disks; reductions are geometric means of paired per-condition ratios, shown as the range across three repeats. These aggregations differ, so a ratio of table medians need not equal the paired reduction. Units: milliseconds.

|Profile|rtol|Before|After|Singlepoly|Multipoly|Nopoly|Paired reduction|
|---|---:|---:|---:|---:|---:|---:|---:|
|Uniform|0.001|0.780|0.738|0.567|0.725|1.740|5.2–6.3%|
|Uniform|0.0001|0.947|0.890|1.110|1.465|3.374|7.6–8.6%|
|Linear LD u=0.5|0.001|1.104|1.019|10.921|14.625|34.243|9.0–9.9%|
|Linear LD u=0.5|0.0001|1.456|1.179|50.045|66.205|153.756|8.7–10.7%|

Accuracy: all 48 main conditions and 56 additional conditions passed. Main comparisons include all four stored tighter reference methods and supplemental native refinements. Additional conditions use native rtol1e-9 and Singlepoly rtol1e-6 references. All 56 additional estimated errors contain the native-reference difference (including reference error). This is tested agreement, not universal certification. All VBM values, sample counts and annulus counts are unchanged across the three repeats. Uniform rtol1e-3 remains slower than Singlepoly in the aggregate.

Embedded weights are computed lazily once per level in long double and cached; these warmed benchmarks do not measure process startup. No lens evaluation uses new extended precision. Gradients retain their original controller.
