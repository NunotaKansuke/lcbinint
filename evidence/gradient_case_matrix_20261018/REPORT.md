# Other light-curve cases: value and fused-gradient timings

2026-10-18, experiment/algebraic_boundary at e740210b plus the existing optimized working tree. No production changes in this measurement task. Native hashes in native-hashes.json.

Compare the saved easy_gradient_20261018/before-package build against current build-route-unified. The saved build already includes the earlier root-handoff and repair optimizations; this measures the subsequent covering-disk, workspace-retention and interval-first improvements, not the entire difference from unoptimized HEAD.

16 inputs at each of atol=rtol=1e-4 and1e-3. Ordinary trajectories401 epochs; saved cases033,081,107,117 have33. Explicit VBM parameter convention, float64, one pinned CPU, OMP/OpenBLAS threads=1. Uniform/linear LD=.5 pairs for ordinary inputs, original LD for saved cases (case107=.999999). Parameters, ranges and LD are in each inputs.json.

Each value/gradient pair is run in a new subprocess, warmed on a different lens. Separate new LightCurve objects for the value and gradient call; order alternates. Three independent paired repetitions, median. Object construction/import and unrelated warmup are outside timers. Public API calls include target geometry/caustic preparation, selection, seed generation, integration and diagnostics. Target caches are not reused between calls or repetitions. Previous same-object repeated-call timings do not have this same cold preparation boundary and must not be directly compared as absolute performance. Preparation shared within one light curve is retained. Ratios can be low because common model preparation dominates both modes; a low full-curve ratio does not imply low per-integral derivative overhead.

All32 conditions have matching route counts across builds and value/gradient modes, finite outputs, numerical convergence at every epoch and all gradient-valid flags true. Value/fused output differences are recorded in comparison.json. These checks are numerical observations, not independent gradient accuracy certification. Three repetitions cannot establish significance of sub-percent timing changes. No VBM comparison was performed.

## Matched results, atol=rtol=1e-4

| Case | N | Value ms | Old gradient ms | Current gradient ms | Current/old | Gradient/value | point/hex/integral |
|---|---:|---:|---:|---:|---:|---:|---|
|ordinary_uniform|401|6.168|6.994|6.496|0.929|1.05|276/121/4|
|ordinary_LD|401|11.021|11.813|11.396|0.965|1.03|276/121/4|
|close_planet_uniform|401|4.862|5.397|5.329|0.987|1.10|340/61/0|
|close_planet_LD|401|4.853|5.312|5.298|0.998|1.09|340/61/0|
|wide_planet_uniform|401|5.207|5.693|5.676|0.997|1.09|343/58/0|
|wide_planet_LD|401|5.193|5.656|5.615|0.993|1.08|343/58/0|
|resonant_uniform|401|6.587|8.208|7.640|0.931|1.16|320/68/13|
|resonant_LD|401|11.499|13.370|12.795|0.957|1.11|320/68/13|
|equal_mass_uniform|401|6.585|7.618|6.995|0.918|1.06|278/118/5|
|equal_mass_LD|401|11.435|12.529|11.956|0.954|1.05|278/118/5|
|large_source_uniform|401|39.563|84.908|77.259|0.910|1.95|0/282/119|
|large_source_LD|401|46.027|101.454|92.283|0.910|2.00|0/282/119|
|case_033|33|4.711|34.973|34.943|0.999|7.42|0/32/1|
|case_081|33|17.243|47.191|41.398|0.877|2.40|0/5/28|
|case_107|33|0.114|0.150|0.137|0.915|1.20|33/0/0|
|case_117|33|20.726|66.524|65.399|0.983|3.16|0/0/33|

At1e-3, point-only close/wide inputs cost ~0.45-0.46ms value and~0.69-0.76ms gradient: ~1.49-1.67x. This is about0.23-0.30ms added work, not rare geometry detection. At1e-4 these trajectories include hex and larger geometry preparation, changing the full-curve denominator. Large-source inputs remain ~2x; case033 remains7.42x at1e-4 despite only one integral epoch, a specific expensive derivative epoch worth distinguishing from blanket slowdowns.

## Reproduce

Use myenv Python with run.py (1e-4) or run_1e3.py (1e-3), then analyze.py. Raw independent timings are the two JSONL files per tolerance; summaries retain all three samples. Changing only --worker flags selects a single case. No source edit, commit or push.
