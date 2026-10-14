# Warm quartic initial-root solve bottleneck

Observed bottleneck: real_root_thetas_warm accepted previous roots after a seed step check <=1e-7, but called Aberth with its default near-machine stopping tolerance. The seed roots are subsequently followed by physical endpoint polishing and geometry-error accounting. The frontier value-only path now passes seed tolerance 1e-10, still tighter than the existing seed acceptance gate. Shared D14 and gradient paths retain seed tolerance zero. No endpoint, count, topology, or integrated error acceptance threshold changed.

Changes: AdaptiveConfig.warm_root_seed_tolerance default0; QuarticWarm.seed_tolerance default0; mapped_radius copies the config into the solver workspace after copying warm anchors; frontier sets1e-10 only for GradientPolicy::None and with_jacobian=false. HOLO_FRONTIER_DISABLE_WARM_SEED_TOLERANCE restores prior behavior.

Matched87 normal conditions at rtol1e-4,5 repeats, fresh objects,1 thread, seed included, caustic construction/distance excluded. A/B: integral0.36986 ->0.34634 ms (-6.36%), whole0.59815 ->0.57412 ms (-4.02%). Reverse order repeat: integral0.37363 ->0.34349 ms (-8.07%), whole0.60405 ->0.57242 ms (-5.23%). No compilation overlapped final timings.

Instrumented kernel (nested stage times, do not add): arc processing0.11205 ->0.08024 ms; root-pair tracking including its quartic fallback0.05694 ->0.04668 ms; endpoint0.03387 ->0.03374 ms; K quadrature0.00905 ->0.00912 ms. Profiler adds overhead; use non-instrumented A/B for speed claims. Warm quartic iterations498.36 ->190.11 per evaluation; cold iterations66.20 unchanged; warm calls49.47 and cold calls4.20 unchanged. So the measured reduction comes from less work in the same seed solves, not omitted nodes or additional cold fallbacks.

Normal192 evaluations: max relative difference from previous candidate1.07e-14; radial point counts unchanged. Shared adaptive1608 checks and frontier20 checks passed with anchor audit. Complete896 evaluations, reference-resolved754, failures0;142 unresolved references excluded. This remains a tolerance-matched D14 comparison, not independent truth or global proof.

Build accepted binary via benchmarks/holonomic/build_seeded_radial_master.sh. Baseline binary saved as build-seeded-events/bench_seeded_radial_before_v12. Profile builds recorded in build_profile.sh/build_profile_before.sh. Run each with initial_minimal_seeds.tsv OUTPUT 5 0; fold/deeper inputs OUTPUT 3 0. Results in summary.json and raw TSVs. Source hashes in manifest.json.
