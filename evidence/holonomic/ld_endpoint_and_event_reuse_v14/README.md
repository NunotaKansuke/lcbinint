# LD endpoint budget and same-epoch event-error reuse

Implementation: extend existing frontier value-only endpoint stopping budget from u=0 to nonnegative LD coefficient0<=u<=1. Keep residuals in the same geometry-error ledger; final convergence gate is unchanged. Gradients and unusual LD outside[0,1] retain strict endpoint polishing. Shared default endpoint budget remains0. The sensitivity model is numerical, not a rigorous enclosure.

Event-error memo: optional epoch_event_errors argument to flux_adaptive_integrate. Frontier owns one cache for a single source/lens/tolerance epoch. Retained events keep identical records, local repairs append new events, so previous event estimates can be reused on integration resumes. Changed parameters in the next epoch always get a new cache. Diagnostic-collection mode disables this memo to preserve per-call event diagnostic records. Metadata and previous integral samples are retained.

Fresh regressions1608 adaptive and23 frontier checks (including three new event-cache tests), zero failures. New tests verify cache remains populated without growth on repeat, cached and uncached LD integrals are exactly identical, and duplicate precision checks vanish.

Performance: same87 normal conditions at rtol1e-4,5 repeats,1 thread,fresh objects,seed included,caustic construction/distance excluded. Integral0.34664 ->0.32623 ms (-5.89%); whole0.57846 ->0.55450 ms (-4.14%). LD subset integral0.39271 ->0.35545 ms (-9.49%). Reverse order repeat integral0.34737 ->0.32715 ms (-5.82%),whole0.57714 ->0.55639 ms (-3.59%),LD0.39416 ->0.35695 ms (-9.44%). Compilation never overlaps accepted timing. Radial point counts unchanged. Maximum relative normal192 candidate difference5.47e-14.

Complete896 evaluations, reference-resolved754,failures0;142 unresolved references excluded from precision acceptance. This remains tolerance-matched D14 comparison, not independent truth/global proof. Current Cartesian previous comparator0.7708ms is a separate earlier measurement; don't present it as a contemporaneous A/B.

Source snapshots and hashes are retained. Build accepted kernel using benchmarks/holonomic/build_seeded_radial_master.sh; previous binary preserved as bench_seeded_before_v14. Run initial_minimal_seeds.tsv with OUTPUT 5 0;fold/deeper inputs with OUTPUT 3 0. Same input paths as preceding evidence directory. RawTSVs and summary.json retain all rows.

Instrumented3-rep matched profile: {"all": {"endpoint_ms": 0.018455693486590038, "setup_ms": 0.03806241379310345, "setup_event_ms": 0.034684689655172414}, "0": {"endpoint_ms": 0.018399418439716313, "setup_ms": 0.038626085106382976, "setup_event_ms": 0.035085602836879434}, "0.5": {"endpoint_ms": 0.018521816666666666, "setup_ms": 0.0374001, "setup_event_ms": 0.03421361666666667}}
Profiler timers add overhead; use uninstrumented A/B for speed claims.
