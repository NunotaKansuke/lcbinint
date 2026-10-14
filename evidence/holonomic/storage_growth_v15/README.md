# Sample storage growth

Observed overhead: radial refine reserved exactly the next required number of AdaptiveSample objects; subsequent refinements reallocated and copied all existing samples. Existing opt-in HOLO_ADAPTIVE_SAMPLE_GROWTH policy is now the default, disabled by HOLO_ADAPTIVE_DISABLE_SAMPLE_GROWTH for matched A/B. Reserve up to1.5 times the next required count, capped by max_node_evals. Only enabled when the configured maximum samples plus maximum panels fit in the memory budget, with no Hermite sidecar. Otherwise exact-size behavior remains. Numerical ordering, samples, precision, convergence and budgets unchanged; no upfront maximum allocation.

Diagnostic additions under HOLO_FRONTIER_COST_PROFILE aggregate allocation time/count across all adaptive resumes. Normal87 cases: allocation+copy0.05202 ->0.02370 ms; reallocations10.59 ->5.18. Instrumented clocks are for attribution, uninstrumented A/B determines performance.

Matched normal87 cases,tol1e-4,5 repeats,freshobjects,1thread,seed included,caustic construction/distance excluded: integral0.32875 ->0.30183 ms (-8.19%); whole0.55720 ->0.53140 ms (-4.63%). Baseline/current compiled with identical flags and current master seed library. All builds complete before timing. Normal192 magnifications and radial node counts bit-identical.

Shared adaptive1608 checks and frontier23 checks pass with anchor audit. Complete896-case comparison reports754 D14-converged references, zero candidate failures;142 reference-unresolved excluded. Independent truth/global proof not claimed.

build accepted binary: benchmarks/holonomic/build_seeded_radial_master.sh. Saved baseline bench_seeded_before_v15. Profile scripts build_profile.sh and build_profile_before.sh. Run initial_minimal_seeds.tsv with OUTPUT 5 0,fold/deeper inputs with OUTPUT 3 0, same input paths as previous artifacts. Raw TSVs, summary.json, copied sources and hashes retain provenance.

Reverse-order repeat: {"before_repeat.tsv": {"candidate_integration_ms": 0.3282424367816092, "candidate_no_caustic_ms": 0.5578754942528735}, "after_repeat.tsv": {"candidate_integration_ms": 0.30174188505747124, "candidate_no_caustic_ms": 0.5317521379310345}}
