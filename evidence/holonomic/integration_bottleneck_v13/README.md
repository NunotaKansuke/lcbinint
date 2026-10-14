# Next bottleneck measurements after v12

Instrumentation-only additions: frontier aggregates physical/estimator/scheduler/setup durations across every resumed adaptive call under HOLO_FRONTIER_COST_PROFILE. Previous last-call stats omitted earlier work. v2 profile additionally measures certify_quartic used for per-node dynamic crossing counts. No accuracy/tolerance/solver changes. normal_final.tsv's192 magnifications are bit-identical to v12 after.tsv.

Matched normal87 source/u conditions at rtol1e-4,3 repetitions,1 thread,fresh epoch objects. Diagnostic observations are arithmetic means over all261 repetitions, not medians. All builds completed before timing. Instrumented times include profiler overhead; compare components here, use v12 uninstrumented A/B for speed claims.

Aggregate integral0.38184 ms: physical radial sample evaluation0.21792, estimator0.01691, scheduler0.00138, setup0.04106, other unpartitioned overhead0.10457 ms. The other bucket includes map/anchor/sample management, allocations and control flow; no specific cause has been isolated, so do not call it wasted work or attribute it all to one operation.

Physical sub-timers (nested): arc processing0.08001 (includes root-pair continuation and fallback0.04678), endpoint polish0.03354, dynamic count Sturm0.02841, K routine0.00905. Remaining physical cost is not yet partitioned. These are not an exhaustive additive hierarchy.

Uniform u0 vsLDu.5: endpoint0.01852 vs0.05118 ms; counts0.02793 vs0.02898; arc0.08019 vs0.07979. Their mean node counts from v12 are close (140 vs143), so LD endpoint cost per epoch remains markedly higher after uniform-only endpoint relaxation. This supports targeting LD endpoint precision next, but does not establish a certified allowable relaxation of LD endpoints.

Setup0.04106: event-location uncertainty0.03760, panels0.00207. Average adaptive calls1.241. Event preparation is nearly all setup time; assess whether previously computed event jets and uncertainty can be reused across resumes. Initial fold diagnostic setup0.0398/event0.0369 supports that the fixed setup cost also occurs nearfold cases, but the fold log includes reference-unresolved cases and is diagnostic only.

Priorities: (1) LD endpoint residual-to-flux budget rather than fixed near-machine stopping; (2) retain physical event derivative/error calculations across resumed calls; (3) partition the0.105ms unmeasured overhead before changing memory/scheduling. Dynamic count is0.028ms and cannot explain the entire remaining cost. Estimator and scheduler are small, so they are not currently priority targets.

build_profile.sh reproduces the diagnostic binary. normal_final.log contains PROFILE columns: id,u,tol,arc,rootpair,endpoint,K,warm sweeps,cold sweeps,warm calls,cold calls,physical,estimator,scheduler,setup,event setup,panel setup,adaptive calls,total integral,dynamic count. summary.json groups u0 andu.5. Product build excludes HOLO_FRONTIER_COST_PROFILE.
