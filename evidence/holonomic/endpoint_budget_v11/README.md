# Endpoint stopping tied to flux tolerance

New frontier value-only uniform-source path reserves 1% of relative tolerance for each endpoint's angular residual relative to its arc width. Stops before an additional Newton update when abs(phi / phi_theta) <= 0.01 * mu_rtol * arc_width. The unchanged geometry ledger still receives the actual residual plus existing lens-map arithmetic sensitivity; the stopping tolerance is not substituted as a claimed rigorous bound. Derivative-small rejection, arc-width resolution, actual integrated geometry error and final convergence gates remain active. This is a numerical residual estimate, not a mathematical enclosure.

AdaptiveConfig.uniform_endpoint_budget_fraction defaults to zero. Frontier enables 0.01 only for GradientPolicy::None and with_jacobian=false; mapped_radius applies it only for u=0. Gradients and limb darkening retain strict endpoints. HOLO_FRONTIER_DISABLE_ENDPOINT_BUDGET gives matched strict baseline. Source default trial initially failed 2 primal snapshot regression checks, so the change was scoped to frontier; initial non-final TSVs and regression.log are rejected/intermediate artifacts.

Fresh final regressions: 1608 adaptive and 20 frontier checks, zero failures, anchor audit enabled. Matched normal 87 source/u cases, tol1e-4, five repetitions per condition; timings are per-condition medians averaged over conditions, no geometry reuse, one thread, caustic construction/distance excluded, seeds included. Builds completed before final timing.

Uniform: integration 0.36304 -> 0.33020 ms (-9.05%); whole 0.59877 -> 0.56805 ms (-5.13%). Combined u=0 and .5: integration 0.38973 -> 0.37212 ms (-4.52%); whole 0.61844 -> 0.60215 ms (-2.63%). Limb darkening unchanged (0.42109 vs0.42139 ms integration). Radial point counts unchanged. Maximum relative candidate difference from strict endpoints over all normal192 evaluations 6.77e-14.

Complete campaign896 evaluations: resolved D14 reference754, candidate failures0; unresolved reference142 excluded from accuracy acceptance. These comparisons do not replace independent truth validation or establish a global support proof.

Build accepted kernel with benchmarks/holonomic/build_seeded_radial_master.sh. Strict A/B build: build_strict.sh. Run either binary on evidence/holonomic/caustic_seeded_connection_v4/{initial,fold,deeper}_minimal_seeds.tsv with OUTPUT 5 0 (normal) or OUTPUT 3 0 (holdouts). Final TSVs and summary.json contain raw observations.
