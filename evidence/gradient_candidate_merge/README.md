# Integrate computed gradient candidate export

Repair source: MB23169 gradient_candidate_repair_checked, copied upstream README/summary below. Includes regular-radial value repair, computed derivative candidate export on unresolved/contact/prepared-support branches, and native/public/JAX regressions. gradient_valid=False and uncertified reasons are retained. This does not certify derivative accuracy or NUTS convergence.

Rechecked the supplied build against recorded hashes and rebuilt it (all targets already up to date). C++ unit_core passed; all38 unified algebraic public/JAX tests passed. checked/summary.json records the real nonzero local/physical/JAX derivatives and scientific_acceptance=false. source-hashes.json covers the four committed implementation/test files.

Validation uses the separate MB23169 build; no default package switch or MCMC launch was performed. The unrelated evidence/backend_cleanup_20261018/run_tests.py modification is excluded. Merge target: experiment/algebraic_boundary, by fast-forward; remove only experiment/orbital-caustic-support after switching to the parent. No push requested.
