# Seed scan and quartic count optimization (v26)

## Accepted result

Single thread, same 87 normal `(id,u)` pairs at tolerance 1e-4, five repetitions and mean of per-condition medians. Every repetition creates fresh magnifiers/workspaces. Caustic construction is excluded: it is paid once per orbit. No compilation ran during accepted timing. Reverse order repeated.

| Stage | Before ms | After ms | Reduction |
|---|---:|---:|---:|
| seed | 0.100891 | 0.084528 | 16.2% |
| radial planning | 0.098932 | 0.090615 | 8.4% |
| integration | 0.222338 | 0.216180 | 2.8% |
| entire path, excluding caustic construction | 0.424766 | 0.393725 | 7.3% |

Reverse order: seed 16.1%, planning 8.4%, integration 1.4%, entire path 6.4%. Attribute the larger total reduction cautiously: integration varies with timing order. Seed and planning gains repeat.

Instrumented count cost: frontier probes 0.043627→0.035491 ms (18.7%), integration count checks 0.028350→0.021285 ms (24.9%). These timers are nested and not additive to the production timings. The frontier timer wraps `has_image`, not every count in repair. Synthetic quartic count benchmark: 136.599→105.449 and reverse 136.353→105.388 ms, about 23% reduction, 600000 evaluations per timed pass.

## Implementation

1. Seed getter uses exactly the master probe/root/seed append logic. Its support descriptor caller does not consume nearest-point, contact-list, or arc-list outputs of `scan_caustic_branches`. A new explicit `support_only` argument omits these outputs. Segment projection remains necessary for blind disk crossings; it is screened using both endpoint distances and conservative axis-aligned bounding boxes. Disk bounding box coordinates are rounded outward with `nextafter`. Every vertex extremum and possible blind crossing retains historical refinement and order. Ordinary scan consumers retain the original default behavior. Caustic construction/cache and derivative interfaces are unchanged.
2. Quartic Sturm chain: fixed-degree remainder kernels for the regular 4→3→2→1→0 chain. Same division, subtraction order, trimming, sign margins and precision escalation. Degree drops resume the generic chain. `HOLO_STURM_GENERIC_CHAIN` provides the generic chain for diagnostics.
3. Cauchy chart comparison: max of coefficient magnitudes divided by the leading magnitude, rather than maximum of four separate ratios. Finite positive division is monotone; chart decisions match. The two chart bounds require two divisions instead of eight. Boundary quartic coefficient construction itself retains the prior long-double arithmetic.

Seed diagnostics before optimization: seed 0.097834 ms, certified probe root solves 0.040411 ms, approximately 26.46 solves, 112.75 returned image seeds per selected case. `certify_seconds` 0.000618 ms does not include the full preceding polyline scan. Diagnostic clocks add overhead; use matched normal runs for accepted speed claims.

## Validation

- 224 source/lens geometries: seed coordinates in hexadecimal floating point, ordering and support flag byte-for-byte identical.
- 896 normal/fold/deeper integration conditions: magnification and node counts exactly identical before/after. Rebuilt before baseline also agrees with v25 values for all conditions.
- 754 converged D14 references: zero candidate failures or tolerance violations. Maximum relative discrepancy / tolerance 0.249. The other 142 unresolved D14 references are excluded from independent accuracy acceptance.
- 200000 random/degenerate quartics over scales 1e-300…1e300: count, certification, precision tier, chart selection, coefficients and sign margin exactly equal to the saved pre-change implementation.
- Existing suites: adaptive 1608, frontier 23, interpolation 43680, known-root quartics 3000; zero failures. Adaptive suite includes gradient policies and resource budgets. Gradient performance was not newly measured.

This is equivalence evidence over the recorded cases, not a new universal support proof. Essential probe solves and seed positions have not been reduced.

## Reproduction / provenance

Use the existing experimental checkout. `seed_before.cpp` pins the shared master's source at this measurement; `seed_after.cpp` and `support_only_scan.patch` contain the local modification. Shared root checkout was not edited. The standard experimental build helper now explicitly links the optimized provider object; `bench_seeded_radial_master` was updated to the accepted binary. Source snapshots, archive hashes and timing contract are in `manifest.json`.

Run `build.sh`, `build_audit.sh`, `build_count.sh`, `build_profile.sh`, and `build_tests.sh`, finish compilation, then `run.sh`. Profile runs are separate, three repetitions. `prepare.py` uses the pinned baseline once it exists, so rebuilding does not silently adopt later shared changes. `quartic_before.hpp` / `quartic_after.hpp` pin the count implementations. Raw outputs and `summary.json` retain all measurements. No commits or pushes.
