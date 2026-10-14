# Seeded q-rho benchmark outlier diagnosis and endpoint fix

Date: 2026-10-13  
Checkout: `/rogue1_8/nunota/lcbinint/experiments/seeded-boundary-events`  
Benchmark HEAD: `f3547723b79828d9c501182746028e99f7bad116`

## Finding

The 20 rows above relative error `1e-3` are five geometries repeated across the two limb profiles and two requested tolerances. They are not reference-file or coordinate-transform errors. The original seeded-frontier run reproduces the bad values. On the exact same transformed inputs, the D14 path agrees with the saved VBM `RelTol=1e-6` values to at most `7.1e-6` relative error; standard Cartesian and polar inverse-ray runs at 128/256 bins (512 bins for the first three cases) agree to at most `1.9e-5`.

The seeded radial march initially finds the affected image bands. The endpoint polishing step then replaces a bracket endpoint with a different, physically valid fold outside that endpoint's marching bracket. This collapses or truncates the band. The remaining cells still integrate and pass the internal value-convergence check, so the benchmark writes `status=OK` despite dropping most of the magnification.

| case / epoch | VBM `mu` | seeded `mu` | seeded / VBM | radial bracket before endpoint polish | final radial band | omitted D14-positive region |
|---|---:|---:|---:|---|---|---|
| 26 / 15 | 33.25349 | 1.45198 | 0.0437 | `[0.432715, 0.438393]` | `[0.438393, 0.438393]` | `[0.432715, 0.438393]`, 2 crossings |
| 35 / 15 | 98.79634 | 45.15899 | 0.4571 | `[0.678418, 0.783568]` | `[0.678417, 0.678417]` | `[0.678417, 0.783572]`, 2 crossings |
| 77 / 23 | 71.28937 | 32.36749 | 0.4540 | `[0.887027, 1.313068]` | `[1.194924, 1.313068]` | `[0.887024, 1.194924]`, 2/4 crossings |
| 95 / 23 | 1717.73783 | 67.88039 | 0.0395 | `[3.348626, 3.408022]` | `[3.407888, 3.408025]` | `[3.352740, 3.407888]`, 4 crossings |
| 98 / 23 | 4091.43574 | 7.67752 | 0.00188 | `[0.498584, 2.062711]` | `[2.062711, 2.062711]` | `[0.498581, 2.062500]`, 2/4 crossings |

The most extreme row was case 98 / epoch 23: the old result retained only about 0.188% of the reference magnification. These were five faulty geometries, each repeated across the four profile/tolerance combinations, not 20 independent geometries.

The D14 quartic root-count audit confirms the omitted intervals contain real image arcs. The seed audit also finds seeds within the omitted D14 arcs: 55 for case 26, 1 for case 35, at least 24 for case 77, 54 for case 95, and at least 113 for case 98. This rules out “no representative seed was available” as the explanation for these five failures.

## Code path

In the pre-fix version of `src/lcbinint/magnification/holonomic/seeded_radial_frontier.hpp`, `endpoint()` first tried `local_event()` inside the final marching bracket. When that attempt failed, the fallback called `local_event(R, 0., image_radius_bound, e)`. That let the root search accept any physical fold over the entire image-radius domain; its source-limb residual could be valid while it belonged to another edge or an interior topology event.

The old fallback accepted that event without checking it against the original sign-changing radial bracket. `b.r_lo` or `b.r_hi` was then replaced by that radius. There was no check that the polished endpoints remained ordered and distinct or that the resulting band retained the seed radius. When both endpoints collapsed to the same event, `cuts` contained one unique radius, so no arc cell was added. `endpoints_resolved` stayed true, and the result reported `local_discovery_completed`.

`coverage_certified=false` was present in the result, but the paper runner's `status=OK` only requires the seed provider's support flag and internal value convergence (`benchmarks/holonomic/bench_seeded_paper_qrho.cpp`, lines 141–153). Thus the flag did not prevent these incomplete radial domains from being reported as OK.

## Independent route checks

The focused runner used the same per-row mass swap and x reflection as the original benchmark and reran the seeded result. For each selected row it also called `epoch_adaptive` (the D14 topology route) and the existing preplanned Cartesian and polar inverse-ray methods. The maximum relative differences from saved VBM `mu` across the five selected rows were:

- D14: `7.1e-6`.
- Cartesian inverse ray: `1.9e-5`.
- Polar inverse ray: `1.9e-5`.

These are numerical cross-checks, not a formal exact-error bound. The large seeded deficits are around 54% to 99.8%, so the independent agreement is decisive for identifying the seeded route's missing image area.

## Evidence files

- `diagnostic_routes_all.tsv`: all five geometries, D14 cell/root counts, and seed counts per D14 arc interval.
- `diagnostic_routes.tsv`: Cartesian/polar resolution ladder for cases 26, 35, and 77.
- `diagnostic_routes_remaining.tsv`: resolution ladder for cases 95 and 98.
- `diagnostic_debug_trace.log`: per-seed marching brackets and final endpoint radii; confirms the exact band collapses above.
- `diagnose_outliers.cpp`, `diagnostic_input.tsv`: focused reproduction source and selected inputs.

## Correction applied

In `src/lcbinint/magnification/holonomic/seeded_radial_frontier.hpp:381-408`, endpoint polishing now keeps fallback searches inside the count-changing radius bracket. If the initial local solve fails, the code bisects that bracket for 16 steps and retries only within the tightened interval. If no bracket-qualified event is found, it marks the endpoint unresolved. Before creating arc cells, it also requires the final band to have positive width and to retain at least one of its originating seed radii. The fast path remains unchanged when the bracket-local solve succeeds.

## Post-fix validation

First, the five geometries were rerun through the focused executable at both target tolerances and both limb profiles: all 20 previously bad rows now agree with the saved VBM `RelTol=1e-6` reference to at most `1.073e-5` relative error. Then the patched executable reran the complete 7,216-input corpus, 14,432 target/profile rows, with three repetitions per row. The join matched every expected input key and reference row exactly.

Among the 14,242 rows with internal `status=OK`, the largest relative difference from the saved reference is `5.60693e-4`; zero exceed `1e-3`. The former 20 failures are preserved in `accepted_error_outliers_pre_endpoint_fix.tsv`, and the corrected run's `accepted_error_outliers.tsv` has no data rows. Full-run condition medians are `0.1727–0.2230 ms`, up by only `0.25–0.91%` from the pre-fix medians. The affected individual rows take longer because they now include image area that was previously omitted.

This is a targeted empirical validation, not a formal radial-support proof. The result still reports `coverage_certified=false`, so neither `status=OK` nor this benchmark establishes a universal completeness guarantee. The updated full results, summary, report, and six-panel figures are in the sibling `raw_results.tsv`, `summary.json`, `REPORT.md`, and `figures/` artifacts. The report is generated by `make_figures.py`.
