# Current Cartesian and Polar six-panel benchmark

Separate six-panel maps compare the adaptive flood-fill Cartesian and Polar routes with the same matched VBM timings. The cohort contains 7,216 geometries (3,608 each uniform and linear LD), both requested relative tolerances, and no per-case method selection.

Each method time is the median of three cold calls after clearing the source-seed cache. The indexed shared seed builder and adaptive integration are timed together; center-image roots and caustic-geometry construction are outside the timer, with caustic geometry prepared once per input. VBM uses matched saved direct-binding timings, one warm call plus three measured calls per input/target. Thus the ratio includes a small difference in timing setup, but uses the same geometries, LD profiles, and tolerances.

The runner uses Cartesian initial bins 16×16 and Polar initial radial bins 2 with angular ratio 256; both use the indexed `cached_binary_image_seeds` provider and a 1,000,000-map budget. Directional adaptation, local row interpolation, local boundary strips, local vertical caps, and defect correction are explicitly enabled for both routes by the matched diagnostic runner.

The top row of each six-panel figure shows the cell median `R=t_VBM/t_method`; values above one favor that method. The table gives the median both over every finite timing/value and over the subset whose actual difference is within target. The bottom row shows cell p95 absolute relative difference from the adjudicated saved RelTol=1e-6 reference. This agreement measure is empirical, not a formal error certificate. The reference corrections are applied before calculating errors. Cells with fewer than eight finite timings are masked.

All eight speed maps share reciprocal factor-of-two bins, and the 1× category is white. The q-rho cohort is a parameter-space coverage grid rather than an iid speed sample.

| method | profile | target | conditions | finite | estimate accepted | reference pass | false accepts | faster than VBM | median VBM/method | median ratio on passing rows | median method ms | p95 error | max error |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cartesian | uniform | 0.001 | 3608 | 3456 | 3258 | 3443 | 12 | 156/3456 | 0.236x | 0.236x | 0.26926 | 0.0004757 | 0.001726 |
| cartesian | uniform | 0.0001 | 3608 | 3456 | 3202 | 3406 | 26 | 657/3456 | 0.300x | 0.309x | 0.34825 | 7.419e-05 | 0.001624 |
| cartesian | linear | 0.001 | 3608 | 3448 | 3256 | 3443 | 4 | 1666/3448 | 0.958x | 0.957x | 0.30533 | 0.0003721 | 0.001175 |
| cartesian | linear | 0.0001 | 3608 | 3448 | 3211 | 3413 | 18 | 2578/3448 | 3.529x | 3.574x | 0.39304 | 6.515e-05 | 0.001104 |
| polar | uniform | 0.001 | 3608 | 3608 | 3608 | 3608 | 0 | 10/3608 | 0.191x | 0.191x | 0.37934 | 0.0002489 | 0.0005 |
| polar | uniform | 0.0001 | 3608 | 3608 | 3606 | 3608 | 0 | 81/3608 | 0.303x | 0.303x | 0.42321 | 3.091e-05 | 9.153e-05 |
| polar | linear | 0.001 | 3608 | 3608 | 3608 | 3608 | 0 | 1550/3608 | 0.846x | 0.846x | 0.39998 | 0.0002263 | 0.0009388 |
| polar | linear | 0.0001 | 3608 | 3608 | 3606 | 3608 | 0 | 3129/3608 | 3.060x | 3.060x | 0.45735 | 3.071e-05 | 7.342e-05 |

Accuracy pass counts are against the saved adjudicated reference and do not certify all inputs. All measured rows, including rejections, actual-error misses, and exceptions, are in `joined.tsv`.

Benchmark manifest: `/rogue1_8/nunota/lcbinint/tests/diagnostics/results/recal2026/cartesian-polar-sixpanel-current-20261016/full/manifest.json` (HEAD `e27f2663b3f2` plus the exact dirty-source/library hashes in that manifest).
Reproduction: `/home/nunota/.miniconda3/envs/myenv/bin/python tests/diagnostics/recal2026/seed_inclusive_cart_polar_benchmark.py --output tests/diagnostics/results/recal2026/cartesian-polar-sixpanel-current-20261016/full` then `/home/nunota/.miniconda3/envs/myenv/bin/python tests/diagnostics/recal2026/make_cartesian_polar_six_panel_20261016.py --campaign tests/diagnostics/results/recal2026/cartesian-polar-sixpanel-current-20261016/full --output figures/recal2026/cartesian-polar-vbm-six-panel-20261016-current`.
