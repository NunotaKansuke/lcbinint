# Polar flood-fill cost-cut experiment

Date: 2026-10-16

Branch: `feature/polar-cost-cut`

Frozen comparison base: `e27f2663b3f2cebc3f3a6bf1196f7b2268f7e197`

## Candidate change

The adaptive Polar support-discovery flood fill now starts with at most two radial bins and caps its coarse angular spacing ratio at 6. The retained support is still checked by the existing image, footprint, and boundary witnesses; the adaptive integration and error-estimation code is unchanged.

The LightCurve auto caller already requested two Polar bins on the frozen base, so the radial-bin cap benefits fixed-grid Polar callers but does not change that auto setting. The auto-path experiment measures the angular-ratio change.

## Matched fixed-Polar comparison

The 7,216-geometry cohort was run at both relative targets (14,432 conditions total), with uniform and linear limb darkening, one CPU, and alternating baseline/candidate executable order. The baseline is the exact frozen e27 runner.

| profile | target | median speed ratio | geometric speed ratio | median map ratio | candidate rejects |
|---|---:|---:|---:|---:|---:|
| uniform | 1e-3 | 1.081x | 1.208x | 1.231x | 0 |
| uniform | 1e-4 | 1.070x | 1.191x | 1.200x | 2 |
| linear | 1e-3 | 1.080x | 1.209x | 1.211x | 0 |
| linear | 1e-4 | 1.067x | 1.192x | 1.175x | 2 |

Across all conditions, map evaluations fell from 233,901,246 to 131,671,281 (43.7%). There were no new reference disagreements relative to the frozen runner. The candidate rejected two strict 1e-4 evaluations in each LD profile. Reference-disagreement counts changed from 2 to 1 for uniform LD and remained 2 for linear LD.

Full run: `tests/diagnostics/results/recal2026/polar-speed-20261014/polar-cost-cut-full-initial2-gridratio6-v1-20261016/`.

## Matched public LightCurve auto comparison

Both versions used the same 7,216 geometries, epochs, tolerances, VBM timing records, CPU affinity, and public `LightCurve.info` vector-call timing boundary. The baseline binary was rebuilt from the frozen e27 source before the candidate binary was restored and rebuilt.

Across 3,608 timed vector batches, baseline/candidate time had a 1.008x median ratio and 1.023x geometric-mean ratio. For batches whose selected method was entirely Polar, median ratios ranged from 1.017x to 1.022x. This is a modest whole-auto improvement; the per-batch q10-to-q90 range was 0.969x to 1.113x, so it is not evidence of a universal per-case speedup. Method selection was identical. There were zero new reference exceedances; all auto false accepts remained on the pre-existing Cartesian path. The candidate failed closed on one additional condition in each LD profile at target 1e-4.

The candidate-vs-VBM six-panel figures and full data are in `figures/recal2026/lightcurve-auto-vbm-six-panel-20261016-polar-cost-cut-gridratio6-v1/`.

The route-selection cohort is not an independent benchmark holdout: all 768 rows in `tests/diagnostics/results/recal2026/predictive-route-20261015/selected-inputs.tsv` also occur in the 7,216-row auto cohort, with `(s,q,rho,x,y,c)` equal at 12 decimal places. The auto timing above therefore describes these already-seen geometries and fixed route choices; it does not establish route-selector generalization or an independent winner fraction. The exact join and input hashes are in `tests/diagnostics/results/recal2026/polar-cost-cut-20261016/fixed-cost/selector-auto-overlap.json`.

## Resolution-independent fixed cost

The exact binary-image seed cache is shared by Cartesian and Polar. Its old lookup scanned up to 1,024 entries and shifted the vector on eviction. The candidate keeps a short linear path, uses a ring for FIFO eviction, and creates a fixed-size open-addressing index only after a cache with at least 64 entries records a real hit. It then adds keys to the index as they are reused. A cold stream of distinct epochs does not allocate or populate the index.

Matched warm lookup improved 1.45x at 64 entries, 4.47x at 256, and 14.19x at 1,024. Through 48 entries, the absolute change was within about 3 ns per hit. A fresh-process pass including seed generation and first exact replay stayed within 0.9–1.2% of the vector baseline. These are cache microbenchmarks, not whole-LightCurve speedups. The API check passed 38/38, and a 2,048-position probe confirmed exact identity for the 1,024 entries retained after eviction. Full paired data and scope are in [the fixed-cost report](tests/diagnostics/results/recal2026/polar-cost-cut-20261016/fixed-cost/FIXED_COST_REPORT.md).

## Accuracy and recovery checks

- Independent Polar references: 798 conditions, 797 accepted, 0 false accepts, 0 contract failures, 0 exceptions; 3 references marked uncertain.
- Public/API recovery checks: 38 passed, 0 failures.
- Full matched auto cohort: 14,432 conditions, 0 new reference exceedances.

These are empirical checks, not a formal error certificate.

## Rejected settings

- One radial bin produced a false accept on geometry 3694 at target 1e-4: relative error `6.29e-4` with an estimated error `1.16e-4`.
- Angular ratio 7 (and 6.5) left geometry 6460 falsely accepted at target 1e-3 with relative error `1.82e-3`. Ratio 8 added two more near-threshold false accepts. These settings were rejected.
- Polynomial sign-reuse prototypes reduced map counts but did not improve wall time, so those code changes were reverted.

## Reproduction

- Fixed Polar: `python tests/diagnostics/recal2026/polar_cost_cut_full_campaign_20261016.py --name polar-cost-cut-full-initial2-gridratio6-v1-20261016`
- Independent validation: `python tests/diagnostics/recal2026/polar_speed_validation.py --name polar-cost-cut-gridratio6-v1-reference-20261016`
- API checks: `POLAR_CHECK_OUTPUT_DIR=/rogue1_8/nunota/lcbinint/tests/diagnostics/results/recal2026/polar-cost-cut-20261016/api-check-gridratio6-v1 python tests/diagnostics/recal2026/polar_local_adaptive_check.py`
