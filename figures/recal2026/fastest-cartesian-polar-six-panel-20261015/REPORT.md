# Measured faster-of-Cartesian/Polar six-panel comparison

Four six-panel figures cover uniform and linear limb darkening at relative targets 1e-3 and 1e-4. For each geometry and target, the winner is the method with the lower median of three `cold_ms` measurements. `cold_ms` includes the indexed seed builder and adaptive integration; caustic geometry and center-image roots were prepared outside the timed region.

The winner is selected from measured timings only, without using the reference error. This is a retrospective per-case timing oracle, not a deployable run-time selector or a population-wide speed guarantee. The speed ratio is the freshly matched direct VBM time divided by the selected method's seed-inclusive median time; values above 1 favor the selected inverse-ray method. Lower panels show empirical error against the saved RelTol=1e-6 reference with the three adjudications applied.

The source corpus has 7216 geometries and 14432 target conditions. 312 geometries (624 target conditions) raised adaptive lattice evaluation-budget exceptions and have no usable timing result; they are listed in `exclusions.tsv`. All remaining finite winner rows are retained, including estimator rejections and actual-error failures.

| profile | target | valid / total | excluded | Cartesian wins | Polar wins | faster than VBM | median VBM / best | median best (ms) | reference pass | accepted | false accepts | p95 error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform | 0.001 | 3456/3608 | 152 | 2340 | 1116 | 20/3456 | 0.224x | 0.2786 | 3403/3456 | 3356/3456 | 49 | 0.0005536 |
| uniform | 0.0001 | 3456/3608 | 152 | 2231 | 1225 | 415/3456 | 0.326x | 0.3299 | 3336/3456 | 3370/3456 | 112 | 7.961e-05 |
| linear | 0.001 | 3448/3608 | 160 | 2351 | 1097 | 1813/3448 | 1.076x | 0.3105 | 3416/3448 | 3350/3448 | 29 | 0.0004388 |
| linear | 0.0001 | 3448/3608 | 160 | 2277 | 1171 | 2996/3448 | 3.809x | 0.3749 | 3348/3448 | 3363/3448 | 91 | 7.009e-05 |

The q-rho coverage cohort was designed for parameter-space coverage, not as an iid speed sample. Per-cell maps mask cells with fewer than eight finite conditions. The data, exclusions, summary, and provenance are saved alongside the figures. Reproduce with:

`/home/nunota/.miniconda3/envs/myenv/bin/python tests/diagnostics/recal2026/fastest_cartesian_polar_six_panel_20261015.py`
