# Synthetic light-curve API benchmark

合成パラメータのみで測定した `lcbinint` / VBMicrolensing 比較。

- in-tree Release build, OpenMP thread setting is taken from the process environment
- tolerance: `0.001` (both engines), 240 default epochs/case; overrides: close_secondary_caustics=400, wide_planet=600
- each row: first call + `5` steady calls; steady is the median of the latter
- binary warm-up setup time is included only in `warmup.extra_ms` and the measured-call total
- triple warm-up probe: `NotImplementedError: the first warm-up implementation supports binary lenses`

## Binary

| case | profile | lcbinint no warm [ms/epoch] | lcbinint warm [ms/epoch] | warm setup [ms] | VBM [ms/epoch] | VBM/lc no-warm | VBM/lc warm | max rel. err no/warm |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `resonant_high_mag` | C0_uniform | 0.657 | 0.139 | 2924 | 0.05055 | 0.07694 | 0.3637 | 0.000382 / 0.000943 |
| `resonant_high_mag` | C1_linear_ld | 0.864 | 0.13 | 3437 | 0.2484 | 0.2875 | 1.91 | 0.000557 / 0.00121 |
| `resonant_large_source` | C0_uniform | 0.9829 | 0.1966 | 3818 | 0.06611 | 0.06726 | 0.3363 | 0.000411 / 0.00101 |
| `resonant_large_source` | C1_linear_ld | 1.189 | 0.2134 | 4245 | 0.5927 | 0.4985 | 2.777 | 0.000558 / 0.00139 |
| `close_binary` | C0_uniform | 0.08299 | 0.01993 | 285.4 | 0.03003 | 0.3618 | 1.506 | 7.55e-05 / 9.1e-05 |
| `close_binary` | C1_linear_ld | 0.1207 | 0.01822 | 338.1 | 0.02538 | 0.2103 | 1.393 | 0.000313 / 0.000343 |
| `close_secondary_caustics` | C0_uniform | 0.0469 | 0.01228 | 680.1 | 0.02229 | 0.4753 | 1.816 | 0.000323 / 0.00067 |
| `close_secondary_caustics` | C1_linear_ld | 0.0347 | 0.009371 | 566.9 | 0.05202 | 1.499 | 5.552 | 0.00052 / 0.000983 |
| `wide_planet` | C0_uniform | 0.01987 | 0.006752 | 796.2 | 0.005583 | 0.281 | 0.8268 | 9.22e-05 / 0.000593 |
| `wide_planet` | C1_linear_ld | 0.02272 | 0.006971 | 946.2 | 0.009521 | 0.4191 | 1.366 | 0.000379 / 0.000663 |
| `small_q` | C0_uniform | 0.004027 | 0.002799 | 1.52 | 0.005395 | 1.34 | 1.928 | 3.02e-05 / 3.02e-05 |
| `small_q` | C1_linear_ld | 0.004007 | 0.002898 | 1.477 | 0.005611 | 1.4 | 1.936 | 0.000277 / 0.000277 |
| `high_q` | C0_uniform | 0.06397 | 0.02307 | 319.3 | 0.04303 | 0.6727 | 1.865 | 0.00108 / 0.000965 |
| `high_q` | C1_linear_ld | 0.05028 | 0.01592 | 287.6 | 0.06878 | 1.368 | 4.321 | 0.000701 / 0.000962 |
| `cusp_small_source` | C0_uniform | 0.1291 | 0.03143 | 544.2 | 0.03167 | 0.2452 | 1.008 | 0.00025 / 0.00068 |
| `cusp_small_source` | C1_linear_ld | 0.1763 | 0.03384 | 665.6 | 0.04257 | 0.2415 | 1.258 | 0.00047 / 0.000907 |

## Triple

| case | profile | lcbinint [ms/epoch] | VBM [ms/epoch] | VBM/lc | max rel. err |
|---|---|---:|---:|---:|---:|
| `triple_default` | C0_uniform | 1.917 | 0.2002 | 0.1044 | 0.000424 |
| `triple_default` | C1_linear_ld | 1.874 | 0.4302 | 0.2296 | 0.000702 |
| `triple_close` | C0_uniform | 0.4398 | 0.2394 | 0.5444 | 0.000179 |
| `triple_close` | C1_linear_ld | 0.4363 | 0.2834 | 0.6497 | 0.00476 |
| `triple_wide` | C0_uniform | 0.707 | 0.1836 | 0.2597 | 1.61e-05 |
| `triple_wide` | C1_linear_ld | 0.7247 | 0.211 | 0.2911 | 0.000136 |

## Route counts

- `binary/resonant_high_mag/C0_uniform`: point_source=59, hexadecapole=93, inverse_ray_cartesian=86, inverse_ray_polar=2
- `binary/resonant_high_mag/C1_linear_ld`: point_source=59, hexadecapole=93, inverse_ray_cartesian=86, inverse_ray_polar=2
- `binary/resonant_large_source/C0_uniform`: hexadecapole=54, inverse_ray_cartesian=186
- `binary/resonant_large_source/C1_linear_ld`: hexadecapole=54, inverse_ray_cartesian=186
- `binary/close_binary/C0_uniform`: point_source=176, hexadecapole=59, inverse_ray_cartesian=5
- `binary/close_binary/C1_linear_ld`: point_source=176, hexadecapole=59, inverse_ray_cartesian=5
- `binary/close_secondary_caustics/C0_uniform`: point_source=295, hexadecapole=88, inverse_ray_cartesian=17
- `binary/close_secondary_caustics/C1_linear_ld`: point_source=295, hexadecapole=88, inverse_ray_cartesian=17
- `binary/wide_planet/C0_uniform`: point_source=562, hexadecapole=27, inverse_ray_cartesian=11
- `binary/wide_planet/C1_linear_ld`: point_source=562, hexadecapole=27, inverse_ray_cartesian=11
- `binary/small_q/C0_uniform`: point_source=228, hexadecapole=12
- `binary/small_q/C1_linear_ld`: point_source=228, hexadecapole=12
- `binary/high_q/C0_uniform`: point_source=111, hexadecapole=107, inverse_ray_cartesian=22
- `binary/high_q/C1_linear_ld`: point_source=111, hexadecapole=107, inverse_ray_cartesian=22
- `binary/cusp_small_source/C0_uniform`: point_source=162, hexadecapole=65, inverse_ray_cartesian=13
- `binary/cusp_small_source/C1_linear_ld`: point_source=162, hexadecapole=65, inverse_ray_cartesian=13
- `triple/triple_default/C0_uniform`: hexadecapole=182, inverse_ray_cartesian=58
- `triple/triple_default/C1_linear_ld`: hexadecapole=182, inverse_ray_cartesian=58
- `triple/triple_close/C0_uniform`: hexadecapole=216, inverse_ray_cartesian=24
- `triple/triple_close/C1_linear_ld`: hexadecapole=216, inverse_ray_cartesian=24
- `triple/triple_wide/C0_uniform`: hexadecapole=240
- `triple/triple_wide/C1_linear_ld`: hexadecapole=240
