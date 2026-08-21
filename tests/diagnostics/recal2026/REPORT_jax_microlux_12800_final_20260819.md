# Final report: full corrected JAX--microLUX comparison

Date: 2026-08-19
Scope: 3,200 rows × 4 epochs = 12,800 epoch evaluations
Primary result: [`jax_microlux_12800_final_adaptive_v6_20260819/results.json`](../results/recal2026/jax_microlux_12800_final_adaptive_v6_20260819/results.json)

## Executive summary

The corrected implementation was rerun for every JAX row, including rows that
had already succeeded in the earlier run. All 3,200 JAX rows completed, with
zero JAX forward skips and zero JAX `dA/dt` timeouts. The derivative first
compile is included in JAX warm-up metadata and excluded from the steady-state
timing columns.

The original microLUX measurements were reused rather than rerun. They are
available for 3,184 rows; 16 old microLUX rows have no output. Among the
available rows, the original microLUX derivative measurements contain 309
timeouts. Therefore the microLUX win-rate denominators are smaller for the
derivative comparison.

The compiled steady-state winner depends on the profile and tolerance:

| profile | target | forward winner | derivative winner | median `R_micro/JAX` (forward / dA/dt) |
|---|---:|---|---|---:|
| uniform | `1e-3` | microLUX: 67.5% | microLUX: 76.8% | `0.389 / 0.264` |
| uniform | `1e-4` | microLUX: 88.9% | microLUX: 93.3% | `0.278 / 0.229` |
| linear, `c=0.5` | `1e-3` | JAX: 57.5% | JAX: 52.4% | `1.637 / 1.157` |
| linear, `c=0.5` | `1e-4` | JAX: 57.1% | microLUX: 57.2% | `1.305 / 0.726` |

Here `R_micro/JAX = t_microLUX / t_JAX`; values above one favor JAX.
These are measured corpus statistics, not a universal runtime law.

## Benchmark and timing definition

The corpus contains 160 binary-lens cases, five `d/rho` values
(`0.2, 0.6, 1.0, 1.4, 1.8`), two source profiles, and two target errors:

| profile | limb-darkening coefficient | target |
|---|---:|---:|
| uniform | `c=0` | `1e-3`, `1e-4` |
| linear | `c=0.5` | `1e-3`, `1e-4` |

The reported JAX and microLUX times are four-epoch block wall times after the
callable has been compiled and warmed. Native timing is the sum of the four
saved native `chosen_seconds` values. The native source corpus does not contain
native derivative timings, so the native comparison is forward-only.

For the original microLUX path, linear rows use the default `n_annuli=10` and
the configured relative tolerance. Uniform rows do not use a limb-darkening
annulus parameter.

## Compiled steady-state timings

Times are seconds per four-epoch block. JAX p50/p90/max are shown; microLUX p50
uses only rows with an available original microLUX measurement.

| profile | target | JAX forward p50 / p90 / max | microLUX forward p50 | JAX dA/dt p50 / p90 / max | microLUX dA/dt p50 |
|---|---:|---:|---:|---:|---:|
| uniform | `1e-3` | `0.0235 / 0.2826 / 0.6867` | `0.00621` | `0.0410 / 0.4811 / 1.5717` | `0.00933` |
| uniform | `1e-4` | `0.0419 / 0.4655 / 2.8302` | `0.00844` | `0.0782 / 0.8482 / 4.9649` | `0.0147` |
| linear | `1e-3` | `0.0345 / 0.3933 / 2.1718` | `0.0380` | `0.0673 / 0.6529 / 3.5783` | `0.0586` |
| linear | `1e-4` | `0.0533 / 0.6222 / 7.5650` | `0.0575` | `0.1765 / 1.3707 / 26.1535` | `0.1066` |

Detailed win counts and ratios are stored in
`summary.<profile>:target=<target>.win_rates_and_median_ratios` in the JSON.

## JAX versus native

The corrected JAX run honors the saved native `chosen_grid` and `chosen_nbin`
where present. The native block is faster when `R_native/JAX < 1`.

| profile | target | matched rows | JAX wins | native wins | median `R_native/JAX` |
|---|---:|---:|---:|---:|---:|
| uniform | `1e-3` | 800 | 22.1% | 77.9% | `0.402` |
| uniform | `1e-4` | 784 | 43.8% | 56.3% | `0.911` |
| linear | `1e-3` | 799 | 9.0% | 91.0% | `0.254` |
| linear | `1e-4` | 791 | 17.2% | 82.8% | `0.503` |

The native comparison is a saved-plan throughput comparison, not a claim that
the native plan is accurate for every row.

## Native versus VBM timing context

The original native speed-discovery corpus supplies the following epoch-level
VBM/native timing context:

| profile | target | VBM win rate | median `R_VBM/native` |
|---|---:|---:|---:|
| uniform | `1e-3` | 99.91% | `0.0962` |
| uniform | `1e-4` | 99.97% | `0.0291` |
| linear | `1e-3` | 66.68% | `0.5819` |
| linear | `1e-4` | 69.15% | `0.4708` |

## Accuracy qualification

Accuracy is reported separately from speed. The JAX values are compared with
the target-specific VBM reference.

| profile | target | JAX within target | native plan passes / fails / missing | maximum JAX relative error | JAX failures on native-pass rows |
|---|---:|---:|---:|---:|---:|
| uniform | `1e-3` | 800/800 | 704 / 96 / 0 | `9.988e-4` | 0 |
| uniform | `1e-4` | 799/800 | 677 / 107 / 16 | `4.281e-4` | 0 |
| linear | `1e-3` | 780/800 | 610 / 189 / 1 | `4.780e-2` | 0 |
| linear | `1e-4` | 788/800 | 508 / 283 / 9 | `3.261e-2` | 0 |

There are 33 raw JAX target failures in total. None occurs on a row whose
saved native plan passes the same VBM target: 31 are on native plans that
already fail their target, and 2 are on rows with no native plan. The two
missing-plan failures are the low-`q` case 10 at `d/rho=0.6` for the two
profiles; the uniform row has a maximum relative error of `4.281e-4` at the
`1e-4` target. This residual is not evidence of a JAX-only regression; it is
an unresolved cross-method accuracy limitation for that missing-plan input.

The linear-profile raw failures are retained in the result rather than hidden
behind the speed statistics. A paper making a strict `1e-3` or `1e-4` accuracy
claim should either restrict the analysis to target-qualified rows or report
these outliers explicitly.

## Route and convergence audit

The final run used the following global policy:

1. Use the saved native route and radial bin count whenever a native plan is
   present.
2. Validate mixed Cartesian/polar plans as complete plans; do not validate a
   partial polar subset against a Cartesian reference.
3. Use Cartesian boundary subdivision 4 by default. Subdivision 8 is selected
   only when a native-certified plan misses the target at subdivision 4; three
   rows used subdivision 8 in the final run.
4. For missing native plans, use generic route-capacity diagnostics. When a
   Cartesian route reaches the configured maximum source-bin capacity, use a
   polar route at the same radial resolution. There is no case-, `q`-, or
   `d/rho`-specific geometry skip.
5. Use the polar angular ladder from 65,536 to 2,097,152 bins. A first-rung
   target pass is accepted; non-monotone misses are allowed to recover; a
   numerically stable miss is recorded as unresolved rather than searched
   indefinitely.
6. Keep the direct numerical target result separate from the conservative
   shared-caustic support certificate.

Across the 3,200 rows:

| item | count |
|---|---:|
| rows using saved native route/nbin | 3,174 |
| rows using JAX calibrated route because native plan is missing | 26 |
| rows containing at least one polar epoch | 1,558 |
| rows entirely Cartesian | 1,642 |
| polar epochs | 4,432 / 12,800 |
| Cartesian epochs | 8,368 / 12,800 |
| fully certified JAX rows | 3,106 |
| rows with `jax_support_valid=false` | 64 |
| rows with support valid but not fully certified | 30 |
| discovery-overflow rows | 0 |

The 64 support-invalid rows are associated with the conservative root/support
certificate (`jax_root_failure=true`); they are not timing failures. The
certificate state is retained in `jax_support_valid` and
`jax_certified_pass`, independently of the direct VBM relative-error result.

## Reproduction artifacts

- Full merged result: [`jax_microlux_12800_final_adaptive_v6_20260819/results.json`](../results/recal2026/jax_microlux_12800_final_adaptive_v6_20260819/results.json)
- Full uniform `1e-3` lane: [`results.json`](../results/recal2026/jax_microlux_12800_full_uniform_1e3_adaptive_v6_20260818/results.json)
- Full uniform `1e-4` lane: [`results.json`](../results/recal2026/jax_microlux_12800_full_uniform_1e4_adaptive_v6_20260818/results.json)
- Full linear `1e-3` lane: [`results.json`](../results/recal2026/jax_microlux_12800_full_linear_1e3_adaptive_v6_20260818/results.json)
- Full linear `1e-4` lane: [`results.json`](../results/recal2026/jax_microlux_12800_full_linear_1e4_adaptive_v6_20260818/results.json)
- Benchmark harness: [`bench_jax_microlux_12800.py`](bench_jax_microlux_12800.py)
- JAX-only rerun driver: [`retry_failed_jax_only.py`](retry_failed_jax_only.py)
- Merge/statistics driver: [`merge_corrected_jax_microlux.py`](merge_corrected_jax_microlux.py)

The old combined JAX artifact is not used as a timing source. Its microLUX
fields are used only as the original microLUX measurement base; every JAX row
in the final artifact comes from the four full corrected lanes above.

## Limitations

This is a compiled repeated-kernel benchmark, not an end-to-end inference
benchmark. It excludes JAX compilation and route warm-up from steady-state
times, does not rerun microLUX, and has no native derivative timing corpus.
The native/VBM timing table is retained as context from the original native
speed-discovery run.
