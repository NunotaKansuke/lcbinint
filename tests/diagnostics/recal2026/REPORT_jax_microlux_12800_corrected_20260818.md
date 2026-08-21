# Superseded report

This 2026-08-18 draft contains the pre-full-rerun statistics. Use the final
full-lane report instead:

[`REPORT_jax_microlux_12800_final_20260819.md`](REPORT_jax_microlux_12800_final_20260819.md)

# Paper-facing report: corrected JAX--microLUX finite-source speed comparison

Date: 2026-08-18
Primary dataset: [`jax_microlux_12800_corrected/results.json`](../results/recal2026/jax_microlux_12800_corrected/results.json)
Benchmark harness: [`bench_jax_microlux_12800.py`](bench_jax_microlux_12800.py)
Comparison: lcbinint JAX backend, lcbinint native warm-up plan, and microLUX

## Executive conclusion

The compiled steady-state winner depends on the source profile.

- For a uniform source, microLUX is faster in most matched cases: 65.8--78.8%
  of forward evaluations and 72.6--82.9% of measured derivative evaluations.
- For linear limb darkening with `c=0.5`, JAX is faster in most matched cases:
  61.5--65.9% for the forward calculation and 53.5--57.7% for `dA/dt`.
- The median per-case ratio `R = t_microLUX / t_JAX` is below one for the
  uniform profile (`R = 0.350--0.392`) and above one for the linear profile
  (`R = 1.200--2.458`). Thus the linear-limb-darkening JAX path is typically
  faster in this corpus, while the uniform-source microLUX path is typically
  faster.
- The comparison is a cache/compile-warm kernel comparison. Compilation and
  route-resolution warm-up are excluded from the reported steady-state block
  times.

These conclusions describe the measured parameter corpus; they are not a
universal runtime law for every binary-lens geometry or source profile.

## Dataset and timing definition

The benchmark contains 160 binary-lens cases, five source-size/distance lanes
(`d/rho = 0.2, 0.6, 1.0, 1.4, 1.8`), two source profiles, and two target
relative errors. Each row contains four reference epochs, giving 3,200 JAX
rows and 12,800 measured epoch evaluations.

The profiles and targets are:

| profile | limb-darkening coefficient | target |
|---|---:|---:|
| uniform | `c=0` | `1e-3`, `1e-4` |
| linear | `c=0.5` | `1e-3`, `1e-4` |

The primary timing quantity is the four-epoch block wall time after the JAX
callable has been compiled and warmed. The reported per-case win is decided
from this block time. For the native comparison, the native block time is the
sum of the four saved native `chosen_seconds` values. Native `dA/dt` timing was
not present in the source warm-up corpus, so the native comparison is
forward-only.

The ratios are defined as follows:

```text
R_micro/JAX  = t_microLUX / t_JAX
R_native/JAX = t_native / t_JAX
R_VBM/native = t_VBM / t_native
```

For the first two ratios, values larger than one mean that JAX is faster. For
`R_VBM/native`, values smaller than one mean that VBM is faster. The reported
median is the median of per-case ratios, not the ratio of the two aggregate
medians.

## JAX versus microLUX

The timing columns are p50 block times in seconds. Win rates are shown as
`JAX / microLUX`; `N` is the number of matched rows used for that metric.

| profile | target | forward p50 JAX / microLUX | forward win rate, `N` | median `R_micro/JAX` | `dA/dt` p50 JAX / microLUX | derivative win rate, `N` | median `R_micro/JAX` |
|---|---:|---:|---:|---:|---:|---:|---:|
| uniform | `1e-3` | `0.0192 / 0.00621` | `34.3% / 65.8%`, 800 | 0.392 | `0.0402 / 0.00933` | `27.4% / 72.6%`, 720 | 0.251 |
| linear | `1e-3` | `0.0225 / 0.0380` | `61.5% / 38.5%`, 800 | 2.148 | `0.0540 / 0.0586` | `53.5% / 46.5%`, 720 | 1.200 |
| uniform | `1e-4` | `0.0335 / 0.00844` | `21.2% / 78.8%`, 793 | 0.350 | `0.0724 / 0.0147` | `17.1% / 82.9%`, 718 | 0.256 |
| linear | `1e-4` | `0.0366 / 0.0575` | `65.9% / 34.1%`, 791 | 2.458 | `0.1037 / 0.1066` | `57.7% / 42.3%`, 717 | 1.503 |

The reduced derivative sample counts are caused by the original microLUX
source data containing derivative timeouts. Those microLUX rows were not
rerun in this correction pass.

## JAX versus native

The native speed-discovery corpus supplies a forward timing for each reference
epoch. Summing those four values gives the native block used here.

| profile | target | matched rows | forward win rate, JAX / native | median `R_native/JAX` |
|---|---:|---:|---:|---:|
| uniform | `1e-3` | 800 | `32.9% / 67.1%` | 0.419 |
| linear | `1e-3` | 799 | `24.2% / 75.8%` | 0.311 |
| uniform | `1e-4` | 784 | `55.9% / 44.1%` | 1.319 |
| linear | `1e-4` | 791 | `51.3% / 48.7%` | 1.050 |

The tight uniform lane is the only lane where JAX has both a majority forward
win and a median ratio above one relative to native. The linear `1e-4` lane is
close to parity by both measures.

## Native versus VBM reference

For context, the original native speed-discovery corpus also contains the VBM
timing ratio. This is an epoch-level comparison and is independent of the JAX
steady-state timing pass.

| profile | target | VBM win rate | median `R_VBM/native` |
|---|---:|---:|---:|
| uniform | `1e-3` | 99.9% | 0.096 |
| linear | `1e-3` | 66.7% | 0.582 |
| uniform | `1e-4` | 100.0% | 0.029 |
| linear | `1e-4` | 69.2% | 0.471 |

Thus the broader pattern is consistent across the comparisons: VBM is
strongly favored for uniform sources, whereas linear limb darkening makes the
grid/native/JAX routes substantially more competitive.

## Route audit and correction of the earlier run

The earlier combined JAX result cannot be used as a valid route comparison.
Although 3,184 rows completed, that run forced the JAX calculation through
Cartesian routes. The 16 timeout rows were therefore not the only problem.
The completed Cartesian rows were discarded and all 3,200 JAX rows were
rerun.

The corrected run uses the saved native `chosen_grid` and `chosen_nbin` for
each epoch. Missing native plans use the JAX routing diagnostics. Polar and
Cartesian epochs are grouped separately in the direct FFI call. For linear
limb darkening, a valid high-resolution polar plan is retained rather than
being converted to a potentially pathological high-resolution Cartesian
derivative; mixed-plan resolution retries expand polar epochs only.

| profile | target | rows containing a polar epoch | rows entirely Cartesian |
|---|---:|---:|---:|
| uniform | `1e-3` | 431 | 369 |
| linear | `1e-3` | 412 | 388 |
| uniform | `1e-4` | 367 | 433 |
| linear | `1e-4` | 357 | 443 |

The entirely Cartesian rows in this table are native-selected Cartesian plans;
they are not a global forced-Cartesian setting.

All 3,200 corrected JAX rows completed, with zero JAX forward skips and zero
JAX `dA/dt` timeouts. The original microLUX fields were preserved rather than
rerun: 3,184 rows contain the original microLUX measurements, while the 16
old timeout placeholders contain no microLUX output.

For the linear profile, microLUX used its default `n_annuli=10`. The original
microLUX run used `tol=retol=target`; uniform-source rows do not use the limb
darkening annulus parameter.

## Accuracy status

The speed result and the target-accuracy result must be reported separately.
The JAX forward values were compared against the target-specific VBM reference
in the corrected result.

| profile | target | JAX rows within target | native chosen-plan failures / missing | maximum JAX forward relative error |
|---|---:|---:|---:|---:|
| uniform | `1e-3` | 800/800 | 96 / 0 | `9.45e-4` |
| linear | `1e-3` | 768/800 | 189 / 1 | `6.85e-2` |
| uniform | `1e-4` | 790/800 | 107 / 16 | `6.84e-1` |
| linear | `1e-4` | 574/800 | 283 / 9 | `8.58e-1` |

The linear `1e-4` lane contains 142 rows that miss the JAX target even though
the saved native plan passes its own VBM check. These are JAX accuracy issues,
not timeouts, and should not be hidden by quoting only the speed win rate. A
paper using the tight-lane accuracy claim should either restrict the speed
analysis to the target-qualified subset or explicitly report these outliers.

## Paper-ready summary paragraph

> In a cache- and compile-warm comparison over 3,200 four-epoch finite-source
> evaluations, the faster implementation depended strongly on the source
> profile. For uniform sources, microLUX won 65.8--78.8% of forward cases and
> 72.6--82.9% of measured derivative cases, with median microLUX-to-JAX time
> ratios of 0.350--0.392. For linear limb darkening (`c=0.5`), JAX won
> 61.5--65.9% of forward cases and 53.5--57.7% of derivative cases, with
> median ratios of 1.200--2.458 in favor of JAX. The native forward comparison
> was profile- and tolerance-dependent, with median native-to-JAX ratios of
> 0.311--1.319. These measurements exclude compilation and route warm-up and
> should therefore be interpreted as repeated-kernel throughput rather than
> end-to-end light-curve runtime. Accuracy qualification is separate: the
> linear `1e-4` lane contains substantial JAX-only outliers despite the absence
> of timing timeouts.

## Reproduction and machine-readable output

The complete row-level result, including timing wins, ratios, route metadata,
native-plan status, and accuracy status, is:

[`tests/diagnostics/results/recal2026/jax_microlux_12800_corrected/results.json`](../results/recal2026/jax_microlux_12800_corrected/results.json)

The following scripts define the corrected measurement and merge policy:

- [`bench_jax_microlux_12800.py`](bench_jax_microlux_12800.py)
- [`retry_failed_jax_only.py`](retry_failed_jax_only.py)
- [`merge_corrected_jax_microlux.py`](merge_corrected_jax_microlux.py)

The machine-readable summaries are under
`summary.<profile>:target=<target>.win_rates_and_median_ratios`, including
matched-row counts, win counts, and p10/median/p90 ratio statistics.

## Scope and limitations

This is a controlled repeated-kernel benchmark, not an end-to-end inference
benchmark. It does not include JAX compilation, JAX route warm-up, microLUX
reruns for the 16 missing source rows, or native derivative timing. The native
and VBM ratios are retained as context from the source speed-discovery corpus;
they do not turn the JAX comparison into a universal production cost model.
