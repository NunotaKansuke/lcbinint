# 引き継ぎ資料：binary multi-run FFI と microLUX の VBM event-annuli ベンチマーク

作成日: 2026-09-16
対象ホスト: `rogue1`
対象 checkout: `/rogue1_8/nunota/lcbinint`
lcbinint: `master` `94bdbb92123f4b52ef952297e09c8fe084db497e`
microLUX: `/rogue1_8/nunota/microlux` の `perf/vbml-speedup`、commit `9378dd33b50e61711f0977dfedd6bcaea6dffbd4`

この資料は、[VBM filled q–rho runbook](HANDOFF_vbm_filled_qrho_kernel_runbook_20260911.md) に従って実行した binary の比較を、同じ checkout と成果物から再開するためのメモである。実装の source diff は既に lcbinint の上記 HEAD に入っている。今回の追加作業は、VBM の event-annuli を使った microLUX 再計測、集計、図の表示修正である。

## 現在の状態

- 4 lane × 902 block = **3,608/3,608 rows completed**。各 block は4 epochなので、値の比較は14,432 epochである。
- linear source では、各 block の4 epochを VBM `BinaryMagDark` (`RelTol=target`) で先に走らせ、最終 `nannuli` の最大値を、その block の static microLUX `n_annuli` として固定した。
- uniform source には annuli override を適用していない。
- 値だけ (`forward`) と、値+微分 (`forward block + synchronized dA/dt block`) を別々に計時した。compile と warm-up は定常計時から除外し、各 block の反復数は2である。
- 速度比は `R=t_microLUX/t_lcbinint` に統一した。`R>1` なら lcbinint が速い。
- 精度パネルは `e=|value-reference|/max(|reference|,1)` の **生の相対誤差**であり、`1e-6` で割っていない。
- 結果ディレクトリはローカルの `.git/info/exclude` で除外されている。大きな JSON/TSV/PNG/PDF はそこに残し、この handoff のみを source control に入れる。

## 入力と固定条件

入力と集計は次の evidence directory にある。

```text
/rogue1_8/nunota/lcbinint/tests/diagnostics/results/recal2026/jax_microlux_binary_multirun_qrho_20260916_saved_native_plan
```

主な条件は次の通り。

- profile: `uniform` と linear limb darkening (`c=0.5`)
- timing target: `1e-3` と `1e-4`
- epoch index: `(0, 7, 15, 23)`
- q–rho 図: `q=geomspace(1e-4,1,13)`、`rho=geomspace(3e-5,1,13)` の12×12 bins
- 図の最小 cell population: 8 epoch records（block row の最低2件 × 4 epoch）
- thread: `OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`、`MKL_NUM_THREADS=1`
- Python 3.10.19、JAX/JAXLIB 0.6.2、NumPy 2.2.6、JAX CPU backend
- lcbinint extension: `/rogue1_8/nunota/lcbinint/build/lcbinint/_lcbinint.cpython-310-x86_64-linux-gnu.so`

値の精度比較に使った VBM reference は runbook 固定の `RelTol=1e-6` value reference である。lane の `target` は速度計測と microLUX/FFI の停止条件に使い、reference の精度設定とは混同しない。

## microLUX の annuli 選択

選択処理は `tests/diagnostics/recal2026/rerun_microlux_vbm_event_annuli.py` にある。

- linear / `1e-3`: 902 block の VBM event `nannuli` は `1..29`、median `6`
- linear / `1e-4`: 902 block の VBM event `nannuli` は `1..81`、median `14`
- event-specific static integer の group 数: `104`
- event-annuli 結合: `3,608/3,608` rows、missing `0`

VBM の probe と annuli 選択時間は microLUX の steady-state timing に含めていない。uniform は limb-darkening annuli を使わないため、event `nannuli` の表を作らない。

## 代表的な集計値

時間は4 epoch blockあたりの秒で、中央値である。

| profile / target | value FFI | value microLUX | `R` | value+`dA/dt` FFI | value+`dA/dt` microLUX | `R` |
|---|---:|---:|---:|---:|---:|---:|
| uniform / `1e-3` | 0.01010 | 0.004544 | 0.6389 | 0.03329 | 0.01108 | 0.4682 |
| uniform / `1e-4` | 0.02487 | 0.005828 | 0.3200 | 0.07914 | 0.01467 | 0.2496 |
| linear / `1e-3` | 0.01334 | 0.01602 | 1.650 | 0.04345 | 0.04286 | 1.248 |
| linear / `1e-4` | 0.02876 | 0.04993 | 2.044 | 0.1044 | 0.1412 | 1.509 |

VBM `RelTol=1e-6` value referenceに対する block内最大相対誤差の median / p95 は次の通り。

| profile / target | FFI | microLUX |
|---|---:|---:|
| uniform / `1e-3` | `1.548e-4 / 3.909e-4` | `2.613e-5 / 2.452e-4` |
| uniform / `1e-4` | `1.360e-5 / 3.836e-5` | `1.611e-6 / 2.693e-5` |
| linear / `1e-3` | `2.543e-4 / 5.572e-4` | `1.998e-4 / 1.003e-3` |
| linear / `1e-4` | `2.484e-5 / 6.925e-5` | `3.618e-5 / 3.359e-4` |

`dA/dt` の VBM reference は runbookにないため、微分については速度と両実装の値比較だけを記録し、微分の精度合否は主張しない。

## 図の仕様と修正点

図は `plot_qrho_maps.py` で16枚（4 profile/target条件 × values/values+`dA/dt` × microLUX/FFI）を生成する。

- 上段: cell median の `R=t_microLUX/t_lcbinint`
- 下段: 選択した engine の raw per-epoch relative error の cell p95
- 空 cell は灰色、最小8 records未満も灰色
- 下段の error colourbar は既存 protocol 図の target-matched scale に合わせた
  - target `1e-3`: `1e-4`–`1e-2`、上端は `≥1e-2`
  - target `1e-4`: `1e-5`–`1e-3`、上端は `≥1e-3`

以前の図にあった `e/1e-6` と、raw 値に対して `1e3` まで広げた色階級は残していない。図の下段で大きな色になるセルは、表示倍率ではなく raw 計算値の誤差である。

## 検証結果

[validation_event_annuli.log](../results/recal2026/jax_microlux_binary_multirun_qrho_20260916_saved_native_plan/validation_event_annuli.log) の最終状態は次の通り。

```text
PASS
rows: 3608
linear annuli mismatches: 0
ratio inversion checks: 3608
figures: 16
final statuses: {'completed': 3608}
```

`results_event_annuli.json` の checkpoint audit でも、正式な4 laneは各902 unique key、duplicate 0、missing 0 である。`smoke.jsonl` / `smoke_t1e-4.jsonl` は再開確認用の別 artifact なので、正式な row 数に足さない。

## 未解決の数値問題

この benchmark は完走しているが、精度を全面的に合格と扱ってはいけない。raw microLUX epoch error の最大値は次の通り。

| lane | 最大 raw error | case / epoch |
|---|---:|---|
| uniform / `1e-3` | `1.20144` | case 95 / epoch 0 |
| uniform / `1e-4` | `1.07969` | case 95 / epoch 3 |
| linear / `1e-3` | `1.01136` | case 90 / epoch 0 |
| linear / `1e-4` | `0.728585` | case 90 / epoch 3 |

FFI 側の最大 raw error は lane により約 `1.38e-3`–`2.32e-3` で、`RelTol=1e-6` を全行で満たしているわけではない。別途行った case 90/95 の microLUX 集中診断では、`microlux/utils.py:177` から `No enough space to insert new samplings ... Current length vs max length` の警告が出た。これは sampling capacity/strategy が外れ値に足りない可能性を示すが、根本原因の確定ではない。strategy を変えた再計測はまだ実施していない。

次に再開する場合は、まず case 90/95 を同じ入力・同じ VBM reference で再現し、warning の有無と sampling 配列上限を記録する。そこで strategy を変更するなら、全3,608 rowsを先に回さず、該当 case の forward/value+`dA/dt` と raw error を確認してから正式な再計測方針を決めること。

## 成果物

正式な evidence directory:

```text
/rogue1_8/nunota/lcbinint/tests/diagnostics/results/recal2026/jax_microlux_binary_multirun_qrho_20260916_saved_native_plan
```

- [merged results](../results/recal2026/jax_microlux_binary_multirun_qrho_20260916_saved_native_plan/results_event_annuli.json)
- [summary](../results/recal2026/jax_microlux_binary_multirun_qrho_20260916_saved_native_plan/summary_event_annuli.json)
- [joined table](../results/recal2026/jax_microlux_binary_multirun_qrho_20260916_saved_native_plan/joined_table_event_annuli.tsv)
- [event-annuli report](../results/recal2026/jax_microlux_binary_multirun_qrho_20260916_saved_native_plan/REPORT_event_annuli.md)
- [figure index](../results/recal2026/jax_microlux_binary_multirun_qrho_20260916_saved_native_plan/FIGURES_event_annuli.md)
- [provenance JSON](../results/recal2026/jax_microlux_binary_multirun_qrho_20260916_saved_native_plan/provenance_event_annuli.json)
- [event-annuli command](../results/recal2026/jax_microlux_binary_multirun_qrho_20260916_saved_native_plan/command_event_annuli.txt)
- [figure renderer](../results/recal2026/jax_microlux_binary_multirun_qrho_20260916_saved_native_plan/plot_qrho_maps.py)
- [representative figure](../results/recal2026/jax_microlux_binary_multirun_qrho_20260916_saved_native_plan/figures/linear_t1e-4_values_plus_dA_dt_microLUX.png)

入力 hash は `input_for_benchmark.json` = `8c3cf6a0db7f6326291a16bc25f62abe5b7e99dac51cd834ec95be585b83ba1c`。詳細な source/artifact hash は [provenance_event_annuli.txt](../results/recal2026/jax_microlux_binary_multirun_qrho_20260916_saved_native_plan/provenance_event_annuli.txt) を参照する。

## 再開コマンド

高コストの lane は既に完了しているため、図や report の確認だけなら再計測しない。

```bash
cd /rogue1_8/nunota/lcbinint
RUN=tests/diagnostics/results/recal2026/jax_microlux_binary_multirun_qrho_20260916_saved_native_plan

# JAX raw lanes（checkpoint があるので、未完了分だけ再開できる）
bash "$RUN/run_lanes.sh"

# microLUX event-annuli lane（正式 run の条件）
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
LCBININT_BENCH_BUILD_ROOT=/rogue1_8/nunota/lcbinint/build \
python tests/diagnostics/recal2026/rerun_microlux_vbm_event_annuli.py \
  --input "$RUN/input_for_benchmark.json" \
  --output "$RUN/event_full_current/results.json" \
  --split-lanes --parallel-workers 2 --repeats 2 \
  --vbm-timeout 60 --forward-timeout 60 --derivative-timeout 60

# raw lane と event-annuli を再結合（入力と output は script 内の固定値）
python "$RUN/assemble_event_annuli_results.py"
python "$RUN/write_report_event_annuli.py"
```

図だけを再生成する場合は、`plot_qrho_maps.py` に `results_event_annuli.json`、`--profile`、`--target`、`--engine {microLUX,ffi}`、`--speed-mode {values,dA_dt}`、`--min-count 8` を渡す。4条件 × 2 speed mode × 2 engine の16通りを回す。`MPLCONFIGDIR` は書き込み可能な一時ディレクトリに設定する。
