# Binary multi-run FFI と microLUX の q–rho ベンチマーク

対象は `master` の binary Cartesian multi-run FFI と microLUX。入力は
runbook の q–rho coverage corpus を使い、各 block の4 epochを同じ条件で
評価した。測定時点の lcbinint は `master` の `75686e87951f` に今回の
作業ツリー差分を加えた状態で、統合後のコード内容はこの測定状態をそのまま
コミットする。

## 測定条件

- 3,608 block（4 lane × 902）、合計 14,432 epoch
- `q=geomspace(1e-4, 1, 13)`、`rho=geomspace(3e-5, 1, 13)` の 12×12 bins
- VBM の value reference は runbook 固定の `RelTol=1e-6`
- lane の target は `1e-3` / `1e-4`、profile は uniform / linear (`c=0.5`)
- compile と warm-up を定常計時から除外、4 epoch block、中央値は2反復
- 値+微分は同期済み forward block と別の `dA/dt` block の合計
- 全4 laneで `OMP_NUM_THREADS=1` 等を固定

時間は4 epoch blockあたりの秒。`microLUX speedup =
t_lcbinint / t_microLUX` で、1より大きいと microLUX が速い。

| profile / target | completed / total | value FFI | value microLUX | speedup | value+`dA/dt` FFI | value+`dA/dt` microLUX | speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| uniform / `1e-3` | 902 / 902 | 0.012995 | 0.003011 | 3.464× | 0.044978 | 0.007406 | 4.611× |
| uniform / `1e-4` | 898 / 902 | 0.073286 | 0.004374 | 12.401× | 0.204606 | 0.011281 | 13.446× |
| linear / `1e-3` | 902 / 902 | 0.017616 | 0.018320 | 0.786× | 0.072095 | 0.049991 | 1.164× |
| linear / `1e-4` | 898 / 902 | 0.095390 | 0.029662 | 2.519× | 0.411709 | 0.084799 | 3.669× |

## 精度

下表は完了 block の VBM `RelTol=1e-6` value reference に対する block 内
最大相対誤差（分母 `max(|reference|, 1)`）の median / p95 である。

| profile / target | FFI median / p95 | microLUX median / p95 |
|---|---:|---:|
| uniform / `1e-3` | `6.335e-6 / 1.952e-5` | `2.613e-5 / 2.452e-4` |
| uniform / `1e-4` | `6.453e-7 / 2.290e-6` | `1.613e-6 / 2.697e-5` |
| linear / `1e-3` | `1.172e-5 / 5.452e-5` | `1.225e-4 / 1.639e-3` |
| linear / `1e-4` | `1.873e-6 / 1.667e-5` | `8.448e-5 / 1.447e-3` |

`1e-4` の各 profile で4 blockずつが native deadline timeout となり、完了数
は898/902。timeout行は結果表に残し、速度や精度の集計で補間していない。
`dA/dt` の VBM reference は runbook にないため、微分については速度比較
だけを記録し、精度合否は主張しない。

uniform では両 target とも microLUX が速い。linear の `1e-3` では value
だけなら FFI が約1.27倍速いが、`dA/dt` を含めると microLUX が速い。linear
の `1e-4` では microLUX が速い。精度は FFI の p95 が全 laneで小さいが、
`1e-6` を全 blockで満たす結果ではない。

## 成果物と再現性

今回の集計は次のローカル evidence directory に保存している。

- [runbook](HANDOFF_vbm_filled_qrho_kernel_runbook_20260911.md)
- [summary](../results/recal2026/jax_microlux_binary_multirun_qrho_20260915/summary.json)
- [joined table](../results/recal2026/jax_microlux_binary_multirun_qrho_20260915/joined_table.tsv)
- 完全な merged rows: `../results/recal2026/jax_microlux_binary_multirun_qrho_20260915/results.json`

`summary.json` は 3,608 unique rows（completed 3,600、timeout 8）、参照元は
全行 `runbook_vbm_reltol_1e-6`。raw lane と checkpoint は大きいためローカル
evidence として保持し、ソース管理にはこの report と compact summary/table
だけを含める。

2×3 のセル別カラーマップは、この canonical q–rho 入力と集計を使う後段作業
として残している。
