# V2 adaptive value-only benchmark checkpoint — 2026-09-11

これは、論文用 q–rho corpus で現行 adaptive V2 を再実行した結果である。以前の固定 `n_r=64` 実行とは異なり、V2 は value tolerance に応じて radial node 数を選ぶ adaptive path を使用した。VBM の `RelTol=1e-3` / `1e-4` の速度結果と、既存の `RelTol=1e-6` reference は再利用し、今回追加で計算したのは現行 V2 の14,432 callsである。

## 計時条件

測定対象は `flux_adaptive_integrate(p,u,topo,cfg,workspace)` の本体である。`LensParams`、`PrimaryFrame`、`classify_cells`（D14/topology）、入力処理、出力処理はtimer外、adaptive setup、event handling、panel refinement、physics、estimator、schedulerはtimer内とした。`mu_atol=1e-16`、gradient policy `None`、Jacobianなし、workspace warm-up後3回の中央値である。したがって、これは full epoch の比較ではなく、既存 VBM direct-kernel timingと同じ pure-kernel境界の比較である。

入力は1,804 geometry/profile/epoch行で、uniform（LDなし）とlinear（LDあり、`c=0.5`）をそれぞれ `RelTol=1e-3`、`1e-4` で評価した。各条件は3,608行、合計14,432行である。

## 結果

全条件で `value_converged=1`、`stop=Converged`、`numerical_status=OK` が3,608/3,608となった。前回の636件失敗は、現行の full-circle評価と近接physical fold保持を反映する前の古いrawであり、今回の結果には残っていない。

| profile | RelTol | rows | V2 p50 ms | V2 p95 ms | median VBM/V2 |
|---|---:|---:|---:|---:|---:|
| uniform (LD off) | 1e-3 | 3608/3608 | 0.122137 | 0.260495 | 0.513567 |
| uniform (LD off) | 1e-4 | 3608/3608 | 0.155332 | 0.362237 | 0.721356 |
| linear (LD on, c=0.5) | 1e-3 | 3608/3608 | 0.168019 | 0.385226 | 1.98919 |
| linear (LD on, c=0.5) | 1e-4 | 3608/3608 | 0.204459 | 0.547826 | 6.59370 |

速度比は `VBM selected_seconds * 1000 / V2 ms` で、1より大きいとV2が速い。LDなしではこの純粋なadaptive evaluatorはVBM direct kernelより遅く、LDありではV2が速い。

## VBM 1e-6 referenceとの精度分布

相対誤差は各V2値と同一入力の既存 VBM `RelTol=1e-6` referenceについて

```text
abs(V2 - VBM_1e-6) / abs(VBM_1e-6)
```

で計算した。以下は全3,608行の分布で、value-converged行だけを含む。`max`は少数の難しい点を含むため、p95/p99と分けて読む。

| profile | RelTol | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|
| uniform | 1e-3 | 6.97574e-8 | 4.40694e-6 | 1.31369e-5 | 2.70588e-4 |
| uniform | 1e-4 | 2.88546e-8 | 3.36594e-7 | 1.01237e-6 | 2.70697e-4 |
| linear | 1e-3 | 8.48743e-7 | 9.34684e-6 | 1.67362e-5 | 2.17776e-4 |
| linear | 1e-4 | 6.46191e-7 | 7.84223e-6 | 1.52542e-5 | 2.17773e-4 |

要求値よりかなり小さい誤差が多いのは、adaptive estimatorが安全側の停止点を選ぶことと、nested Fejér detailが実際の局所誤差より保守的になることによる。これは今回の分布からの解釈であり、要求 toleranceがreference誤差の上限を直接保証するという意味ではない。誤差図では小さい領域を確認できるよう、cell p95の階級を `1e-8` 以下から `3e-4` まで拡張した。

## 図の集計条件

q–rhoパネルは runbook の canonical edge

```text
q   = geomspace(1e-4, 1, 13)
rho = geomspace(3e-5, 1, 13)
```

による12×12 binで、各cellの最低人口は8点である。4条件とも144/144 cellが充填されている。したがって、今回のq–rho図に穴はなく、A–rho / A–qで灰色になる領域は別の座標射影で最低人口に達しないcellである。灰色とadaptive non-convergenceを同じ意味には扱っていない。

図の誤差カラーバーは cell p95 relative error の境界を

```text
0, 1e-8, 1e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4
```

に設定した。reference、誤差式、p95 reducer、raw値は変更していない。

## 再現と成果物

実行コマンド、入力hash、runner hash、raw hashは [evidence/holonomic/v2_vbm_adaptive_qrho_20260911/command.txt](../../evidence/holonomic/v2_vbm_adaptive_qrho_20260911/command.txt) と [provenance.txt](../../evidence/holonomic/v2_vbm_adaptive_qrho_20260911/provenance.txt) に保存した。機械可読結果は [summary.json](../../evidence/holonomic/v2_vbm_adaptive_qrho_20260911/summary.json)、結合済み行データは [joined_v2_vbm.tsv](../../evidence/holonomic/v2_vbm_adaptive_qrho_20260911/joined_v2_vbm.tsv)、V2 rawは [v2_results.tsv](../../evidence/holonomic/v2_vbm_adaptive_qrho_20260911/v2_results.tsv)、生成レポートは [REPORT.md](../../evidence/holonomic/v2_vbm_adaptive_qrho_20260911/REPORT.md) である。

4条件のPNG/PDFは evidence の `figures/` にある。VBM速度artifactと `RelTol=1e-6` referenceは runbookに記載された既存 `/tmp` artifactを使用した。
