# Gradient adaptive deferred 16→32 checkpoint (2026-09-25)

## 結論

勾配の誤差ledgerを使った2次元adaptive経路を、初期計算の固定費を増やさない形で追加した。`gradient_adaptive=true` の研究設定では、通常の8/16 K-ruleを最初に計算し、value snapshotが確定した後、角度ledgerが半径ledgerを明確に上回る同一半径のサンプルだけを高次数へ昇格する。昇格時は同じgeometry、quartic/root-pair、parameter derivative stateを再利用する。

この設定は既定値を変更せず、V2/V3 routerにも接続していない。Strict/ValueFirstの既存契約、`ToleranceMet` / `FiniteUncertified` / `Invalid` の意味、fail-closed挙動は維持した。

## 実装

- `AdaptiveConfig::gradient_adaptive` を追加。既定は`false`。
- ValueFirstではgradient phaseの追加workに、既存のvalue-first budgetとの最小値として次を適用。
  - node budget: 8192（ベンチ設定）
  - round budget: 16（ベンチ設定）
- `gradient_adaptive_dominance=1.25` を用い、角度と半径のledgerが拮抗しているだけでは角度upgradeを発生させない。
- 初期サンプルは従来の8/16 K-rule。gradient adaptive sampleには初期16-node aggregateと同じRのgeometry stateを保存する。
- schedulerが角度側を選んだときだけ、同じRで16→32（以後は既存の次次数列）を計算する。radial sampleの新規構築は行わない。
- `gradient_local_refinement` は別の研究スイッチとして残し、今回のdeferred経路では有効化していない。p-refinementとh-splitの結果を混ぜないためである。

## matched benchmark

入力は`evidence/holonomic/gm_coverage_cases.tsv`の108 cases、value `mu_rtol=1e-5`、gradient `atol=1e-6, rtol=1e-4`、cold/warm、Strict/ValueFirst、各3 repeat。同一実行ファイル・同一設定で、以下を比較した。

- baseline: `gradient_adaptive=false`, radial-only
- deferred: `gradient_adaptive=true`, 初期8/16、必要時16→32、node 8192、round 16

時間は各caseの3 repeat中央値をcase分布にして集計した。

| mode | policy | arm | p50 ms | p90 ms | p95 ms | p99 ms | max ms | mean evaluations | all 5 gradient ToleranceMet |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| cold | Strict | baseline | 0.593 | 1.455 | 3.736 | 54.381 | 54.656 | 376.9 | 44/108 |
| cold | Strict | deferred | 0.596 | 1.476 | 4.424 | 53.962 | 54.167 | 389.9 | 44/108 |
| cold | ValueFirst | baseline | 0.582 | 1.468 | 3.798 | 14.789 | 14.923 | 242.6 | 44/108 |
| cold | ValueFirst | deferred | 0.586 | 1.480 | 3.511 | 4.811 | 4.822 | 187.9 | 44/108 |
| warm | Strict | baseline | 0.499 | 1.353 | 19.335 | 61.116 | 61.905 | 416.8 | 43/108 |
| warm | Strict | deferred | 0.495 | 1.365 | 19.531 | 61.491 | 62.278 | 418.6 | 43/108 |
| warm | ValueFirst | baseline | 0.500 | 1.359 | 8.079 | 24.930 | 25.251 | 266.5 | 43/108 |
| warm | ValueFirst | deferred | 0.486 | 1.358 | 3.396 | 15.158 | 15.290 | 181.4 | 43/108 |

paired結果では、値の差は全324 rowsで0、stop差も0。Strictはgradient qualityの改善・悪化とも0。ValueFirstはcoldで3 rowsが一部成分`FiniteUncertified`から`ToleranceMet`へ改善し、悪化は0。warmでは改善・悪化とも0だった。`TopologyUnresolved`、`EventLocationLimited`、`InnerAccuracyLimited`はこの変更では解消しない。

解釈として、deferred経路は通常ケースの初期angular costを増やさず、ValueFirstの過剰なgradient workを抑えながらp95/tailを下げる候補になっている。ただしcold Strictのp50/p95はわずかに悪化しており、全体のgradient coverageを増やす変更ではないため、production defaultへは昇格しない。Strictでcoverageを増やすにはevent/locationや未解決inner errorを別に扱う必要がある。

## 検証

- `test_adaptive_radial`: 1608 checks, 0 failures
- `test_holonomic_m7`: 10397 checks, 0 failures
- relevant CTest: 8/8 pass（quartic、root pair、projective fold、radial atlas、D14 event系）
- value parity: paired 324 rowsで最大絶対差 0
- status parity: paired 324 rowsで差 0

## raw / 再現

- [commands](../../evidence/holonomic/gradient_adaptive_deferred16to32_20260925_commands.txt)
- [summary.json](../../evidence/holonomic/gradient_adaptive_deferred16to32_20260925_summary.json)
- [baseline cold](../../evidence/holonomic/gradient_adaptive_deferred16to32_20260925_baseline_cold.csv)
- [deferred cold](../../evidence/holonomic/gradient_adaptive_deferred16to32_20260925_deferred_cold.csv)
- [baseline warm](../../evidence/holonomic/gradient_adaptive_deferred16to32_20260925_baseline_warm.csv)
- [deferred warm](../../evidence/holonomic/gradient_adaptive_deferred16to32_20260925_deferred_warm.csv)

このcheckpointではproduction router/defaultの変更、gradient toleranceの緩和、既存のD14/PF6経路の変更は行っていない。
