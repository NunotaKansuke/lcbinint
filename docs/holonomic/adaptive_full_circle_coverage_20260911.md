# Adaptive V2 full-circle coverage checkpoint — 2026-09-11

## 判定

adaptive V2 の `kFull` を専用評価へ分離し、残っていた2種類の失敗も原因に対応した。論文用 q/rho corpus の value-only rerun は、RelTol=1e-3 と 1e-4 の両方で 7,216/7,216 行が `value_converged=1`、`numerical_status=OK`、`stop=Converged` になった。

旧結果では 14,432 行中636行が `TopologyUnresolved` だった。内訳は、full-circle未実装による624行と、tiny-source/near-axisの残り12行だった。今回の結果では両方ともゼロである。

## 実装

`adaptive_epoch.hpp` に adaptive専用 `mapped_full_circle()` を追加した。`kFull` cellでは radial node ごとのquartic/root/endpoint探索を行わず、`F0=2πR` を直接返す。LDの全周積分は `phi>0` を確認したうえで、32→64→128→256 のnested periodic trapezoidを使い、既存slotを再利用する。32/64等の差を `inner` error ledgerへ渡し、Strict Jacobianでは内部からuser parameterへの絶対値写像で微分誤差を上限評価する。既存固定V2の `full_circle_terms()` は変更していない。

`radial_events(..., retain_adaptive_metadata=true)` では、近接した異なる `physical_real` eventを旧absolute merge toleranceで結合しない。D14が返した各physical foldをそのままadaptive cell boundaryへ渡す。固定-resolution側の従来merge挙動は維持する。

adaptive側の `adaptive_arc_intervals()` は、既存arc solverが `kDegenerate`、cell期待root数との不一致、または期待するarc kindとの不一致になったnodeだけ、同じquarticのSturm certificateとqf root isolationで補修する。angular gridは使わず、certificateまたは孤立化が失敗した場合は従来どおりfail-closedとなる。

warm continuationが `ArcKindMismatch`、`DegenerateChart`、`RootContinuationMismatch` を返した場合は、同一Rをfresh quartic/root stateで一度だけ再評価する。cold retryも失敗した場合は成功扱いにしない。sample単位のreject reason、panel/level/node/R、retry結果をdiagnostic CSVへ出力する。

## 失敗原因

- `c9` はR=0.015950811586656035 と R=0.015950908555727977 の2つのdistinct physical foldを持つ。差は約 `9.6969e-8` で、旧 `merge_tol=1e-7` により後者が消えていた。今回adaptive metadataでは両方を保持し、その間の4-crossing cellを復元した。warm retryを強制しても直らない種類のイベント分割欠落だった。
- `c92` は failing nodeの直接t-chart Aberthが3本のnear-real rootを返した一方、Sturm certificateは2本だった。adaptiveだけが証明付きqf isolationの2本を使ってarcを再構成するようにし、単純なodd-root rejectを解消した。

個別diagnosticの結果は `evidence/holonomic/v2_adaptive_full_circle_20260911/hard_case9/` と `hard_case92/` に保存した。今回の再実行では両ケースとも `sample_reason_counts.csv` に有意なrejectがなく、cold retry成功に依存していない。

## 全corpus結果

入力は既存 VBM RelTol=1e-6 referenceを持つ 3,608行/conditionで、uniform/linear × RelTol=1e-3/1e-4 の4条件、合計14,432行である。相対誤差は入力に保存された同一epochのreferenceに対して計算した。

| profile | RelTol | rows | value converged | relative error p50 | p95 | p99 | max | whole ms p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform | 1e-3 | 3608 | 3608 | 6.98e-8 | 4.41e-6 | 1.31e-5 | 2.71e-4 | 0.130 | 0.282 | 0.406 | 0.682 |
| linear | 1e-3 | 3608 | 3608 | 8.49e-7 | 9.35e-6 | 1.67e-5 | 2.18e-4 | 0.178 | 0.408 | 0.545 | 0.930 |
| uniform | 1e-4 | 3608 | 3608 | 2.89e-8 | 3.37e-7 | 1.01e-6 | 2.71e-4 | 0.165 | 0.380 | 0.658 | 2.217 |
| linear | 1e-4 | 3608 | 3608 | 6.46e-7 | 7.84e-6 | 1.53e-5 | 2.18e-4 | 0.219 | 0.582 | 0.918 | 1.682 |

これはcoverage/correctnessのrerunであり、full-epoch Jacobian性能の判定ではない。`mapped_full_circle()` のJacobian経路は既存のvalue/gradient contractに従うが、今回のcorpus runnerはvalue-onlyである。

## 再現

ビルドは optional pybind targetを切って行う。

```bash
cmake -S tests/holonomic_cpp -B /tmp/lcbinint-holonomic-adaptive-build \
  -DCMAKE_BUILD_TYPE=Release -DLCBININT_BUILD_HOLONOMIC_M7_PY=OFF
cmake --build /tmp/lcbinint-holonomic-adaptive-build -j2
```

full corpusとhard-case diagnosticのコマンドは
`evidence/holonomic/v2_adaptive_full_circle_20260911/command.txt` にある。raw結果と機械可読summaryはそれぞれ `v2_results.tsv` と `summary.json` である。

## 検証

- `test_adaptive_radial`: 1,482 checks, 0 failures
- `test_quartic_sturm`: PASS
- full corpus: 14,432/14,432 value-converged、14,432/14,432 numerical status OK
- hard case 9/92: sample reject 0
- independent reference audit for c9/c92: all usable rows violation 0
- full-circle Jacobian: five user-parameter central-difference checks passed
- angular fixed-grid fallbackは新しいadaptive分岐に導入していない
- fixed V2 `full_circle_terms()` と固定-resolution event mergeの通常経路は変更していない

CTestは15テスト中14テストが通る。`holonomic_m7_reference` はテスト本体がfixtureを相対パスで開く既存仕様のため、build directoryからのCTestではfixtureを開けない。repo rootから同じbinaryを直接起動すると `10397 checks, 0 failures` になる。この作業によるテスト失敗ではない。

## 残る注意点

`c9`の修正はadaptive metadataだけでphysical eventを保持するため、近接physical foldが増えた場合にadaptive cell数が増える。これはsamplingで省略するより正しい保守的挙動である。full-circleのperiodic ruleは全周の `phi>0` を各nodeで検査するため、分類が誤ってfullになる場合を黙って受け入れない。
