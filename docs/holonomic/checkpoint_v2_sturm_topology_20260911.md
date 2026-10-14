# V2 topology authority: Sturm quartic certificate

基準は `dev/holonomic` の `c688400` です。今回の修正では、固定角度gridをV2のtopology authorityとfallbackから外し、D14で分割された各open radial cellを境界quarticのdegree-4 Sturm実根数で判定しました。

## 判定

実装・テスト・production-path profileまで完了しました。108ケース、`n_r=64`で、coldのtopologyは1370 cellすべてがdouble Sturm tierで認証され、`TOPOLOGY_UNCERTAIN=0`、production中の512/3072/4096角度grid呼び出しはすべて0でした。

旧経路ではquarticが見つけたarcを512点gridで再確認し、食い違った場合に3072点gridの結果で上書きしていました。これは細いarcの存在をsamplingで証明する構造で、今回のwide-planet例では実際のarcをemptyに変えていました。

## 実装

- `quartic_sturm.hpp` に固定次数のSturm chainを追加しました。double、DD、`__float128`の順で符号marginを確認し、曖昧なchainはfail-closedにします。
- cellごとは中央radiusのquarticを1回だけprobeします。実根数0なら一点の `phi` 符号でfull/emptyを決め、2または4ならarcsとします。
- incumbent Aberth root数とSturm countが食い違う場合は、その時だけqf Sturm isolationを行います。localなcomplex root countだけでtopologyを決めません。
- `p4` が小さい場合は `u=-1/t` のreciprocal quarticへ移り、`theta=pi+2 atan(u)`で同じarc builderへ渡します。単純な無限遠根は認証し、厳密イベント上のmultiple rootは未解決としてfail-closedにします。
- `arcs_at` と `grid_intervals` は独立診断用APIとして残しました。`classify_cells`、`radius_terms`、`radius_value`、fast topology/band pathからは呼びません。
- `V2Profile` にSturm tier、count mismatch/isolation、reciprocal chartの計数を追加しました。

固定角度gridを削除したことで、数値不確実性を重いangular rescueへ流すのではなく、原因であるquarticのprojective chartとroot countで扱う構成になっています。

## thin-arcの再現

wide-planetの `R=2.5013971414257967` では、Sturmが2 crossingを返し、production arcは

```text
theta = [6.2830817045081035, 6.2831295188810881]
width = 4.7814372984511522e-05 rad
```

です。同じ点を `arcs_at(...,3072)` に渡すとcrossingは0でemptyになります。この差が旧M7 fixtureのwide-planet 2 epochに残っていた約 `3.96e-6` のmu差の原因でした。

旧fixtureの該当2行は、独立algebraic-boundary probeと照合したphysical値へ更新しました。`u=0` の絶対差は `3.98e-12`、`u=0.5` は `2.95e-8` でした。詳細なprobe出力、旧grid値、新値は [独立照合ログ](../../evidence/holonomic/v2_sturm_wide_independent_20260911.txt) と [machine-readable evidence](../../evidence/holonomic/v2_sturm_topology_20260911.tsv) に残しています。

## ベンチマーク

再現コマンド:

```bash
cmake --build build-holonomic-m7 --target test_quartic_sturm test_gm_coverage test_holonomic_transport test_reciprocal_chart test_root_pair test_chart_p4 bench_v2_profile
ctest --test-dir build-holonomic-m7 -R 'holonomic_(quartic_sturm|m7_reference|gm_coverage|transport|reciprocal_chart|root_pair|chart_p4)' --output-on-failure
./build-holonomic-m7/bench_v2_profile evidence/holonomic/gm_coverage_cases.tsv 1 4
./build-holonomic-m7/bench_v2_profile evidence/holonomic/gm_coverage_cases.tsv 3 3
```

同一case set・同一 `n_r=64` のmatched profile（best-of-one、warmは4 epoch trajectory）では次の通りです。

| lane | 旧Phase 8 p50 ms | Sturm path p50 ms | 旧status | 新status |
|---|---:|---:|---:|---:|
| cold value | 0.969483 | 0.887760 | 104/108 | 108/108 |
| cold value+5Jac | 1.124730 | 1.001845 | 104/108 | 108/108 |
| warm value | 0.875327 | 0.718360 | 308/324 | 316/324 |
| warm value+5Jac | 0.993879 | 0.786458 | 308/324 | 316/324 |

この比較は実行時ノイズを含むため、細かな速度差の採用根拠はraw profileを優先します。3 repetition profileでは cold value/value+5Jacが `0.799810/0.927326 ms`、warm value/value+5Jacが `0.732325/0.856256 ms` でした。

旧pathのproduction profileは [v2_phase8_profile_direct_default.txt](../../evidence/holonomic/v2_phase8_profile_direct_default.txt)、今回のraw結果は [v2_sturm_profile_20260911_matched.txt](../../evidence/holonomic/v2_sturm_profile_20260911_matched.txt) と [v2_sturm_profile_20260911.txt](../../evidence/holonomic/v2_sturm_profile_20260911.txt) です。

今回のmatched profileでは、旧pathの `grid512=1370`、`grid3072=138`、`topology_escalations=46`、`uncertain=4` に対し、Sturm pathは `grid512/3072/4096=0/0/0`、`sturm=1370`、double accept=1370、ambiguous=0、mismatch/repair/fail=0/0/0 でした。

## 検証と残課題

関連CTestは7/7 pass、`test_holonomic_m7` は10397 checks / 0 failuresです。reciprocal chartの係数・微分・root mappingの最大誤差はそれぞれ `2.763e-16`、`2.556e-16`、`3.884e-16` でした。実行ログは [v2_sturm_ctest_20260911.txt](../../evidence/holonomic/v2_sturm_ctest_20260911.txt) です。

今回のSturm certificateは、double係数をDD/qfで再構成して符号marginを検査する有限精度certificateです。区間演算による形式的な丸め区間証明ではありません。chainが曖昧な場合は値を作らずfail-closedします。また、cell内でD14が全radial discriminant eventを網羅するという前提は従来通りで、そこは今回の変更範囲ではありません。

`dump_reference.py` と古いgrid値を使う資料は、独立診断・歴史的比較として残しています。production topologyの根拠には使いません。
