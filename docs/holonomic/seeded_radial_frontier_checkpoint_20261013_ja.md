# Seed起点のradial frontier checkpoint（2026-10-13）

研究用経路。CALIBRATED=false。標準dispatchへの昇格なし。

## 方法

caustic由来の既存seedを使い、各seed半径の角度四次式の実根数を調べる。既に探索したradial bandでは再marchを省くが、seed半径の根数情報は残す。像のある帯を内外へ伸ばし、外端のfoldだけ局所Newtonで求める。通常精度で止まる端点だけ既存の高精度Newtonへ進める。レンズ極による多項式の余分な解は元のレンズ写像の残差で除外する。

角度積分は既存の高速kernelを使う。積分中に評価した点の実根数が変わった区間だけ局所foldを追加する。変更しないpanelのrefinement treeと評価値は再利用する。配置別ルール、5%距離閾値、複数zero点条件、固定2回修復制限は使わない。失敗時は同じ局所区間を追加計算し、未解決なら未解決として返す。

これは観測した根数変化の局所修復であり、全区間に隠れた変化がないという数学的な完全性証明ではない。coverage_certified=false。

## 比較条件と結果

host=rogue1、GCC 11.5.0、-O3 -march=native、base=f3547723b79828d9c501182746028e99f7bad116。u=0,0.5、相対tol=1e-3,1e-4、value-only。各評価3回の中央値、A/B順序交互。各回fresh workspace。入力はv4の保存済みcaustic seed。

**両経路ともseedが既にある時点から計測。caustic構築・seed構築は含まない。全pipelineの速度比較ではない。**

| 群 | 比較可能評価 | 新経路中央値 ms | D14中央値 ms | 新準備中央値 ms | D14準備中央値 ms | 合計時間 新/D14 |
|---|---:|---:|---:|---:|---:|---:|
| 通常 |178|0.318|0.704|0.068|0.471|0.703|
| fold |288|0.385|0.561|0.086|0.342|0.991|
| 浅い接触 |288|0.407|0.550|0.096|0.322|1.129|

754評価すべてcandidate収束かつD14との差が指定許容差内。D14未収束142評価は一致・速度の採用集合から分離。全896評価を保存。通常群p90は新2.591ms対D14 2.444ms、foldは0.957対1.264、浅い接触は1.022対1.314。中央値の改善だけではtail改善を意味しない。浅い接触群の総時間は約13%増加する。新積分時間中央値は通常0.242、fold0.293、浅い接触0.309msで、D14の0.202、0.208、0.187msより増える。次の速度課題はここ。

## 検証と限界

共有adaptive回帰1608 checks / 0 failures、新frontier・resume回帰16 checks / 0 failures。resumeで片側子panelだけ残る不具合を検出して修正し、両側tree保持の回帰を追加した。途中の失敗計測は消さず保存した。

全成分の数学的完全性、独立高精度参照に対する絶対精度、gradient、MOASARC ABI全体は未検証。D14との差の検証は独立真値に対する検証ではない。

## 再現

作業checkout: /rogue1_8/nunota/lcbinint/experiments/seeded-boundary-events

```sh
g++ -std=c++17 -O3 -march=native -fext-numeric-literals -Isrc benchmarks/holonomic/bench_seeded_radial_frontier.cpp -lquadmath -o build-seeded-events/bench_seeded_radial_frontier
# NAME = initial, fold, deeper
build-seeded-events/bench_seeded_radial_frontier evidence/holonomic/caustic_seeded_connection_v4/NAME_minimal_seeds.tsv evidence/holonomic/seeded_radial_frontier_v5/NAME_witness_final.tsv 3
```

現在の採用結果: evidence/holonomic/seeded_radial_frontier_v5/{initial,fold,deeper}_witness_final.tsv と witness_summary.json。
