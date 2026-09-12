# Phase31: D14構造を使うbinary64評価の精度診断

ユーザーの指示に合わせ、D14の数学的構造を使って高精度演算を減らせるかを主題にした。基準1b1bb1c。solver、受理gate、production routerは変更していない。

## 比較

7216 geometryのcold root traceから、double presearch候補と最終候補を各14根取得。合計202048点。座標は保存済みbinary64 exportであり、qf rootそのものではない。同じ点でNewton補正 D/D' を比較した。

0. qfのC3/G4/Z3から生成した展開14次係数をdoubleへcastしHorner
1. qfのC3/G4/Z3をdoubleへcastして既存構造式を評価
2. 物理parameterから v-1, v-m, L, U, B等の因子形をdoubleで直接評価（一次dualで微分）

referenceは同じphysical parameterからqfで生成したC3/G4/Z3のqf評価。method2の式はoff-root複素点で全7216 geometryをqf照合し、D/D'各々のscaled差最大2.10e-28。これは式の数値的cross-checkであり形式証明ではない。

## 観測

|探索途中・補正誤差/(1+abs(v))|expanded|blocks|直接因子形|
|---|---:|---:|---:|
|finite中央値|9.06e-14|2.09e-16|8.69e-15|
|finite p90|1.33e-7|3.34e-11|2.92e-9|
|非有限数/101024|0|20|28|

誤差が1e-12*(1+abs(v))未満かつ最近傍分離の1e-6未満という**診断用**条件を全14根で満たすgeometryは2104→4614（blocks）、直接因子形2626。最終候補でも2156→4564、直接因子形2620。これらは誤差上界でもsolver成功条件でもない。referenceを見て測った結果であり、本番でこの分類を利用するには別の安い誤差判定が必要。

blocksは中央値で明確に強い一方、上位1%では補正誤差がroot separationを超える例もある。D'の悪条件/相殺や係数丸めを一律doubleへ落とせない。直接因子形はqf係数を正しく丸めてから使うblocksより多数例で悪く、「因子を残せば常に良い」とはならない。

## 限界と次段階

これは評価精度の診断でありwall-time勝利ではない。method0はv座標でのHornerで、現行balanced presearchそのものの再実行ではない。method1はqf係数構築費をまだ払っている。共通qf referenceも極端clusterで任意精度真値と同一とは保証しない。NaN/Infを除いたquantileと非有限件数は別記している。

次に分解すべきはC/G/Zの係数丸め・低次Horner評価・最終discriminantの相殺・D'のconditioning。特に全てをDD化せず、相殺する最終combinationだけ補償する方法と、局所中心へshiftした低次blocksを検討する。準備費と最終qf gateまで含めて測り、単にdoubleの残差で受理しない。

## 再現

```
python benchmarks/holonomic/summarize_d14_factor_accuracy.py prepare --input evidence/holonomic/v2_adaptive_best_trajectory_20260913/input_snapshot.tsv --roots evidence/holonomic/adaptive_extreme_phase28/roots.tsv.gz --output /tmp/p31_input.tsv
c++ -O3 -DNDEBUG -std=gnu++17 -fext-numeric-literals -Isrc -march=native benchmarks/holonomic/bench_d14_factor_accuracy.cpp -o /tmp/p31_probe -lquadmath
taskset -c 0 /tmp/p31_probe /tmp/p31_input.tsv > /tmp/p31_accuracy.tsv 2> /tmp/p31_identity.txt
python benchmarks/holonomic/summarize_d14_factor_accuracy.py summary --input /tmp/p31_accuracy.tsv --output /tmp/p31_summary.json
```

raw606144行をaccuracy.tsv.gzへ保存。新しいsolver経路ではないので、この段階でwhole benchmarkを追加実行しない。
