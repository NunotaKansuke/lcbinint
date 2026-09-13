# Phase42: arc境界vectorの再確保削減

基準7b6ac66。Phase41で約72–82 usを占めたarc/root trackingについて、数値計算を変えずheap allocationを減らす研究candidateを追加した。HOLO_ARC_COMPACT_STORAGEで有効化、default OFF。production router/fixed APIのdefault behaviorは変更しない。

## 変更

- transportしたthetaをsort後、重複除去を別vectorへpushせず元vector内でcompactする。前回受理要素との差が1e-11を超えると保持する既存predicateをそのまま使う。
- cold quarticで最大4個の実rootをcollectするvectorをreserve(4)。
- arc結果vectorをtheta数/2でreserve。受理個数を固定したり余分なarcを切り捨てたりしない。

branch/残差/phi sign/暖機thin-arc certificate、Newton回数、誤差契約は変更しない。既存点や浮動小数演算の順序はそのまま。

## Profile A/B

同じ7216行、warm-timing、同一compiler設定/CPU0でcandidate→baseline。各profileのsteady2706行、先頭coldを除外。

|profile|arc baseline/candidate ms|whole baseline/candidate ms|
|---|---:|---:|
|uniform|.071244 / .069362|.251672 / .247763|
|LD|.082197 / .079536|.310687 / .307287|

値/ok/node数の差0。stageからallocation削減と対応した数usの改善が見えるため、whole benchmarkまで進めた。

## 通常whole benchmark

14,432行、3反復、cold/warm/radial、RelTol1e-3/1e-4。profile instrumentationなし。baselineはPhase39で同じcompiler flagsで測ったwhole_baseline.tsvを再利用。今回のprofile A/Bとは異なりwholeのbaselineは同時再測定ではない。CPU1のcorrectness作業とコンパイルがcandidate測定の一部に重なっている。小さな性能差の一般化にはこの限界がある。

|tol|経路|baseline p50|candidate p50|baseline p99|candidate p99|単位|
|---|---|---:|---:|---:|---:|---|
|1e-3|cold|.388049|.383651|2.954086|2.946130|ms|
|1e-3|warm|.297829|.293692|1.049234|1.020049|ms|
|1e-3|radial|.120197|.116573|.375580|.368428|ms|
|1e-4|cold|.431165|.425524|2.979013|2.988330|ms|
|1e-4|warm|.344346|.339105|1.306858|1.290596|ms|
|1e-4|radial|.155761|.150472|.539342|.529942|ms|

中央値は約1–1.5%改善、radialは約3%。1e-4 cold p99は微増。全14,432行、全経路でmu差0、status差0、node数差0。warmの通常whole集計はtrajectory先頭coldを含む。p90/p95/max、既存reference超過数はwhole_summary.json参照。

## 検証・採否

adaptive unit1484 checks 0 failures。独立reference550行中usable466、observed violation0。解析5Jac220行は全成分差0、品質差0（Phase39と比較）。1e-4 VBM referenceの既存3件超過は変化なし。

明確に削れるallocationがあり、数値結果も一致しているため研究candidateとしてコードを保持する。ただし新defaultにはせず、全体を劇的に速くしたとは主張しない。VBM1e-3勝利の目標は未達。D14 root set自体を変更しないので新しい独立14-root求根比較は実施していない。

## 再現

```
c++ -O3 -DNDEBUG -std=gnu++17 -fext-numeric-literals -Isrc -march=native -funroll-loops -ffp-contract=fast -fno-math-errno -DHOLO_D14_NATIVE_RESIDUAL_SCREEN -DHOLO_D14_DIRECT_CONVOLUTION -DHOLO_ADAPTIVE_STATIONARY_ENDPOINT -DHOLO_D14_WARM_NOISE_HANDOFF -DHOLO_D14_SCALAR_CONSTANTS -DHOLO_ARC_COMPACT_STORAGE evidence/holonomic/adaptive_estimator_phase10_20260913/runner_hybrid.cpp -o /tmp/p42_whole -lquadmath
taskset -c 0 /tmp/p42_whole evidence/holonomic/v2_adaptive_best_trajectory_20260913/input_snapshot.tsv /tmp/whole.tsv 3
python benchmarks/holonomic/summarize_adaptive_reciprocal.py --baseline ../adaptive_extreme_phase39/whole_baseline.tsv --candidate ../adaptive_extreme_phase42/whole_candidate.tsv --output ../adaptive_extreme_phase42/whole_summary.json
```

profileは同flagsでbench_adaptive_stage_profile.cppをcompileしINPUT warm-timing。baselineはHOLO_ARC_COMPACT_STORAGEのみ外す。reference/Jacは同defines、-O3 -DNDEBUG -std=gnu++17 -fext-numeric-literals -Isrc（march/unrollなし）。入力はevidence/holonomic/adaptive_radial_20260911/reference_cases.tsv、checkerとgradient runnerはPhase39と同じ。unitは通常defaults＋candidate define。

raw/profile/検証結果はevidence/holonomic/adaptive_extreme_phase42/。
