# Phase41: profile専用診断費と本番費用の分離

基準9f43ac3。warm前処理の不採用後、次の対象を決めるためprofileをuniform/LD別に見直した。d14_diagnostic_msの約31 usはprofが存在するときだけ実行するposthoc共役/Vieta/cluster集計で、通常whole runnerには入らない。これを本番固定費として最適化対象にしてはいけない。

## 変更

V2Profile::collect_d14_posthoc（default true）を追加。falseの場合だけposthoc diagnostic blockを省略。solverの残差、completeness、共役処理、event分類の受理条件は変更しない。productionのprofileなしpathも変更しない。

bench_adaptive_stage_profileにwarm-timing/cold-timingモードを追加。warm/coldは従来通りdiagnosticあり。未知のmodeはexit2にしてwarm/cold取り違えを防ぐ。timingモードでも通常のstage timer overheadは残る。profileなしのproduction wallそのものではない。

## 実測

同一7216行のwarm trajectoryでtiming→diagnosticを順次単回測定、同一binary/CPU0。先頭cold epochを除いた各profile2706遷移の中央値。

|stage ms|uniform|LD|
|---|---:|---:|
|whole（timing profile）|.251236|.309476|
|topology|.092370|.093233|
|radial physics|.106777|.146398|
|arc|.071737|.081946|
|endpoint|.021265|.024621|
|D14Real|.021868|.021938|
|D14 metadata|.011980|.012099|
|D14 residual|.011799|.011921|
|double presearch|.007952|.007977|

各stageは親子関係を含むため、この表の中央値を全部足してwholeと比較しない。wholeと主要stageの隙間を全てallocation費と推測しない。steady-stateだけの表であり、trajectory先頭を含む通常whole benchmarkとは母集団が異なる。

## 検証

7216行のpaired checkでmu差0、ok差0、node数差0。timing modeのdiagnostic時間は全行0。比較スクリプトは入力key/行数をassertし、rawからsummaryを再生成する。無効modeの拒否はコード上追加したが専用自動テストは未追加。

これは本番を31 us速くした変更ではない。既存whole runnerには元からposthoc diagnosticがない。今回の成果は、測定にだけ課していた費用を分離し、次の対象を正しく選べること。

## 次の対象

warm D14 presearchは約8 usで、前処理を追加しても節約余地が小さい。D14だけでなくuniformですら約107 usあるradial physics、特に約72 usのarc/root trackingが重要。LD K-ruleだけを削ってもuniformには効かない。D14構造の研究を続ける場合も、この残存費用を無視してVBM勝利を見積もらない。

## 再現

```
c++ -O3 -DNDEBUG -std=gnu++17 -fext-numeric-literals -Isrc -march=native -funroll-loops -ffp-contract=fast -fno-math-errno -DHOLO_D14_NATIVE_RESIDUAL_SCREEN -DHOLO_D14_DIRECT_CONVOLUTION -DHOLO_ADAPTIVE_STATIONARY_ENDPOINT -DHOLO_D14_WARM_NOISE_HANDOFF -DHOLO_D14_SCALAR_CONSTANTS benchmarks/holonomic/bench_adaptive_stage_profile.cpp -o /tmp/p41_profile -lquadmath
taskset -c 0 /tmp/p41_profile evidence/holonomic/v2_adaptive_best_trajectory_20260913/input_snapshot.tsv warm-timing > /tmp/timing.tsv
taskset -c 0 /tmp/p41_profile evidence/holonomic/v2_adaptive_best_trajectory_20260913/input_snapshot.tsv warm > /tmp/diagnostic.tsv
python benchmarks/holonomic/summarize_profile_posthoc.py --diagnostic /tmp/diagnostic.tsv --timing /tmp/timing.tsv --output /tmp/summary.json
```

rawはevidence/holonomic/adaptive_extreme_phase41/。VBM比較図の再生成はしていない。数値カーネルを変更しておらず、本番性能改善の主張もしない。
