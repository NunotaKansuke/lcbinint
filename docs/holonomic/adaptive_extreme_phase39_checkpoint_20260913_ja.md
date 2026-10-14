# Phase39: D14構造Hornerの既知zero work削減

基準800200b。前段の円板認証が高価だったため、現行fast pathへ戻り、C3/G4/Z3評価の先頭Horner stepを専用化してwhole epochまで測定した。

## 試作と判断

実係数leading termに対し、最初のderivative=0×v+leadingをleadingへ置換し、valueの最初の複素積を実係数×実部/虚部の2積で作る。以後のHorner順序は維持。研究flag HOLO_D14_HORNER_START。新しいsolverや受理gateの緩和はない。

小計測の補正1回はPhase33保存値498 nsに対し469 ns。ただしwhole epochで利益が小さく、試作はheaderから撤去してpatchのみ保存した。数学的に有限演算では同じでも、nonfiniteや符号zero等の例外演算の全入力同値を主張しない。

## 同じ入力・compiler・CPUでのwhole比較

14,432行、各3反復、full cold/warm/radial-only、value-only。入力・精度・設定は共通。baselineは今回再コンパイル・再測定。warm集計には各trajectory先頭cold epochを含む。

|RelTol|経路|baseline p50 ms|candidate p50 ms|baseline p99 ms|candidate p99 ms|
|---|---|---:|---:|---:|---:|
|1e-3|cold|.388049|.387327|2.954086|2.846570|
|1e-3|warm|.297829|.295698|1.049234|1.059980|
|1e-3|radial|.120197|.120804|.375580|.374330|
|1e-4|cold|.431165|.429991|2.979013|2.884495|
|1e-4|warm|.344346|.343434|1.306858|1.299012|
|1e-4|radial|.155761|.156085|.539342|.547788|

warm中央値の改善は1e-3で約0.7%、coldは0.2%未満。radial側も0.5%程度変動しており、今回の程度の差は測定変動と明確に区別できない。全体勝利と判断せず不採用。maxは約50 msのtailが残る。p90/p95/maxもraw summaryに保存。

候補→baselineの順で逐次測定、交互ランダム化ではない。CPU0へ固定。CPU1のcorrectness作業およびコンパイルが一部並行し、shared cache/powerへの影響は残る。CPU/compiler情報はenvironment.json。大幅な利益が出なかった候補へ追加の大規模測定は行わない。

## 正確性

whole全14,432行でmu差0、status差0、node数差0。両Tol・全経路でvalue convergenceは同一。独立reference 550行のうちusable466、observed violation0。既存VBM referenceに対する1e-4の3件超過は従来通りで、新規増加なし。1e-3は超過0。

ValueFirst解析5Jacの220行は全成分差0、品質/stop差0（Phase28保存値と比較）。adaptive unit 1484 checks、0 failures。unitは候補flagと通常defaultの組合せ、reference/Jacは今回の研究baseline flagsと候補の組合せ。新しい独立14-root parityは未実施。採用しないので完了したcorrectnessを越える追加試験は行わない。

## 再現

`evidence/holonomic/adaptive_extreme_phase39/horner_start.patch`を基準headerへ適用した状態で以下を実行する。baselineでは最後の候補defineを外す。

```
c++ -O3 -DNDEBUG -std=gnu++17 -fext-numeric-literals -Isrc -march=native -funroll-loops -ffp-contract=fast -fno-math-errno -DHOLO_D14_NATIVE_RESIDUAL_SCREEN -DHOLO_D14_DIRECT_CONVOLUTION -DHOLO_ADAPTIVE_STATIONARY_ENDPOINT -DHOLO_D14_WARM_NOISE_HANDOFF -DHOLO_D14_SCALAR_CONSTANTS -DHOLO_D14_HORNER_START evidence/holonomic/adaptive_estimator_phase10_20260913/runner_hybrid.cpp -o /tmp/p39_whole -lquadmath
taskset -c 0 /tmp/p39_whole evidence/holonomic/v2_adaptive_best_trajectory_20260913/input_snapshot.tsv /tmp/whole.tsv 3
python benchmarks/holonomic/summarize_adaptive_reciprocal.py --baseline ../adaptive_extreme_phase39/whole_baseline.tsv --candidate ../adaptive_extreme_phase39/whole_candidate.tsv --output ../adaptive_extreme_phase39/whole_summary.json
```

小計測はbench_d14_factor_accuracy.cppを同じresearch definesでコンパイルし、Phase31入力に`timing`引数。reference/Jacは同defines、-O3 -DNDEBUG -std=gnu++17 -fext-numeric-literals -Isrc、march/unrollなし。runnerはそれぞれcheck_adaptive_reference.cpp INPUT warm、bench_warm_gradient_parity.cpp INPUT。INPUTはevidence/holonomic/adaptive_radial_20260911/reference_cases.tsv。

## 結論

既知zero workは実際にあったが、そこを削っても今回のwhole中央値差は数us。VBM1e-3に勝つ目標は未達。構造式の数演算削減だけを延々と続けるより、評価・反復を丸ごと減らすことが必要。production router/fixed APIは変更していない。
