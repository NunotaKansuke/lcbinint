# Phase47: Fejér sample格子を利用したwarm anchor探索

基準e4b6d31。root solverの受理変更ではなく、adaptive nodeごとのwarm seed選択で空スロットを読む無駄を削った。HOLO_ADAPTIVE_SPARSE_ANCHORで有効、default OFF。production router/fixed-n_r API変更なし。

## 不変条件と実装

level LのFejér sampleは256-slot latticeのk*(256/2^L)、1<=k<2^Lにだけ格納される。既存sample assignmentはこの位置への書き込みだけであり、祖先panelのlevelは完成済みlevelを保持する。

新探索は現在panelではmax(今回level,旧level)、祖先では完成済みlevelを使って占有可能slotだけ走査する。昇順slotと祖先の順序、厳密d<distance判定を保持し、等距離なら従来と同じ先行sampleを選ぶ。7点panelでは256回から7回へ減る。総費用の256/7倍高速化を意味しない。

HOLO_ADAPTIVE_ANCHOR_AUDITを付けると従来全slot探索も実行し、選んだpointerが違えばabortする。監査をtiming binaryへ入れない。

## 検証

7216-row warm trajectoryの全nodeで監査完了、mu/ok/node差0。adaptive unit1484 checks 0 failures（監査有効）。独立reference550行中usable466、observed violation0（監査有効）。解析5Jac220行も監査完了、成分差0、品質差0。reference/JacはPhase42との比較。高精度要求や祖先panel利用のテストを含むが、任意に外部から壊したpanel stateに対する保証ではない。

## 通常whole A/B

14,432行、各3反復、同じ入力・compiler flags・CPU0。candidateは監査OFF。baselineはPhase42 binaryを今回再実行した（compact storage有効、sparse anchorなし）。現在HEADとの違いには後続のprofile専用カウンタ等があるため、同HEADからdefineだけ違えてcompileした厳密交互A/Bではない。baseline再実行でも保存値と整合する改善が出たが、小さい差の一般化には注意する。

|tol|経路|baseline p50 ms|candidate p50 ms|baseline p99 ms|candidate p99 ms|
|---|---|---:|---:|---:|---:|
|1e-3|cold|.383476|.379189|2.953323|2.943619|
|1e-3|warm|.293487|.289025|1.020971|1.015999|
|1e-3|radial|.116709|.111987|.368300|.364144|
|1e-4|cold|.425526|.418595|2.977612|2.976820|
|1e-4|warm|.339612|.333439|1.289860|1.278888|
|1e-4|radial|.150563|.144458|.533650|.522277|

全14,432行・全経路でmu差0/status差0/node数差0。warm集計はtrajectory先頭coldを含む。既存1e-4 VBM referenceの3件超過は変化なし、1e-3は超過0。p90/p95/max、coverage詳細はwhole_summary.json。

candidate→baselineを順次測定。CPU1のcorrectnessおよびコンパイルがcandidate測定の一部に重なる。詳細environment.json。中央値改善約1–2%、radial約4%。大きな勝利ではなく、数値結果を維持する小さな研究候補として保持する。VBM1e-3に勝つ目標は未達。

## 再現

```
c++ -O3 -DNDEBUG -std=gnu++17 -fext-numeric-literals -Isrc -march=native -funroll-loops -ffp-contract=fast -fno-math-errno -DHOLO_D14_NATIVE_RESIDUAL_SCREEN -DHOLO_D14_DIRECT_CONVOLUTION -DHOLO_ADAPTIVE_STATIONARY_ENDPOINT -DHOLO_D14_WARM_NOISE_HANDOFF -DHOLO_D14_SCALAR_CONSTANTS -DHOLO_ARC_COMPACT_STORAGE -DHOLO_ADAPTIVE_SPARSE_ANCHOR evidence/holonomic/adaptive_estimator_phase10_20260913/runner_hybrid.cpp -o /tmp/p47_whole -lquadmath
taskset -c 0 /tmp/p47_whole evidence/holonomic/v2_adaptive_best_trajectory_20260913/input_snapshot.tsv /tmp/whole.tsv 3
python benchmarks/holonomic/summarize_adaptive_reciprocal.py --baseline ../adaptive_extreme_phase47/whole_baseline.tsv --candidate ../adaptive_extreme_phase47/whole_candidate.tsv --output ../adaptive_extreme_phase47/whole_summary.json
```

監査profileは同flags＋HOLO_ADAPTIVE_ANCHOR_AUDIT、bench_adaptive_stage_profile.cpp INPUT warm-timing。reference/Jacは同defines＋監査、march/unrollなし。unitは通常defaults＋sparse＋audit。各runner/inputはPhase42参照。

raw/summaryはevidence/holonomic/adaptive_extreme_phase47/。次はrefinementごとのsamples.reserve(target)による再確保・コピーがremaining costに寄与するかを調べる。capacityの増やし方を変える場合もmax_bytes契約を維持する。
