# Phase48: budgetを保つsample capacity growth

基準d642e85。refinementごとにsamples.reserve(target)していた箇所を、HOLO_ADAPTIVE_SAMPLE_GROWTHで必要個数の約1.5倍まで確保する研究candidateにした。default OFF。production router/fixed API変更なし。

## メモリ契約

余分なcapacityで後のpanel作成を妨げないよう、max(現在panel capacity,設定max_panels)と設定max_node_evalsの全容量がmax_bytesに収まる場合だけ拡張する。積のoverflowを避けるため除算で上限確認してから計算。reserve_targetはmax_node_evals以下。予算が厳しい場合は従来のtargetだけ確保する。

この環境でAdaptiveSample=504 byte、AdaptivePanel=1464 byte。default全上限は17,264,640 byteで、max_bytes=67,108,864 byteより小さい（layout.txt）。新たなsampleを先行評価せず、sample再利用・node budget・refinement scheduleは変更しない。

## Profile

同じ7216行、candidate→baseline、同じHEAD/flagsのgrowth有無、CPU0、各単回。warm steadyではuniform whole .242801→.2419005 ms、LD .302750→.302249 msと小さい。coldではuniform .3860455→.380768 ms、LD .4402825→.4328835 ms。physics中央値はほぼ同じなのでallocation側の費用削減と整合するが、allocation回数そのものは今回直接計数していない。

cold/warm両profileともmu/ok/node数差0。warm steadyは先頭coldを除く。cold profileは全rowを新規workspaceで評価。

## 通常whole

14,432行、3反復、D14/topology込み。baselineはPhase47保存値、今回candidateのみ新規測定。厳密な同時交互A/Bではないため、小さな差を一般化しない。candidate測定の一部とCPU1 correctness/コンパイルが重なる。

|tol|経路|baseline p50 ms|candidate p50 ms|baseline p99 ms|candidate p99 ms|
|---|---|---:|---:|---:|---:|
|1e-3|cold|.379189|.377035|2.943619|2.943680|
|1e-3|warm|.289025|.286734|1.015999|1.008726|
|1e-3|radial|.111987|.110599|.364144|.357882|
|1e-4|cold|.418595|.416298|2.976820|2.973556|
|1e-4|warm|.333439|.330162|1.278888|1.271540|
|1e-4|radial|.144458|.142047|.522277|.516812|

通常warm集計には先頭coldを含む。全行でmu差0、status差0、node数差0。p90/p95/max、既存reference超過数はwhole_summary.json。1e-3 VBM reference超過0、1e-4既存3件は変化なし。

## 検証・判断

anchor旧走査との一致監査を有効にしてunit1484 checks、0 failures。独立reference550行中usable466、observed violation0。解析5Jac220行は成分/品質差0（Phase47比較）。低予算のunit設定では拡張条件が成立しないため従来reserveを維持する。

研究候補として保持。ただしwarm steady自体の利益は小さく、今回のwhole差も約1%以内なのでproduction採用を決める強い証拠ではない。VBM1e-3勝利目標は未達。

## 再現

Phase47のwhole compileコマンドへ-DHOLO_ADAPTIVE_SAMPLE_GROWTHを追加。runner/入力/3反復/CPU0は同じ。

```
python benchmarks/holonomic/summarize_adaptive_reciprocal.py --baseline ../adaptive_extreme_phase47/whole_candidate.tsv --candidate ../adaptive_extreme_phase48/whole_candidate.tsv --output ../adaptive_extreme_phase48/whole_summary.json
```

profile runnerは同flagsでbench_adaptive_stage_profile.cpp、INPUT warm-timing/cold-timing。baselineはgrowth defineのみ外す。unitはsparse/growth/anchor audit、reference/JacはPhase47同defines＋growth（監査有効、march/unrollなし）。raw/summaryはevidence/holonomic/adaptive_extreme_phase48/。
