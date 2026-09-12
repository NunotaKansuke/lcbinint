# Phase28: D14構造式の実数定数乗算

基準9205b20。HOLO_D14_SCALAR_CONSTANTS研究flagで、構造式の実数定数×複素数だけを2成分のスカラー乗算へ置換した。一般複素積のzero項を省く。式の他の括弧・演算順は維持。既定OFF、production router変更なし。

Phase27のwarm noise handoffに重ね、7216行のstage profileをcandidate→baseline、baseline→candidateの順でCPU0で測った。steady集計はtrajectory_pos>0の5412行。これはprofiler付き時間であり、最終whole benchmarkではない。

|観測|D14Real baseline→candidate µs|profile whole baseline→candidate ms|
|---|---|---|
|1|22.7745→21.7100|.309157→.308300|
|2|22.6290→21.7335|.309776→.308192|

各観測7216行でmu完全一致、status/node差0。adaptive unit1484 checks/0 failures。小さいが同方向の差が見えたため研究flagを保存する。cold/warm両tolのnon-profile whole、全根比較、独立reference、解析Jacはこの変更について未検証。採用・VBM勝利とはしない。

特に非有限中間値では0*Infの省略で挙動が変わり得る。既存最終gateは残るが、有限値の同一性だけで例外挙動の同等性を主張しない。次の全根/難例監査対象。

再現: benchmarks/holonomic/bench_adaptive_stage_profile.cppをPhase27wholeのcompiler flags（warm handoffを含む）でbuildし、candidateだけ-DHOLO_D14_SCALAR_CONSTANTSを追加。両binaryをtaskset -c 0で、v2_adaptive_best_trajectory_20260913/input_snapshot.tsvとwarm引数で実行。rawとrepeated_summary.jsonはevidence/holonomic/adaptive_extreme_phase28。
