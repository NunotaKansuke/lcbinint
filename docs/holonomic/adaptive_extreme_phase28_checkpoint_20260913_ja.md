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

## 全体・独立検証の追記

Phase27 wholeをbaselineとし、同じ14432行、repeat3、CPU0でnon-profile wholeを測定した。

|RelTol|lane|baseline p50 ms|candidate p50 ms|baseline p99 ms|candidate p99 ms|
|---|---|---:|---:|---:|---:|
|1e-3|cold|.389410|.387790|3.181767|2.958745|
|1e-3|warm|.299171|.297091|1.050455|1.051681|
|1e-4|cold|.432451|.431341|3.219027|3.043799|
|1e-4|warm|.345696|.344702|1.299394|1.310342|

全laneでmu完全一致、node/status差0、7216/7216収束。VBM reference超過は1e-3=0、1e-4=既存3。cold max55.850→51.855ms。scalar化はqfにも適用するためtailの改善はあり得るが、単一whole run比較だけで効果量を確定しない。warm p99は僅かに悪化しており、全quantile非回帰を主張しない。引き続き研究flagで保持。

全14432根集合のbinary64 export完全一致、role/event/cell/completeness差0、qf-cold94→94。独立reference550行はusable466でobserved violation0、残り84は未認証。mu/stop差0。解析5Jac220行も値/品質/stop完全一致。

参考: root監査はCPU1でwholeと並行実行、reference/gradientのcompileと実行はwhole初期と重なった。CPU affinityは分離したが、共有cache/電力影響を完全には除けない。小さい中央値差の限界として扱う。

再現whole: Phase27whole build flagsに-DHOLO_D14_SCALAR_CONSTANTS追加、runner_hybrid.cppを /tmp/p28_whole にbuildし、input_snapshot.tsv whole_candidate.tsv 3 を指定。summarize_adaptive_reciprocal.pyのパスはadaptive_extreme_phase11起点なので --baseline ../adaptive_extreme_phase27/whole_candidate.tsv --candidate ../adaptive_extreme_phase28/whole_candidate.tsv --output ../adaptive_extreme_phase28/whole_summary.json。
rootはbench_d14_root_work_detail.cppにnative residual screen、warm noise handoff、scalar constantsを指定し 1 both。比較oracleはphase27 roots/events。
reference/gradientは-O3 -DNDEBUG -std=gnu++17 -fext-numeric-literals -Isrcとwarm noise handoff/scalar constants（march/unrollなし）。check_adaptive_reference.cppにreference_cases.tsv warm、bench_warm_gradient_parity.cppに同入力。

新たな全root誤差やreference violationは見つからなかった。非有限中間値を含む全入力の同等性は引き続き未証明。production defaultは変更しない。
