# Phase 56: 中央値帯域のgeometry仕事量と診断入力

基準a60a025。最近の微小変更の棄却が続いたため、通常ケースで大きい費用を調べる診断入力を作った。solver/API/router変更なし、新しい高速化の主張なし。

Phase54通常wholeの1e-3 uniform、trajectory先頭を除く2706行の40–60 percentile（0.193466–0.245418ms）を抽出した542行を、同じPhase54のprofileへjoin。mu/node数一致を確認。先頭epochを含む従来wholeのp50とは母集団が異なるので、速度改善と見せない。

中央帯域でのinstrumented stage中央値:

|stage|ms|
|---|---:|
|topology|0.090480|
|physics|0.120889|
|arc（physicsの内側）|0.080609|
|endpoint（physicsの内側）|0.026388|
|D14Real|0.021595|
|D14 residual|0.011616|
|metadata|0.014234|
|double presearch|0.008067|
|setup|0.011707|
|estimator|0.003612|
|scheduler|0.001046|

parent/childの重複があるので全stageを加算しない。profile timer付きの別実行であり、通常wholeの内訳を厳密に復元するものではない。大きい候補としてはnode geometry、D14Real、metadataが残る。schedulerや未収束時の小さい分岐削減だけでは差を埋めにくい。

中央帯域を時間順に並べ、重複trajectoryを除いて32軌道を等間隔抽出。対応するuniform/LDと4epochを元入力から文字列のまま転記し256行を保存した。物理parameterによるroutingではなくdiagnostic workloadの選択である。既存runnerで実行し、元full profileとmu/ok/nodes/trajectory_posは全256行一致した。

この入力は次の仮説を安く落とすためだけに使う。tail/難例を代表せず、reference certificateでもなく、全体勝利判定をこのsubsetへ縮小しない。採用時は既存hard/referenceと14432行wholeを再検証する。今回unit/reference再実行なし。目標は未達。

再現:
```
python benchmarks/holonomic/select_median_work_cases.py --whole evidence/holonomic/adaptive_extreme_phase54/whole_candidate.tsv --profile evidence/holonomic/adaptive_extreme_phase54/cache1.tsv --input evidence/holonomic/v2_adaptive_best_trajectory_20260913/input_snapshot.tsv --output evidence/holonomic/adaptive_extreme_phase56
```
Phase54 build_trial.shで作るp54_cacheを使い、`taskset -c 0 /tmp/p54_cache evidence/holonomic/adaptive_extreme_phase56/input_snapshot.tsv warm-timing`でprofile再生成。選択は保存済timingから行うため、新しいtimingで再選択すれば同じ32軌道になるとは限らない。出典SHA256をsummary.jsonに保存。
