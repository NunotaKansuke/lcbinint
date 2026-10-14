# Phase44: VFloor拒否の分離と初期値再選択（試作不採用）

基準a5593a8。VFloor関連件数が多いため、予測時v<=0、補正後v<=0、補正後0<v<=VFloorを個別カウントした。全て観測だけでproduction受理条件は変更しない。

## 結果

warm steady各2706行。baselineのLDは予測非正1713、補正後非正289、補正後tiny-positive7942。uniformは1719/298/6161。VFloorタグは主に本当に小さい正のvから出ている。タグは重複するため足し合わせない。

次に、予測vが非正になった場合のみ、前の有効な(m,v)から同じ最大5回Newtonを始め直す案を試した。新しいroot solveはなく、同じ最終残差/step/VFloor/branch/薄arc判定を通す。旧seed自体もfinite、v>VFloor、TMax以内を要求する。予測失敗を無条件受理する案ではない。

quarticへ戻る回数はuniform43840→43800、LD47400→47360。各40件、約0.1%減に留まる。arc中央値はuniform.069128→.0687165 ms、LD.0793295→.0789635 ms。追加初期値分岐を入れる根拠が弱いため撤去しpatch保存のみ。診断カウンタは残した。

同7216行、mu差/(1+|mu|)最大2.10e-13、ok/node数差0。ただし独立reference/Jac追加検証は未実施。profileで実際に省けた仕事がごく小さいため通常14,432-row whole比較には進めない。単回profile中央値の微小な改善を本番勝利とは扱わない。

## 再現

Phase43と同じcompile flags、runnerはbench_adaptive_stage_profile.cpp INPUT warm-timing、CPU0。baseline→candidateを逐次測定。candidateはrestart_from_seed.patch適用と-DHOLO_MV_RESTART_FROM_SEEDを追加。どちらもPhase42のHOLO_ARC_COMPACT_STORAGE有効。入力はevidence/holonomic/v2_adaptive_best_trajectory_20260913/input_snapshot.tsv。

raw/summary/patchはevidence/holonomic/adaptive_extreme_phase44/。stage profiler末尾にpredictor_nonpositive/corrected_nonpositive/corrected_tiny_positiveを追加した。既存列順は保持、既存カウンタ意味も維持。

## 次の論点

単なる予測飛び出しより、0<v<=VFloorで打ち切るpairの方が大きい。ここでthresholdだけを緩めるのではなく、局所の根の分離・branchを別途認証することでquartic再探索を省けるかが次の候補。ただし小さいvではendpointの精度不足もあり得るため、存在証明とphysical observable精度を両方確認する必要がある。現段階ではVFloorを変更していない。
