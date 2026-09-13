# Phase 55: 未収束D14Realの中間残差評価省略 — 不採用

基準ef3d47c。D14Realがfiniteだが未収束のとき、既存コードはqf全根残差を求めた後、残差の大きさに関係なくqf warmへ進み、再びqf全根残差を求める。最初の評価を省く案を測定した。

研究flag HOLO_D14_SKIP_REJECTED_RESIDUALは、EVENT_CONTRACT_RESEARCHが無効かつreal.converged=falseのときだけ中間worst_resにinfinityを置き、既存qf warmへ進む。qf warm後の残差計算・convergence・completeness・topology判定は変更しない。event候補記録研究では旧観測値が必要なので省略しない。

## 結果

Phase54の中点有理式を双方で有効化し、今回のdefine一個だけを変えた。CPU0、同HEAD/flags、compile後にbaseline→candidate→candidate→baseline。RelTol=1e-3 warm-timing全7216行。先頭epochを除く5412行で集計。

|run|uniform whole中央値 ms|LD whole中央値 ms|全5412行残差評価合計 ms|
|---|---:|---:|---:|
|base1|0.241056|0.299292|68.212668|
|candidate1|0.240849|0.299093|67.359594|
|candidate2|0.241306|0.300228|67.353822|
|base2|0.240747|0.299935|68.141553|

残差評価合計で約0.8ms減ったが、1行平均では約0.15us。これは全行で0.15usずつ省けたという意味ではなく、未収束行だけの変更を母集団で平均した値。wholeに一貫した改善はなく不採用。中間値の観測を変える複雑さを追加する利益が薄いためsourceは戻し、patchを保存した。

全4 runsでmu / convergence / nodesは7216行完全一致。今回のscopeは棄却判断用のprofile A/Bに限定し、独立root parity/reference/5Jac/unit/cold/1e-4の追加検証は行っていない。既存最速候補・production router・最終certificateは変更なし。1e-3 uniformでVBMに勝つ目標は未達。

## 再現

基準commit上で `git apply evidence/holonomic/adaptive_extreme_phase55/skip_rejected_residual.patch`。repository rootから同ディレクトリのbuild_trial.sh、run_trial.sh、summarize.pyを順に実行。raw再生成時は既存rawを保護すること。

cache1/cache2というraw名は前のscriptを引き継いだcandidate名でありcache実装ではない。通常whole runnerではなくV2ProfileScope付き計測である。summary.jsonに各stageの分位点と合計を保存。
