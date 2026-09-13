# Phase 63: cold quarticの初期半径短縮 — 不採用

基準29aae58。汎用cold quarticのCauchy円が大きすぎる仮説を、既存反復を変えずに試した。doubleかつdegree=4かつseedなしの場合だけ、descending係数a0,...,a4から

`F=2 max(|a1/a0|, sqrt(|a2/a0|), cbrt(|a3/a0|), sqrt(sqrt(|a4/a0|)))`

を求め、finite/positiveで既存Cauchy boundより小さいときだけ初期円半径に使う。Fujiwara型の保守的なboundで、厳密演算ならr=Fにおける低次項の総和は最高次項の高々1/2+1/4+1/8+1/16=15/16なので根は円内にある。binary64計算をoutward boundとは主張しない。これはseed選択であり最終根のcertificateを置き換えない。

反復式、40回上限、精度、real-root判定、後段endpoint/error ledgerは変更なし。D14 degree14にも適用しない。Phase59候補flagsを双方で有効。

## 試作2案

1. 全4係数からFを計算。
2. `2*|a1/a0| >= Cauchy bound`ならFが小さくならないのでsqrt/cbrt計算を省く。

2は同じseed選択をより安く得るための事前判定。1のpatchに2を重ねるのではなく、それぞれ基準からの独立patchとして保存。

## 診断入力結果

Phase56の256行、CPU0、baseline→candidate→candidate→baseline。RelTol=1e-3 warm-timing、先頭epoch除外のprofile中央値。いずれもmu最大scaled差2.058e-13、convergence/node差0。

|案/run|uniform whole ms|uniform arc ms|LD whole ms|LD arc ms|
|---|---:|---:|---:|---:|
|1 base1|0.229296|0.066227|0.303346|0.069023|
|1 candidate1|0.233132|0.066369|0.306473|0.069507|
|1 candidate2|0.231956|0.066514|0.299654|0.068834|
|1 base2|0.227755|0.065818|0.301738|0.069257|
|2 base1|0.228483|0.066005|0.304131|0.069003|
|2 candidate1|0.227342|0.065914|0.305755|0.069852|
|2 candidate2|0.227541|0.065989|0.303530|0.069616|
|2 base2|0.230478|0.066215|0.304393|0.068567|

arc時間に明確な改善がなく両案不採用。subsetで速くない候補を落とした結果であり、全caseで効果がないという証明ではない。boundが実際に小さくなった割合やsweep差は今回未測定で、どの原因が支配するかは未確定。全case/通常whole/reference/Jac/unitへは拡大せずsourceを戻した。最速候補/routerに変更なし。VBM勝利は未達。

## 再現

evidence/holonomic/adaptive_extreme_phase63のunguarded.patchまたはguarded.patchを基準commitへapplyし、build_trial.shとrun_trial.shで案1を再現。案2は同じ基準へguarded.patchをapply後、baseline binaryをbuild_trial.shと同じflagsで作り、build_guarded.shとrun_guarded.shを実行。summarize.pyは両者のrawを集計する。cacheというraw名はcandidateを意味する。raw再生成前は元ファイルを保護する。
