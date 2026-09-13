# Phase 51: endpoint exact two-entry cache A/B — 不採用

基準 06478d4。Phase 50で同一引数再訪がphi評価の約5.7%だったため、実際の保存・照合費用込みで利益が出るか、小さい試作を行った。

## 試作と判定

`HOLO_ENDPOINT_TWO_CACHE` はpolish_endpoint呼出し内の直近2個の異なる引数とPhiValDthetaを保存。値とsignbitが同じ引数だけ再利用する。Newton更新・6反復・stationary/small-step/reliability判定は変更しない。最終評価も同じキャッシュを利用。NaNは一致しない。反復を打ち切る近似ではない。

全7216 trajectory rowsをwarm-timingで baseline→cache→cache→baseline の順にCPU0で測定。コンパイルは同じHEAD・同じflagsでcache defineだけ相違。compile終了後にtiming開始。各trajectory先頭を除く5412 rowsの中央値を下表に示す。全費用を含むwholeだが、V2Profileの計測費用が入るため通常runnerのwallとは区別する。

|run|uniform whole ms|uniform endpoint ms|LD whole ms|LD endpoint ms|
|---|---:|---:|---:|---:|
|base1|0.243152|0.021308|0.300591|0.024715|
|cache1|0.243104|0.021081|0.301477|0.024598|
|cache2|0.243286|0.021126|0.300280|0.024396|
|base2|0.243534|0.021241|0.301839|0.024584|

endpointで約0.1–0.3 usの差は見えるがwholeはbaseline自身の変動と同程度。大幅改善につながらず、通常経路に状態・照合を追加する理由が薄いので不採用。source変更は完全に戻した。試作はraw隣のtwo_cache.patchで保存。

全4 runs、7216 rowsのmu / convergence / node countは完全一致。今回は不採用を決める仕事量・性能実験に限定し、1e-4/cold/analytic5Jac/独立reference/unitの追加検証は行っていない。上限到達が丸めによる振動か真の未収束かの完全分類も未達。キャッシュ結果だけを根拠に反復上限・誤差条件を緩和しない。

## 再現と成果物

基準commit上で `git apply evidence/holonomic/adaptive_extreme_phase51/two_cache.patch`。
その後同ディレクトリの `build_trial.sh`、`run_trial.sh`、`summarize.py` をrepository rootから順に実行。run scriptは既存rawを上書きするため、保存済rawを別ディレクトリへ保護してから実施すること。

inputs/flagsはbuild/run scriptsに固定。入力はv2_adaptive_best_trajectory_20260913/input_snapshot.tsv、RelTol=1e-3、前epoch D14 rootsを再利用する既存L2経路。raw四本とsummary.jsonに各stageのp50/p90/p95/p99/maxを保存。

最速候補とproduction routerは変更なし。1e-3 uniformでVBMに勝つ目標は引き続き未達。今回の結果から、endpointの単純な同一引数cacheを掘り続ける優先度は下げる。
