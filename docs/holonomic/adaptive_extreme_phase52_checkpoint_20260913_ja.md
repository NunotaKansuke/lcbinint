# Phase 52: predictor拒否後の不要Newtonを省く — 不採用

基準84b5343。既存real_root_thetas_transportではpred_wildが立った場合、correctorの結果に関係なく最終的にok=falseになる。それでもcorrectorを実行していたため、既存の無条件拒否を早める試作を行った。

`HOLO_MV_EARLY_PREDICTOR_REJECT` でpred_wild && !dbgの場合だけNewtonへ入らずbreakする。次の候補setは元のwarm stateのコピーなので、その廃棄位置を早める。根の再探索や新しいfallbackを増やさず既存quartic経路へ進む。debug有効時は従来のtraceを残す。受理gate・VFloor・ステップ許容値は変更しない。diagnosticのNewton回数やcorrected_v分類は実際に評価しなくなった分だけ減る。

## A/B

同一HEAD・入力・flags、CPU0、compile終了後にbaseline→candidate→candidate→baseline。RelTol=1e-3、warm-timing、7216 rows。中央値は先頭epochを除く5412 rowsの値。wholeは全stageを含むがV2Profile timer付きであり通常whole runnerとは区別する。raw名cache1/cache2は前回scriptの名前を引き継いだもので、今回cacheは実装していない。

|run|uniform whole ms|uniform rootpair ms|LD whole ms|LD rootpair ms|
|---|---:|---:|---:|---:|
|baseline1|0.243064|0.052617|0.301788|0.061181|
|candidate1|0.243878|0.052617|0.304082|0.061318|
|candidate2|0.243768|0.052493|0.302415|0.060988|
|baseline2|0.243553|0.052667|0.301593|0.061642|

Newton総更新はuniform 542028→510851 (-5.75%)、LD 600209→565738 (-5.74%)。しかしwholeの改善は見えず、処理回数削減だけでは採用しない。削った計算の総費用が小さい、分岐/code layoutが相殺した等は仮説で、今回それらの原因を分離してはいない。

全4 runsで7216行のmu / convergence / nodesは完全一致。unitは1484 checks 0 failures。速度利益が見えないためsource変更を戻しpatchだけ保存した。cold/1e-4/reference/5Jacの追加検証および通常whole runnerでの採用試験には進まない。production routerと最速候補に変更なし。1e-3 uniformのVBM勝利は引き続き未達。

## 再現

基準上で `git apply evidence/holonomic/adaptive_extreme_phase52/early_predictor.patch`。
repository rootから同directoryのbuild_trial.sh、run_trial.sh、summarize.pyを順に実行（raw再生成時は既存rawを事前保護）。

unitコマンド:
```
c++ -O3 -DNDEBUG -std=gnu++17 -fext-numeric-literals -Isrc -DHOLO_MV_EARLY_PREDICTOR_REJECT tests/holonomic_cpp/test_adaptive_radial.cpp -o /tmp/p52_unit -lquadmath
/tmp/p52_unit
```

raw四本、summary.json、unit.txt、patchをevidence/holonomic/adaptive_extreme_phase52へ保存。summaryにはp50/p90/p95/p99/maxとNewton回数を記録する。
