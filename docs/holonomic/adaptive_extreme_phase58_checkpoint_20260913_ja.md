# Phase 58: quartic warm Aberthの固定次数コンパイル

基準b2b8c9d。hardware profileを受け、反復式を変えずに次数4をcompile-timeに固定する試作を行った。`aberth<R,FixedDegree=0>`のresearch templateを作り、HOLO_QUARTIC_FIXED_DEGREE時のwarm quartic degree=4だけ専用インスタンスへ渡した。次数が違う場合は汎用版へ戻す。cold solve、iteration上限、収束判定、root filtering、最終endpoint精度条件は変えない。Phase54中点有理式は双方で有効。

## 診断結果と判断

Phase56 subset256行で値/収束/node一致、小さい改善が見えたため全7216行warm-timingへ広げた。いずれも同HEAD/flags、CPU0、baseline→candidate→candidate→baseline。全ケースでも値/収束/nodeは完全一致。

以下は全7216行のうちtrajectory先頭を除く5412行のprofile中央値。V2ProfileScope付きなので通常whole runnerのタイミングとは区別する。

|run|uniform whole ms|uniform arc ms|LD whole ms|LD arc ms|
|---|---:|---:|---:|---:|
|base1|0.240613|0.067845|0.298911|0.078203|
|candidate1|0.240581|0.066439|0.298299|0.076705|
|candidate2|0.239924|0.066397|0.298956|0.076680|
|base2|0.241119|0.068080|0.299244|0.077987|

arcでは約1.4–1.7us減るがwhole差は小さい。特にLDのwholeはbaseline自身の変動と同程度。固定次数化だけで大幅な改善に繋がる証拠にはならず、今回の本線候補へ追加しない。sourceを戻しpatchとして残す。局所的な改善まで否定する結果ではなく、通常wholeでの十分な利益をまだ確認していないという判定。

subsetの利益だけで採用しなかったことが今回の重要な点。次にquartic自体を変える場合、compile-time unrollより反復・評価演算の中身が対象となる。ただし反復上限/精度を雑に緩めない。

今回は採用を見送ったためunit/独立reference/解析5Jac/cold/1e-4/通常whole runnerへは広げていない。production router、既存最速候補は変更なし。1e-3 uniformでVBMに勝つ目標は未達。

## 成果物と再現

`evidence/holonomic/adaptive_extreme_phase58/fixed_degree.patch`を基準commitへapply後、同ディレクトリのbuild_trial.sh、run_trial.sh、summarize.pyでsubset比較。run_full.shとfull/summarize.pyで全profile比較。raw再生成前に保存済rawを保護する。cacheというファイル名はcandidateを意味し、cache実装ではない。

full/には7216行×4実行、root directoryには256行×4実行を保存。summary.jsonは各分位点と全行parityを記録。
