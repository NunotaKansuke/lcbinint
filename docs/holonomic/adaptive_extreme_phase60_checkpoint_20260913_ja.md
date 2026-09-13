# Phase 60: quartic warm seed精度の追加A/B — 1e-11を維持

基準bf1b156。既存研究flag HOLO_QUARTIC_WARM_SOLVE_TOLを1e-11（Phase59候補）と1e-9で比較した。source変更なし、既存の最終endpoint/accuracy/status条件は変更しない。

同HEAD/flags、CPU0、baseline→candidate→candidate→baseline。RelTol=1e-3の全7216行warm-timing。先頭epochを除く5412行のprofile中央値:

|run|uniform whole ms|uniform arc ms|LD whole ms|LD arc ms|
|---|---:|---:|---:|---:|
|base1 (1e-11)|0.229946|0.056988|0.285850|0.065769|
|candidate1 (1e-9)|0.229339|0.056135|0.286027|0.065342|
|candidate2 (1e-9)|0.229077|0.056090|0.285619|0.065170|
|base2 (1e-11)|0.229309|0.056933|0.286752|0.065955|

全7216行でstatus/convergence/node変化0、max |delta mu|/max(1,|mu|)=7.734e-12。arcの追加短縮は約0.5–0.9usで、wholeは測定変動と同程度。Phase59の約10us短縮とは異なり、追加でseed精度を下げる十分な利益は示さない。1e-9は採用せず、1e-11の研究候補を維持する。

今回のstatusはprofileのvalue convergenceであり通常wholeの全numerical_status監査ではない。独立reference/5Jac/unit/cold/1e-4/通常wholeは未実施。速度利益が薄いのでそこへ追加計算を進めなかった。validate_trial.shは準備した未実行コマンドであり実行済みvalidationの証拠ではない。

再現はevidence/holonomic/adaptive_extreme_phase60のbuild_trial.sh、run_trial.sh、summarize.py。full/にraw四本、profile_summary.jsonに分位点と値差を保存。cache1/cache2はcandidateを意味するファイル名。元のrawを再生成する場合は先に保護する。

今回production/router/最速候補の変更なし。1e-3 uniform VBM勝利目標は未達。
