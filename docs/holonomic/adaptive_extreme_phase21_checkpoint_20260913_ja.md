# Phase21: topology timing attribution

基準7e6deb5。solver・精度・production routerは変更せず、profileのみ分解。
prepare、conjugate cleanup、adaptive metadata、chart event、diagnosticのtimerを追加。
既存のdiagnostic処理（共役性、Vieta、cluster）はprofile時にだけ動く。
この費用を通常実行の削減余地に数えてはいけない。

7216入力、trajectory先頭を除く5412行のprofile中央値(ms):
prepare .001242、conjugate .0035695、metadata .012094、chart .000024、
diagnostic .030802、presearch .0337975、D14Real .022718、
residual .012035、completeness .000899。
solve_total .1070255、topology .156338。
各中央値は加算不可。行ごとにsolve_totalから既知stageを引いた
未分類部分のp50/p90/p99は .002276/.002383/.002555 ms。

Phase20の非profile wholeから初回epochを除いた同tolのwarm中央値は
topology .117290 ms、whole .284255 ms。
今回のprofile topologyからdiagnosticを引いた中央値は .126359 ms。
残る差にはtimer・他のprofile処理・別runの変動が含まれる。
この差し引き値を通常実行の実測値と呼ばない。

注意: 既存wholeのwarm列はtrajectory先頭のcold epochも含む。
今回はこの集計方法を勝手に変更せず、steady-stateを補助集計として追加した。
VBM勝利条件を都合よく縮小したものではない。

次の対象は実際に約34usかかるdouble presearch。
全根状態・停止条件を維持し、独立なinteraction項の計算をまとめて
SIMD化できるか検討する。sumの加算順は維持し、異常値検出も残す。

再現: bench_adaptive_stage_profile.cppをPhase20 wholeと同じflags、
HOLO_D14_NATIVE_RESIDUAL_SCREEN/DIRECT_CONVOLUTION/
HOLO_ADAPTIVE_STATIONARY_ENDPOINTでcompile。
input_snapshot.tsv warmを渡しCPU0で実行。
raw profile.tsv、summary.json、exclusive_summary.json参照。
