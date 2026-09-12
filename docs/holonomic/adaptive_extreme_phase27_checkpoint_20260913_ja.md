# Phase27: warm限定のnoise-scale handoff

基準62e5526。HOLO_D14_WARM_NOISE_HANDOFFを追加し、前epochのseedがある場合だけPhase26の予備探索終了条件を使う。coldでは追加proxy計算も行わない。これはcandidateの精度昇格判断であり、受理certificateではない。D14Real/qfの最終gate、production router、既定設定は変更しない。

## Whole epoch

比較baselineはadaptive_extreme_phase20/whole_candidate.tsv。入力はv2_adaptive_best_trajectory_20260913/input_snapshot.tsv。14432行、repeat3、CPU0、D14/topologyを含む。warm列は各trajectoryの最初のcold epochも含む。

|RelTol|lane|baseline p50 ms|candidate p50 ms|baseline p99 ms|candidate p99 ms|
|---|---|---:|---:|---:|---:|
|1e-3|cold|0.390098|0.389410|3.178741|3.181767|
|1e-3|warm|0.317700|0.299171|1.159774|1.050455|
|1e-4|cold|0.433897|0.432451|3.214608|3.219027|
|1e-4|warm|0.365982|0.345696|1.396692|1.299394|

coldの値は完全一致。warm最大relative mu差4.13e-12。全行value収束、status/node差0。保存済みVBM reference超過数は1e-3=0、1e-4=既存3のまま。cold時間の小差を高速化/非回帰証明と解釈しない。これは保存baselineとの比較で、交互実行による反復統計ではない。

## Correctness

14432根集合で最大scaled root差2.21e-14、role/physical/event/cell/completeness差0。qf-cold100→94、6行で変化。根座標はbinary64 export比較であり区間包含証明ではない。
独立reference550行のうちusable466、observed violation0、mu/stop差0。解析5Jac220行も値・品質分類一致。adaptive unit1484 checks、0 failures。

研究flagとして保持し、まだVBM勝利とはしない。特にuniformの固定費が残る。root比較rawとwhole/reference/gradient summaryはevidence/holonomic/adaptive_extreme_phase27に保存。

## 再現

wholeはrunner_hybrid.cpp（adaptive_estimator_phase10_20260913）を
-O3 -DNDEBUG -std=gnu++17 -fext-numeric-literals -Isrc -march=native -funroll-loops -ffp-contract=fast -fno-math-errno
にHOLO_D14_NATIVE_RESIDUAL_SCREEN、HOLO_D14_DIRECT_CONVOLUTION、HOLO_ADAPTIVE_STATIONARY_ENDPOINT、HOLO_D14_WARM_NOISE_HANDOFFの各-Dを付け、-lquadmathでbuild。
`taskset -c 0 /tmp/p27_whole evidence/holonomic/v2_adaptive_best_trajectory_20260913/input_snapshot.tsv evidence/holonomic/adaptive_extreme_phase27/whole_candidate.tsv 3`

rootはbenchmarks/holonomic/bench_d14_root_work_detail.cppにnative residual screenとwarm noise handoffを指定し、同入力で `1 both`。
`check_d14_reciprocal_parity.py`でphase17のroots.tsv.gz/events.tsv.gzと比較。
referenceはcheck_adaptive_reference.cpp、gradientはbench_warm_gradient_parity.cppを-O3（march/unrollなし）、warm noise flag付きでbuild。入力adaptive_radial_20260911/reference_cases.tsv。referenceのみ末尾warm指定。
