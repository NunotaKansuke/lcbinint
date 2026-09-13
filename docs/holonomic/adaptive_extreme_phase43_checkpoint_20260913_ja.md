# Phase43: root-pair仕事量の正しい計測

基準f45ecfe。arc時間の多くを占めるroot trackingの仕事量を調べた。既存rootpair_newton_iterationsはNewton loopの外でpair通過ごとに増えており、名前と意味が違っていた。

## 変更・検証

rootpair_newton_iterationsを実際に試行したcorrector loop回数へ修正し、旧集計に相当するrootpair_newton_pairs_acceptedを追加した。後者はpair局所guard通過数であり、その後の集合branch/certificate成功を意味しない。determinant不良でbreakしたloopも試行として数える。

stage profiler末尾にtracking/fallback/guard/実反復数を追加。既存列名と順序は保持。profileなしの本番では計数しない。solver演算・受理gateは一切変更しない。

同じ7216-row trajectoryでPhase42候補と比較し、mu/ok/node数差0。全rowでiterations>=pairs_accepted、calls=warm_success+cold_fallsを機械チェックした。これはinstrumentationの検証で、kernelの新しい高速化ではない。

## warm steady-state実測

各profile2706行。先頭cold epochを除く。timing-only profile、RelTol1e-3、Phase42 compact storage有効。

|累積件数|uniform|LD|
|---|---:|---:|
|rootpair呼び出し|172682|195940|
|transport成功|128842|148540|
|quarticへ戻る|43840|47400|
|predictor拒否|9607|11182|
|Newton/局所guard拒否|8973|10748|
|branch reject集計|18611|21961|
|TMax関連拒否|255|254|
|VFloor関連拒否|7974|9746|
|実Newton loop試行|542028|600209|
|局所pair通過|148055|167900|
|quartic cold calls|15037|15037|
|quartic warm calls|45545|49295|

transport成功率はuniform74.61%、LD75.81%。guard列は排他的原因分類ではなく、branch_rejectはwarm失敗全般にも増える。これらを合計して拒否総数にしない。rootpair_cold_fallsには初回seed作成も含み、全てが追跡失敗ではない。quartic cold/warm countersも別consumerを含み得るのでrootpair件数と一致するとは限らない。

arc中央値はuniform.0699955 ms、LD.0807105 ms。内包rootpair時間は.0531095/.061727 ms。両者を加算しない。

## 次の候補

TMaxよりVFloor関連が多い。ordinary foldで(m,v)は正則なのにv閾値で追跡を失う局所ケースについて、実際のEO残差・determinant・符号・endpoint分離を調べる価値がある。ただしこの件数だけでguardが不要とは結論しない。thresholdをただ緩めて速度を作る変更はしない。foldで誤ったbranchや消えたpairを通さないことが必要。

## 再現

```
c++ -O3 -DNDEBUG -std=gnu++17 -fext-numeric-literals -Isrc -march=native -funroll-loops -ffp-contract=fast -fno-math-errno -DHOLO_D14_NATIVE_RESIDUAL_SCREEN -DHOLO_D14_DIRECT_CONVOLUTION -DHOLO_ADAPTIVE_STATIONARY_ENDPOINT -DHOLO_D14_WARM_NOISE_HANDOFF -DHOLO_D14_SCALAR_CONSTANTS -DHOLO_ARC_COMPACT_STORAGE benchmarks/holonomic/bench_adaptive_stage_profile.cpp -o /tmp/p43_profile -lquadmath
taskset -c 0 /tmp/p43_profile evidence/holonomic/v2_adaptive_best_trajectory_20260913/input_snapshot.tsv warm-timing > /tmp/profile.tsv
python benchmarks/holonomic/summarize_rootpair_work.py --input evidence/holonomic/adaptive_extreme_phase43/profile.tsv --baseline evidence/holonomic/adaptive_extreme_phase42/profile_candidate.tsv --output evidence/holonomic/adaptive_extreme_phase43/summary.json
```

CPU0単回profile。追加カウンタ自体の費用があるため微小なwhole speed差を主張しない。既存profileのnewton_iterationsと意味が変わった点に注意。raw/summaryはevidence/holonomic/adaptive_extreme_phase43/。
