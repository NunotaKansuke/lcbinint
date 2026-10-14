# Phase33: 部分precision評価の速度とfirst-sweep試作（不採用）

基準4c908e5。Phase32のmethod7の費用を測定し、その後D14Real mixed Aberthの最初の1 sweepだけに組み込んだ。既定OFFの試作は最終的に撤去しprototype.patchに保存した。

## 評価器microbench

7216-case saved candidateから固定hashで2082点抽出、同じ点・係数を事前準備。1回warmup除外、8回交互順、各32周、CPU0。method0は全D14Real、method1はdouble blocks→D14Real組合せ。中央値は約498 ns→369 ns。係数構築、Aberth interaction、受理検査、全epochはこの測定に入らない。非有限0。shared inputsとメモリfootprintを含むmicrobenchでありproduction stageの26%高速化を意味しない。

## solver A/B

最初の1 sweepに限りdouble blocksを使用し、以後は従来D14Real。scheduled root経路は従来通り。非有限p/dpはその場で従来評価へ戻す。cheap sweepだけでは収束を宣言せず、少なくとも従来precisionのsweepを通す。最終qf/residual/completeness gateは不変。

7216行warm profileでstatus/node差0、最大scaled mu差1.78e-12。
steady p50: D14Real .021605→.028743 ms、whole .307972→.313754 ms。
全row合計: D14Real 230.42→258.10 ms、qf 1389.37→1531.29 ms、whole 4262.45→4445.25 ms。

単体評価を軽くしても、追加補正とqf再計算の費用に負けた。profileだけで不利が明確なので、追加full benchmark/referenceは行わず不採用。精度条件を緩めて救済しない。

次は全D14Real精度を維持したまま、次数3/4のC/G/Zと微分でvの冪・共通項を共有できるかを調べる。既存精度を落とす案と演算数削減を分ける。

## 再現

micro: Phase32 probeを-O3 -DNDEBUG -std=gnu++17 -fext-numeric-literals -Isrc -march=native -DHOLO_D14_SCALAR_CONSTANTSでbuild、inputはPhase31 prepare生成物。
`taskset -c 0 /tmp/p33_probe /tmp/p31_input.tsv timing`
profile: Phase28のprofile flagsへHOLO_D14_FIRST_MIXED_SWEEP追加、bench_adaptive_stage_profile.cpp、input_snapshot.tsv warm。baselineは同flagsからFIRST_MIXEDだけ除外したbinary。candidate→baselineの順でCPU0。rawと集計を本evidenceに保存。stage profileには診断の固定費を含む。
