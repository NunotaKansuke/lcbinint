# D14 qf-warm root-wise trace と局所pair polish

## 結論

qf warmの24 sweep後にqf cold 400 sweepへ落ちる難例のうち、調べた24試行中2試行は、支配的な近接root pairを `(m,d^2)` の式で局所補正し、元のglobal qf step検証と既存certificateを通すことでqf coldを回避できた。該当したのはcase 149 / uniform / dbin 1 / epoch 15とcase 9 / uniform / dbin 2 / epoch 7のtrajectory-warm solveである。

この局所候補は対象2 epochの全体時間を約54–55 msから約4.6–5.2 msへ下げた。root set、cell/event数、value statusは一致し、最大scaled `mu` 差は `1.92e-12` だった。一方、同じtail corpusのfull-coldではqf coldを避けられず、局所試行費用が残った。warmでも他のtailが残りp99/maxは改善しなかった。したがって、これは**成功した限定的research候補**であり、production採用や全体集団の高速化とは結論しない。

## 対象と手順

開始時のbranch/HEADは `dev/holonomic @ c8c319e6a1ecb0093f03ac42a37406c5e016a265`。既存の `d14_overflow_whole_epoch_tail_20260912` trajectory snapshotと、そのroot-work benchmarkを使った。D14 warmがqfへ渡す有限14-root候補、24 sweepのrootごとのAberth step、Newton correction、`|P|`, `|P'|`, 最近傍間隔、root role hintを記録し、最終root-work表と割り当てて用途を照合した。

pair候補はqf warmがstep条件 `1e-20` を満たさず、既存scaled residualが `1e-12` 以下の場合だけ試す。最大Aberth correction rootとその最近傍rootを選び、対称な二根式

\[
E(m,d^2)=\frac{P(m+d)+P(m-d)}2,\qquad
O(m,d^2)=\frac{P(m+d)-P(m-d)}{2d}
\]

を8回以内のbacktracking Newtonで解く。局所pairの収束だけでは採用しない。pair step条件を通った時だけglobal qf Aberthを1 sweep実行し、そのglobal step、既存qf residual `<=1e-12`、scalar certificateをすべて通った候補だけを後段へ渡す。その後も既存のcompleteness、root-set、conjugacy/Vieta、physical topology検証をそのまま実行する。どのthresholdも緩めず、qf cold fallbackも削除していない。

最初のpilotでは全24候補にglobal verifierを走らせたが、pair Newtonが収束していない20試行ではglobal verifierも一度も合格しなかった。そこでresearch variantはlocal pair stepが収束した時だけglobal verifierを行うようにした。最終版は24試行中4試行がpair局所収束し、その4つだけglobal verifierを実行、2つが合格した。合格した2行のglobal verifierは各1 sweep。残り22試行は元のqf cold経路へfail-closedで戻った。

## root-wise観測

代表的なqf warm callでは14根すべてfiniteで、qf warm residualは小さいがglobal max-stepだけが規定値に残った。driver rootは単独で固定されず、少数root間を交替した。25個のtrace callすべてでdriver rootの最近傍rootがあり、scaled pair gap `sep / max(1, |m|)` は最大 `3.37e-4`、24/25は `1e-4` 以下だった。今回のtrace corpusに、local single-root Newtonを優先すべき明瞭なisolated driverは見つからなかったため、single-root routeは追加していない。

case 149 / epoch 15 / warmではdriver pair `10,11` が主で、pair center driftは約 `2.1e-47`、`d^2` の相対driftは約 `2.1e-15`。補正は2 local iterations / 14 backtracks、pair residualは `2.26e-36 -> 1.56e-64`、global verifier 1 sweepで合格した。両rootはfinal physical classifier上のpositive-real候補だった。

case 9 / epoch 7 / warmでは24-sweep traceのmodal pairは `12,13`、separationは約 `3.09e-9`、center drift `1.06e-15`、`d^2` relative drift `5.66e-7`。終端の最大correctionから再選択された局所pair `10,12` は2 local iterations / 1 backtrackで収束し、pair residual `3.98e-59 -> 1.20e-99`、global verifier 1 sweepで合格した。両rootもpositive-real候補だった。

局所candidateの全根集合はincumbent qf-cold root setとHungarian assignmentで照合した。case 149の最大root差は `1.21e-16`、case 9は `5.72e-16`。両方ともrole mismatch 0、completeness/root-count/conjugacy/Vieta failureは0で、topology status・cell数・event数も一致した。

## matched whole-epoch A/B

全14,432-rowを再実行する代わりに、qf-cold tailが確認されていた2 trajectoryから各4 epoch、計8入力rowを抜き出したtail-targeted比較である。runnerは各rowについてadaptive value-only (`GradientPolicy::None`, fold map/sample reuse有効)、`atol=1e-16`、`RelTol=1e-3` と `1e-4`、full-cold / L2 full-warm / radial-onlyを同じ入力で測る。各armは3 timed repeats/processを2回実行し、順序を入れ替えた。1 laneあたり16 rows（2 trajectory × 4 epoch × 2 tolerance）。CPUはXeon Gold 6530のcore 0に固定、GCC 11.5.0、両armの最適化flagは同一。

時間はms。summaryは2 processの各row medianのmedianである。

| lane | arm | p50 | p90 | p95 | p99 | max | 16 rows合計 |
|---|---|---:|---:|---:|---:|---:|---:|
| full-cold | baseline | 2.361 | 55.298 | 56.166 | 56.187 | 56.192 | 341.487 |
| full-cold | pair research | 2.360 | 55.737 | 56.651 | 56.742 | 56.765 | 344.307 |
| full-warm | baseline | 53.299 | 55.488 | 56.044 | 56.146 | 56.172 | 549.447 |
| full-warm | pair research | 4.752 | 55.514 | 56.502 | 56.564 | 56.580 | 353.529 |
| radial-only | baseline | 0.294 | 0.441 | 0.516 | 0.615 | 0.639 | 5.264 |
| radial-only | pair research | 0.288 | 0.439 | 0.515 | 0.612 | 0.636 | 5.228 |

full-warmでqf cold 400 sweepが消えた個別rowはcase149 epoch15（各toleranceで `54.27–54.31 -> 4.58–4.60 ms`）とcase9 epoch7（`54.61–54.97 -> 4.91–5.24 ms`）。full-coldではqf cold削減0行。対象subsetでfull-cold合計は約0.8%増えた。full-warmのp50改善はこの2 tail trajectoryを意図的に選んだ結果で、p90以上と最大値は他のqf-cold tailが残るため改善していない。この表を全corpusの速度予測へ外挿してはならない。

対象16 rowsのvalue convergence/status/stop、topology status/cell/event数、node/evaluation/panel/split数は3 laneすべてで一致した。full-warmの最大scaled `mu`差 `1.92e-12`、estimated-error差の最大 `6.21e-11`。cold/radialの `mu` とerror出力は一致した。

## 判断と次の焦点

pair local Newtonがglobal qf coldを安全に避ける例があることは確認できたが、24試行中2成功で、coldの利益がなく、whole-epoch tail最大値も下げていない。従ってresearch-only macroのまま保持し、default/routerには入れない。単根Newtonも、今回抽出したdriverはすべて近接rootを持ち、Aberth stepとNewton correctionの大きさもほぼ同じだったため、根拠を持って追加できるisolated caseがなかった。

残る最小課題は、pair local solverへ無差別に24回入るのではなく、qf warm traceに現れたcluster状態から「局所pairのequationsが収束可能か」を数値的に早く判定すること。ただし新しいheuristicを production routingへ入れず、次はrejected 22試行を見分けるearly reject条件をresearch A/Bし、同じglobal gatesとwhole-epoch測定を維持する。

## 検証と再現

`cmake --build build-holonomic-global -j2` 成功、`ctest --test-dir build-holonomic-global --output-on-failure -j2` は18/18 pass。trace/summary Python scriptsは `py_compile` 済み。research pair routeは `HOLO_D14_QF_PAIR_POLISH_RESEARCH=1` compile defineと同名runtime envの両方が必要で、default buildには含まれない。

raw timing/root tables、lossless gzip trace、filtered input、summary JSON、source hashes、exact commandは [`evidence/holonomic/d14_qf_warm_rootwise_20260912/`](../../evidence/holonomic/d14_qf_warm_rootwise_20260912/) にある。再生成手順は同directoryの `run_commands.txt`。
