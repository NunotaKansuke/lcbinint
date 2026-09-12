# D14 overflow修正のwhole-epoch効果と残存qf-cold tail

## 判断

1. **overflow修正は残す。** 同一14,432-rowのfull-cold/full-warm比較で、topologyのstatus・cell数・event数は全行一致し、value coverageはsafe側14,432/14,432だった。legacy側はwarmの4行で `GRADIENT_UNRELIABLE / EventLocationLimited` となり、value convergenceを失った。case 92ではlegacy側のdouble presearchが非有限になってqf coldへ落ちる一方、safe側はqf coldを使わず、whole-epoch最大時間を約29.9 msから約2.3 msへ減らした。
2. **中央値の高速化とは言えない。** 全行を混ぜたfull-coldのp50は0.534 msから0.568 ms（+6.3%）、full-warmは0.492 msから0.522 ms（+6.2%）。通常域の安定性・statusと大きなtailを改善する代わりに、robust double quotientに中央値コストがある。
3. **残る最大tailはoverflow事故ではない。** case 0/64/149とwarm case 9のtraceではdouble seed、D14Real、qf warm、qf coldの根は有限で、最初の非有限updateは観測されなかった。qf warmの14-root candidateは残差certificateを通るが24 sweep後も厳しいstep条件を満たさず、`solve_d14()` がCauchy seedでqf coldを開始する。close-root interactionと全root共通のstep停止条件が関与しているが、係数/根の条件の問題と反復方式の問題の寄与はまだ分離できていない。
4. **次に見る最小箇所は `solve_d14()` のqf warm rejectからqf cold seedを選ぶ部分と、`aberth_d14_struct()` のroot-wise step推移。** 有限warm candidateをcold側へ継続するだけのA/Bは全epochで遅くなったため採用しない。step/residual/completeness/topology gateを緩めず、close clusterに対する反復仕事の配り方を調べる。

## 比較条件

作業開始時に `dev/holonomic`, `origin/dev/holonomic`, `FETCH_HEAD` はすべて `56d1be6ce857ee623bb32550befcfac9a4c46ac4` で一致した。比較対象はこのcommitのsafe complex quotientと、同じsourceを `HOLO_D14_FORCE_LEGACY_DOUBLE_DIV=1` でビルドして旧binary64 quotientだけを戻したbaselineである。DD、D14Real、qf、入力、値only adaptive設定は共通。

入力は1,804 trajectory × 4 epoch、linear/uniform profile、`RelTol=1e-3, 1e-4` の合計14,432 rows。各rowを3回計測し中央値を保存した。設定はvalue-only (`GradientPolicy::None`, `atol=1e-16`)、fold map/sample reuse有効。GCC 11.5.0、Intel Xeon Gold 6530、whole-epoch runnerは `taskset -c 0-7`。両runとも `HOLO_D14_LOCAL_PAIRS=1`, `HOLO_D14_ACTIVE_PRESEARCH=1`, `HOLO_D14_ACTIVE_TOL=1e-12`, `HOLO_D14_ACTIVE_PATIENCE=2`。入力SHA256は `6646492658ed416e75cb8c6c821ee853f79cdf7761fc41441186aa67b81fa05b`。

初回取得rawも `overflow_*_initial_binary.tsv.gz` として保存したが、runner sourceを凍結して再ビルドしたbinaryと初回binaryのhashが一致しなかったため、**以下の主結果はrun_commands.txtの固定runner・現HEADでsafe/legacyを再実行したraw**から計算している。summaryには再実行binaryとrunner sourceのhashを記録する。

同じsafe binaryで追加のsafe-only whole runもあり、primary safe runとのp50差はcold +0.0013 ms、warm +0.0011 ms、radial +0.0005 msだった。これはpaired A/Bではないため主summaryには混ぜず、[safe_unpaired_repeat_summary.json](../../evidence/holonomic/d14_overflow_whole_epoch_tail_20260912/safe_unpaired_repeat_summary.json) とrawだけ補助証拠として残す。

`full-cold` は各epochでD14/topologyを作り直す。`full-warm` はtrajectory内の前epoch D14 rootをL2 warm seedに使い、各epochでtopologyを解き直す。`radial-only` はtopologyを事前構築しadaptive積分だけを測る。legacy compile switchはD14だけでなくbinary64 complex division全体を戻すため、radial-onlyにも影響し得る。したがってradial-onlyはD14 stageを除いた比較であり、未変更コードのnoise controlではない。

## Whole-epoch結果

値はms、各セルを `safe / legacy` の順で示す。

| lane | p50 | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|
| full-cold | 0.568 / 0.534 | 0.937 / 9.934 | 1.131 / 12.824 | 3.530 / 19.241 | 56.527 / 57.625 |
| full-warm | 0.522 / 0.492 | 0.887 / 0.897 | 1.070 / 1.203 | 2.002 / 14.150 | 56.218 / 55.037 |
| radial-only | 0.168 / 0.164 | 0.349 / 0.339 | 0.439 / 0.426 | 0.716 / 0.695 | 2.258 / 2.145 |

topology status/cell/event数の不一致は全laneで0。value convergenceはsafe側がcold/warm/radialすべて14,432/14,432。legacy側もcold/radialは全件収束したが、warmはcase 45、linear/uniform、`d_bin=1`, epoch 15の `RelTol=1e-3,1e-4` で計4行失敗した。legacyは `GRADIENT_UNRELIABLE / EventLocationLimited`、safeは `OK / Converged`。両方収束した行での `|Δmu| / max(1, |mu_legacy|)` は最大 `9.89e-12`、p99は約 `1.15e-12`。

whole-epoch timing内のtopology stage p50はcold `0.372 / 0.326 ms`（safe +14.0%）、warm `0.252 / 0.233 ms`（+8.1%）。paired topology差の中央値はそれぞれ+0.0425 ms、+0.0132 msだった。これは以前のD14専用計測で見ていた3–5%より大きく、今回のwhole-epoch測定では隠れなかった。radial-onlyもp50で約2.3%遅く、adaptive stage p50もwarmで約3.5%遅い。よって中央値コストはtopology/D14だけでは説明できない。legacy switchがquartic/root-pair等のbinary64 complex divisionにも適用されるため、そのdivisionの全利用箇所と3-repeat計時の揺れはこのrunだけでは分離できない。

case 92だけを見ると80 rows。safe側は全laneでcoverage 80/80、legacy warmはcoverage 80/80だがrootworkではcold 16回/warm 4回のpresearch nonfinite→qf coldが観測された。safeではこのcaseのqf coldは0回。cold whole-epochのsafe/legacy p50,p90,p95,p99,maxは `0.792/0.915`, `1.598/29.186`, `1.659/29.349`, `2.384/29.953`, `2.450/30.013 ms`。warmは `0.730/0.775`, `1.578/4.617`, `1.660/26.716`, `2.260/29.343`, `2.275/29.837 ms`。従ってcase 92では中央値はほぼ同等でも、legacyの高percentile tailがsafeで消えた。

## qf-cold発生数と残存tail

full D14 rootwork tableは各lane 7,216 rows（2 profileを含む）。この専用計測での `qf cold calls / qf cold sweeps / 400-sweep rows` は次のとおり。

このrootwork rawはbase commitに含まれる [case92 overflow evidence](../../evidence/holonomic/d14_case92_presearch_overflow_20260912/) の `full_safe_*` / `full_legacy_double_*` で、同directoryの `sha256_manifest.txt` と `run_commands.txt` で照合できる。今回のtaskではcase92 countと難例をそこから取り、qf warm/coldのsweep診断はcurrent sourceで採り直したfocused traceから補っている。

| lane | safe | legacy |
|---|---:|---:|
| cold | 58 / 12,498 / 22 | 944 / 87,140 / 22 |
| warm | 44 / 12,912 / 26 | 256 / 28,876 / 20 |

trajectory benchmarkは2つのtolerance targetを個別実行するので、このrootwork件数を単純に2倍するとsafe/legacyのwhole runでcold laneは116/1,888回、warm laneは88/512回相当となる。これはrootwork表からの換算であり、trajectory rawに直接記録されたcounterではない。safeでも400-sweepの最悪群は残り、coldはcase 0:14、case 64:2、case 149:6 rows、warmはcase 0:14、case 9:2、case 64:2、case 149:8 rows。

top tailの代表trace:

| case / lane | double presearch | D14Real | qf warm → qf cold / reported clusters | close pair / 最終cold最大step |
|---|---|---|---|---|
| 149 uniform dbin1 epoch7, cold | 14/14 finite, 200 sweeps | finite, 25 update/root, nonconverged | 24 sweep, max step `3.46e-16`, residual `7.4e-27`; reject → Cauchy seedから400 sweep; clusters 0 | sep `1.90e-5`; root 6/7; `|P|=2.26e-36`, `|P′|=7.59e-16`, `|S|=6.04e3`, `|P′−PS|=7.59e-16`, step `2.97e-21` |
| 0 linear dbin0 epoch0, cold | 14/14 finite, 200 sweeps | finite, 25 update/root, nonconverged | 24 sweep, max step `8.80e-13`, residual `2.76e-19`; reject → Cauchy seedから400 sweep; clusters 0 | sep `1.06e-5`; root 1/3; `|P|=4.51e-36`, `|P′|=3.82e-15`, `|S|=1.64e4`, `|P′−PS|≈|P′|`, step `1.18e-21` |
| 64 linear dbin0 epoch0, cold | 14/14 finite, 200 sweeps | finite, 25 update/root, nonconverged | 24 sweep, max step `1.04e-15`, residual `5.17e-20`; reject → Cauchy seedから400 sweep; clusters 0 | sep `8.01e-6`; root 1/3; `|P|=7.35e-40`, `|P′|=2.50e-18`, `|S|=2.43e4`, `|P′−PS|≈|P′|`, step `2.94e-22` |
| 9 uniform dbin2 epoch7, warm | 14/14 finite, 60 sweeps | finite, 25 update/root, nonconverged | 24 sweep, max step `1.72e-18`, residual `1.22e-26`; reject → 400-sweep cold; clusters 1 | sep `3.09e-9`; root 8/7; `|P|=2.79e-58`, `|P′|=4.75e-37`, `|S|=1.19e6`, `|P′−PS|≈|P′|`, step `5.87e-22` |

ここで最初に失敗するのは非有限演算ではなく、qf warmの上限24 sweep時点でstep convergence flagがfalseになること。qf coldの400 sweepでもcase 0/64/149/9は全14根finiteで、`D14QF_FIRST_BAD` は0件。case 149の400 sweep終端step `2.97e-21` は既存 `1e-22` 閾値を超えるが、qf residualとscalar certificateは通る。`|P′−PS|` は `|P′|` とほぼ同じで、trace上はAberth denominatorの大きな相殺がtailの直接原因とは見えない。case 9の極小root separationと大きい `S` はcluster conditioningの関与を示唆する。case 0/64/149はsep約`1e-5`であり、現時点で「本質的な悪条件だけ」と断定する材料はない。

case149でexpanded/blockを個別比較するとtopologyは双方14 cells/13 eventsで一致したが、root-set Hausdorff差は`7.84e-8`。case 0は`2.86e-12`、case 64は`4.15e-15`。case149のexpanded presearch seedはfinal rootとのassignmentが14中6のみで、最大移動は約`8.1e-3`だった。これらは近接rootで表現/seed依存性が上がる証拠だが、差そのものが誤根を意味するわけではない。

## 試した案と不採用理由

qf warmがfiniteでscalar residual gateを通る場合に、そのcandidateをqf coldのCauchy seedの代わりに使うcompile-time A/Bを実施した。qf cold総sweepはcold 12,498→11,104（−11.2%）、warm 12,912→11,650（−9.8%）だが、qf cold call数は変わらず、400-sweep rowもcold 22→26、warm 26→26。14,432-row whole epochではstatus/topology一致、最大scaled `mu`差`1.46e-12`だった。coldのp50/p90/p95/p99/maxは`0.56772/0.93698/1.13088/3.53016/56.527`から`0.56908/0.93464/1.11210/3.49254/58.091 ms`、warmは`0.52191/0.88712/1.07006/2.00215/56.218`から`0.52304/0.88403/1.06129/2.11124/58.611 ms`。中央値は僅かに遅く、warm p99とcold/warm maxも悪化したため不採用。

scalar polynomial certificateだけで未収束warm setを受理する案も不採用。case49で6行のtopology mismatch（1 event/cell欠落）が出た。case49 dbin1 epoch0では、正しいphysical roots `1.0703e-8`, `2.1679e-9` に対して誤候補が複素根/負実根へ寄り、root correctionがnearest separationと同程度だった。local/scalar residualだけではphysical root classificationの代わりにならない。

## 変更と検証

production router、固定`n_r`経路、qf tolerance、residual/completeness/topology gateは変更していない。追加したC++変更は `HOLO_D14_TRACE_QF_ITERATIONS` 定義時だけ動くqf per-sweep traceと、trace対象行の識別出力である。通常buildではtrace処理はcompile-time除外される。qf candidate continuation案は全epochで勝たなかったため、実装コードは作業treeから除き、再現用patchとrawだけ保存した。

`cmake --build build-holonomic-global -j2` 成功、`ctest --test-dir build-holonomic-global --output-on-failure -j2` は18/18 pass。trace-enabled rootwork binaryもcompile・case149実行済み。raw TSVはgzipでlossless圧縮し、入力 snapshot、集計JSON、trace、compile/runコマンド、hash manifestを `evidence/holonomic/d14_overflow_whole_epoch_tail_20260912/` に保存した。

詳細な再現コマンドは同directoryの [run_commands.txt](../../evidence/holonomic/d14_overflow_whole_epoch_tail_20260912/run_commands.txt)、whole-epoch summaryは [whole_epoch_summary.json](../../evidence/holonomic/d14_overflow_whole_epoch_tail_20260912/whole_epoch_summary.json)、root tail集計は [qf_tail_summary.json](../../evidence/holonomic/d14_overflow_whole_epoch_tail_20260912/qf_tail_summary.json)。
