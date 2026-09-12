# Phase20: real qf D14 event evaluation

基準7f88ca9。adaptive metadataでのd14_struct_eval(vq+0i)を
HOLO_D14_REAL_EVENT_EVAL研究flagで実数専用評価へ変更。
元のC3/G4/Z3、D14/D14primeの式、加算/乗算のgrouping、qf精度は維持。
根探索・residual・completeness・event分類のgateは変更しない。
試作時点では既定を変更せず、最終採用判断は末尾に記す。production routerは不変。

bench_d14_real_event.cppで全7216入力の各32実数点(v=1/8,...,4)、
230912点についてD,Dprimeのqf数値一致を確認（imagも0）。
これはeventそのものの位置サンプルではない。event consumerの検証は
別途whole及びreference比較で行う。
1評価のmicrobench中央値はcomplex 3.757us、real .597us。
同じbinary内でcomplex→real順に測るため順序/cacheバイアスはありうる。
whole勝利の根拠には使わない。

再現: bench_d14_real_event.cppを-O3 -DNDEBUG -std=gnu++17
-fext-numeric-literals -Isrc -march=nativeでcompile。
入力はv2_adaptive_best_trajectory_20260913/input_snapshot.tsv、CPU0。
wholeはPhase19のcompileに-DHOLO_D14_REAL_EVENT_EVALを追加。
3repeat、CPU0、同じ入力。referenceとJacobianは同じflagsのPhase17baselineと比較。
whole計測中の短いcorrectness実行はCPU1。

## Whole結果・採用

14,432行でmu/status/node差0。各tol7216/7216 value収束。
1e-3 cold p50 .4132525→.3900975 ms、warm .339915→.3176995 ms。
1e-4 cold p50 .4562005→.4338965 ms、warm .3888005→.365982 ms。
radial-only p50は1e-3 .120266→.120207 msでほぼ不変。
1e-3 warm p99は1.150877→1.159774 msと約9us増え、
他のcold/warm p99は短縮した。tail勝利とは主張しない。

reference550行（usable466、observed violation0）は同じcompiler flagsの
Phase17baselineと完全一致。解析5Jac220行もbyte一致。
VBM reference超過数は1e-3=0、1e-4=既存3で不変。

実数専用評価をadaptive metadata経路の既定に採用する。
旧経路A/BはHOLO_D14_DISABLE_REAL_EVENT_EVALで指定。
測定時のHOLO_D14_REAL_EVENT_EVALは既定化後不要。
固定n_r経路（retain_adaptive_metadata=false）とproduction routerは不変。
Phase17-19の別研究flagを既定に変更したわけではない。
今回のwhole候補値はPhase17-19研究flagを併用した値であり、
無指定のproduction全体性能とは混同しない。

既定化後のadaptive unit: 1484 checks/0 failures。
profile別1e-3 whole warm p50はLD .340269 ms、uniform .291227 ms。
まだVBMへの全面勝利ではない。次は約.158ms残るwarm topologyの内訳を
exclusive timerで確認し、未分類の固定費を追う。
