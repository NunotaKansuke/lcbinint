# Phase 61: cold quarticはwarm失敗ではなくseed不在が支配

基準0f87e35。HOLO_QUARTIC_WORK_PROBEでreal_root_thetas_warm内のcold入口とcold/warm sweep総数を分類した。数値演算・seed・iteration上限・精度条件に変更なし。diagnostic時のみ既存Aberthのiterations出力を受け取る。

分類順は、呼出し入口でQuarticWarm.valid=false→unseeded、validだがdegree相違→degree_change、cold_streak>=kColdStreak→disabled、それ以外でwarmを試した後coldへ進む→after_warm。unseededはcell初回だけを保証する分類ではなく、明示的なseed無効化も含む。

## 実測

Phase59候補（quartic warm solve tol1e-11、中点有理式ほか）を基準にRelTol=1e-3 warm-timing、全7216行。先頭epochを除く各profile2706行で:

|profile|cold calls|unseeded|after warm reject|disabled/degree change|cold sweeps|warm calls|warm sweeps|
|---|---:|---:|---:|---:|---:|---:|---:|
|uniform|15037|15037|0|0|353459|45555|236736|
|LD|15037|15037|0|0|353459|49307|261046|

coldは平均23.506 sweep/call、warmはuniform5.197、LD5.294。ここではwarm失敗後のcoldはゼロである。したがって「warmの失敗を救ってcoldを減らす」変更はこのcorpusの当該経路では対象を外す。root-pair transport失敗からquartic warmへ落ちる件数は別であり、この結果をroot-pair failure=0と混同しない。

cold回数は一行平均約5.56でありpanel/cell初期化との関係が疑われるが、今回panel IDとの直接対応は記録していない。すべてcell初回と断定しない。

今後の有望な検討対象は、最初のquartic seedを安く得ること。D14 physical eventの既存接触情報からdeflated quadraticとroot pairを作れるか等は候補だが未実装・未検証。seedが不完全なら新しいfalse successを起こすため、既存の完全性・branch判定を省く理由にはしない。

## 検証・限界

Phase60の1e-11 baselineと全7216行でmu / convergence / nodes完全一致。adaptive unitは1484 checks 0 failures。新しい精度変更はないため独立referenceの再実行なし。新しい高速化の採用もない。既存最速候補・production routerは維持。VBM勝利目標は未達。

probeには計測費用があり、今回のstage timingから速度改善を主張しない。今回分類したのはreal_root_thetas_warmからのcold solveで、他APIの直接cold Aberthを含む全solver分類ではない。

## 再現

repository rootでevidence/holonomic/adaptive_extreme_phase61/build.sh、その後:
```
taskset -c 0 /tmp/p61_profile evidence/holonomic/v2_adaptive_best_trajectory_20260913/input_snapshot.tsv warm-timing > evidence/holonomic/adaptive_extreme_phase61/profile.tsv
python benchmarks/holonomic/summarize_quartic_work.py --baseline evidence/holonomic/adaptive_extreme_phase60/full/base1.tsv --probe evidence/holonomic/adaptive_extreme_phase61/profile.tsv --output evidence/holonomic/adaptive_extreme_phase61/summary.json
c++ -O3 -DNDEBUG -std=gnu++17 -fext-numeric-literals -Isrc -DHOLO_QUARTIC_WORK_PROBE -DHOLO_QUARTIC_WARM_SOLVE_TOL=1e-11 tests/holonomic_cpp/test_adaptive_radial.cpp -o /tmp/p61_unit -lquadmath
/tmp/p61_unit
```
既存rawを再生成する場合は保護してから実行する。
