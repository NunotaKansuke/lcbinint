# Phase 12: D14 arithmetic scheduling研究（進行中）

基準c118a2a。1e-3でのVBM whole-epoch勝利は未達。
二つのcompile-time研究候補を個別に測定。既定経路は変更していない。

- HOLO_D14_INTERLEAVED_HORNER: presearchでP/P'の独立Horner chainを
  交互に評価する。それぞれの式・演算順序を維持し、命令の並列実行を狙う。
  微分をPの中間値から生成する別のrecurrenceへは変えない。
- HOLO_D14_RESIDUAL_MAX_NORM: qf残差の最大二乗ノルムを先に選び、
  平方根と共通正規化を最後に一回行う。qf多項式評価と閾値は同じ。

全7,216入力cold profile（RelTol1e-3、value None、CPU0、逐次実行）:

| candidate | presearch p50 ms | residual p50 ms | profiled whole p50 ms |
|---|---|---|---|
| baseline | .184454 | .029414 | .559647 |
| interleaved | .170859 | .029406 | .545673 |
| maxnorm | .184271 | .026479 | .560902 |

両候補とも全行でmu差0、convergence/node不一致0。
これはprofile付き単回測定でありwhole-epoch採用の根拠としては不足。
特にmaxnormはstageを削れてもwholeで勝てたとは言えない。
interleavedをprofileなし3 repeatsで全14,432行cold/warm/radial測定中。
参考baselineは同flagsのPhase11 whole_endpoint.tsv。

再現（他の計算をCPU0へ同時投入しない）:
```sh
c++ -O3 -DNDEBUG -std=gnu++17 -fext-numeric-literals -Isrc -march=native benchmarks/holonomic/bench_adaptive_stage_profile.cpp -o /tmp/p12_profile_base -lquadmath
c++ -O3 -DNDEBUG -std=gnu++17 -fext-numeric-literals -Isrc -march=native -DHOLO_D14_INTERLEAVED_HORNER benchmarks/holonomic/bench_adaptive_stage_profile.cpp -o /tmp/p12_interleaved -lquadmath
c++ -O3 -DNDEBUG -std=gnu++17 -fext-numeric-literals -Isrc -march=native -DHOLO_D14_RESIDUAL_MAX_NORM benchmarks/holonomic/bench_adaptive_stage_profile.cpp -o /tmp/p12_maxnorm -lquadmath
```
各binaryの引数はevidence/holonomic/v2_adaptive_best_trajectory_20260913/input_snapshot.tsv。
stdoutをevidence/holonomic/adaptive_extreme_phase12/profile_{base,interleaved,maxnorm}.tsvへ保存。
集計: `python benchmarks/holonomic/summarize_d14_arithmetic_profile.py`。
wholeはPhase10 runner_hybrid.cppに`-funroll-loops -ffp-contract=fast -fno-math-errno`
とinterleaved flagを追加し、input・output・3を引数に指定する。

## Profileなしwhole完了、採用は保留

1e-3 cold p50 .483153→.471667 ms、warm .433158→.428403 ms。
1e-4 cold .526939→.515294、warm .493633→.490124。
全14,432行×3laneでmu差0、status/node不一致0、value coverage同一。
1e-4のVBM参照超過3行も同じ。1e-3超過0。
p99は概ね同等だが、maxが56→62msへ増加しているため採用しない。
case0 uniformの初期行へ集中。case149の既存tailはほぼ不変。
whole_tail.tsvへ該当行を保存。case0/149の80入力でcandidate→baselineの
逆順再測定を開始し、計測変動か反復経路の変化かを次に確認する。
両研究flag付きのtest_d14_root_schedulingはpass。

whole集計:
```sh
python benchmarks/holonomic/summarize_adaptive_reciprocal.py --baseline whole_endpoint.tsv --candidate ../adaptive_extreme_phase12/whole_interleaved.tsv --output ../adaptive_extreme_phase12/whole_summary.json
```

## 逆順tail再測定

candidate→baseline、同80入力/160出力、3 repeats、CPU0。
case0 cold maxはbaseline54.034/candidate53.883ms、warm53.894/53.807ms。
case149 cold max56.237/57.286ms、warm56.148/56.429ms。
いずれもmu差0。初回case0の62msは再現せず、測定変動が疑われるが
原因を断定しない。研究flagを維持し、既定ONはまだ行わない。
次の残件は反復数・root parityの確認と、maxnormのwhole評価。
