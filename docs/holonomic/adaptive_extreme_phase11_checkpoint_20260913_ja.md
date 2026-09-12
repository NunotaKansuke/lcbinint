# Phase 11: whole-epoch 1e-3 VBM勝利へ向けた進行中記録

基準HEAD f0b4f12。目標は未達であり、本checkpointは完了報告ではない。

既存描画scriptと同じVBM対応行を使った目標再確認:
- uniform VBM中央値0.061369 ms、現行warm whole中央値0.425728 ms。
- linear VBM中央値0.314456 ms、現行warm whole中央値0.487586 ms。
- 行ごとのVBM/V2比の中央値はそれぞれ0.1604と0.7191。
これらは既存保存timingの比較で、今回同時測定したVBMではない。

通常adaptive value None、RelTol=1e-3を全7,216入力でprofileした。
D14 presearch中央値0.20956 ms、D14Real 0.02337 ms、qf residual
0.02939 ms、radial physics 0.11015 ms。profile内のstageには包含関係が
あり全列を足してはならない。またprofile時だけ走るroot診断があるため、
whole値を通常benchmarkの速度と同一視しない。

既存記録から固定sweep短縮はqf費用増加で棄却済みと確認。
今回の新候補はHOLO_D14_PRESEARCH_RECIPROCAL compile flag。
active presearchの相互作用ループで既にfinite/positiveを確認したdn2を
再利用し、分子1の複素除算の重複検査を省く。反復順序・全根保持・
後段精度gateは維持する。通常buildは既存経路。

profile A/B:
- presearch p50 0.20956 -> 0.18435 ms。
- whole（profile付き）p50 0.59284 -> 0.56465 ms。
- D14Real中央値0.02337 -> 0.02349 ms。
- physics中央値0.11015 -> 0.11061 ms。
- value coverage 7216/7216両方、node mismatch 0。
- mu絶対差最大2.82e-10。独立reference再検証・root/event parityは未実施。

採用未決。次はprofile無効のwhole cold/warm matched複数repeatと
root/event/reference audit。以降は予備探索・qf残差・geometryの費用を削る。

再現（CPU 0、単一processで逐次実行）:
```sh
c++ -O3 -DNDEBUG -std=gnu++17 -fext-numeric-literals -Isrc -march=native benchmarks/holonomic/bench_adaptive_stage_profile.cpp -o /tmp/adaptive_stage_profile -lquadmath
c++ -O3 -DNDEBUG -std=gnu++17 -fext-numeric-literals -Isrc -march=native -DHOLO_D14_PRESEARCH_RECIPROCAL benchmarks/holonomic/bench_adaptive_stage_profile.cpp -o /tmp/adaptive_stage_recip -lquadmath
taskset -c 0 /tmp/adaptive_stage_profile evidence/holonomic/v2_adaptive_best_trajectory_20260913/input_snapshot.tsv > evidence/holonomic/adaptive_extreme_phase11/profile_cold.tsv
taskset -c 0 /tmp/adaptive_stage_recip evidence/holonomic/v2_adaptive_best_trajectory_20260913/input_snapshot.tsv > evidence/holonomic/adaptive_extreme_phase11/profile_reciprocal.tsv
```

前回報告の訂正: hybrid estimatorはweighted detailを最大1/8へ減らす
経験的error modelの変更であり、「数値保証を一切弱めない」とは言えない。
独立reference判定可能458行で違反0という有限corpusの観測と、厳密保証は
区別する。今後もreference不使用/未収束行を隠さず評価する。

## Profileなしwhole A/B（3 repeats）

同一GCC flags、CPU0でbaseline完了後candidateを逐次実行した。
両方ともwarm-up trajectoryは測定から除外。stage値はwhole中央値の
repeatから採取する既存harnessを利用。

| Tol | lane | p50 baseline→candidate ms | p90 | p99 |
|---|---|---|---|---|
| 1e-3 | cold | .516039→.484884 | .827129→.794610 | 3.396251→3.250189 |
| 1e-3 | warm | .449440→.435863 | .752661→.736640 | 1.287009→1.260746 |
| 1e-3 | radial | .126066→.126015 | .248816→.248637 | .393171→.390918 |
| 1e-4 | cold | .560362→.529382 | .897242→.865568 | 3.460589→3.275376 |
| 1e-4 | warm | .511981→.498131 | .845274→.830767 | 1.568186→1.586406 |
| 1e-4 | radial | .163956→.163639 | .328747→.328789 | .568732→.571872 |

各tol/laneで7216/7216 value convergence、status mismatch 0、node mismatch 0。
mu相対差最大1.95e-12。VBM参照に対する要求tol超過は1e-3で両方0、
1e-4で両方3（既存と同数）。この3件をviolation 0と表現しない。

candidate独立angular reference checkerは550行中466行がreference usable、
その範囲でviolation 0。残る84行は検証成立の主張に含めない。
既存D14 root scheduling/overflow単体テストをcandidate compile flag付きで
実行してpass。root/event全集合の独立parityは未完了なので研究flagを維持。
1e-4 warm p99 +1.16%も隠さず、次のtail確認に残す。

whole再現:
```sh
c++ -O3 -DNDEBUG -std=gnu++17 -fext-numeric-literals -Isrc -march=native -funroll-loops -ffp-contract=fast -fno-math-errno evidence/holonomic/adaptive_estimator_phase10_20260913/runner_hybrid.cpp -o /tmp/phase11_base -lquadmath
c++ -O3 -DNDEBUG -std=gnu++17 -fext-numeric-literals -Isrc -march=native -funroll-loops -ffp-contract=fast -fno-math-errno -DHOLO_D14_PRESEARCH_RECIPROCAL evidence/holonomic/adaptive_estimator_phase10_20260913/runner_hybrid.cpp -o /tmp/phase11_recip -lquadmath
taskset -c 0 /tmp/phase11_base evidence/holonomic/v2_adaptive_best_trajectory_20260913/input_snapshot.tsv evidence/holonomic/adaptive_extreme_phase11/whole_base.tsv 3
taskset -c 0 /tmp/phase11_recip evidence/holonomic/v2_adaptive_best_trajectory_20260913/input_snapshot.tsv evidence/holonomic/adaptive_extreme_phase11/whole_recip.tsv 3
python3 benchmarks/holonomic/summarize_adaptive_reciprocal.py
```
