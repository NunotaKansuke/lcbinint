# Phase40: warm seedの構造Newton前処理（不採用）

基準2045425。前epoch rootに現在のC3/G4/Z3式でD14Real Newtonを1回適用してから、既存double basin探索へ渡す案を試した。stepが最近傍旧root間隔の1/4以上、または非有限なら元seedを保持。この1/4は候補生成上のguardであって受理certificateではない。既存D14Real/qf補正・全根検査・topologyは全部通す。新solverや独立Newtonだけによる受理は追加しない。

## 結果

同じ7216-row trajectory inputでwarm stage profile。先頭cold epochを除く5412遷移を比較した。

|stage|baseline中央値 ms|candidate中央値 ms|
|---|---:|---:|
|whole（profile instrumentation込み）|.3168705|.3317855|
|topology|.126721|.148161|
|prepare|.001188|.0232515|
|double presearch|.008018|.0078845|
|D14Real|.021856|.0216045|

全5412遷移のqf累積費用も884.18→1018.64 msへ増加した。準備に約22 us余計に払い、presearch/後段補正の中央値はほぼ変わらない。根を現在の式で近づけるだけでは後段の仕事を削れなかった。

比較は同入力/同compiler flags/CPU0、candidate→baselineの逐次単回profile。baselineコンパイルがcandidate runと一部重なり、共有cache/powerの影響は排除できない。ただし追加準備費と省けた探索費の差が大きく、次の大規模whole再測定へ進める根拠がない。profileのwhole値を計測なしの通常whole runner値と直接比較しない。

全7216行でok/node数差0。最大mu差/(1+|mu|)=6.82e-13。独立14-root parityやreference/5Jac追加試験は未実施。不採用なので試作をsourceから撤去しpatch保存だけとした。production変更なし。

## 再現

warm_seed_correction.patchを基準headerへ適用し、candidate defineを指定。baselineでは候補defineだけ外す。

```
c++ -O3 -DNDEBUG -std=gnu++17 -fext-numeric-literals -Isrc -march=native -funroll-loops -ffp-contract=fast -fno-math-errno -DHOLO_D14_NATIVE_RESIDUAL_SCREEN -DHOLO_D14_DIRECT_CONVOLUTION -DHOLO_ADAPTIVE_STATIONARY_ENDPOINT -DHOLO_D14_WARM_NOISE_HANDOFF -DHOLO_D14_SCALAR_CONSTANTS -DHOLO_D14_WARM_SEED_CORRECTION benchmarks/holonomic/bench_adaptive_stage_profile.cpp -o /tmp/p40_profile -lquadmath
taskset -c 0 /tmp/p40_profile evidence/holonomic/v2_adaptive_best_trajectory_20260913/input_snapshot.tsv warm > /tmp/profile.tsv
```

raw/summary/patchはevidence/holonomic/adaptive_extreme_phase40/。seed correction、係数cast、qf旧root差からの最近傍分離計算をprepareに含む。guard計算をさらに安くできる可能性はあるが、もともとpresearchは約8 usで、今回ほぼ減っていない。追加前処理を小さく磨くより、既存後段の受理までに必要な評価を減らす方を優先する。
