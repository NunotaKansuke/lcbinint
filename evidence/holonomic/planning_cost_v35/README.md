# v35: 固定費の内訳と端点精密化の最終jet再利用

## 調査
対象は低増光A<1.1、uniform248入力、linear251入力。既存684入力の行列を使用。要求rtol=1e-3/1e-4を維持。

低増光では根数probe中央値88回。march内の各外縁12回の二分探索、局所fold Newton、積分setupのDD端点精密化、および観測した根数変化の確認が準備費用を構成する。通常の未計測binaryと、clockを追加したprofile binaryを分けて使った。

以下はprofile binaryのrtol=1e-3、3反復、uniform群の集団中央値。時計を多数追加したため絶対値を未計測binaryの速度と混ぜない。countはmarch等の内側、eventはendpoint/repairの内側なので全列を足さない。setup_eventはintegrationの内側。

|profile範囲|before [µs]|after [µs]|
|---|---:|---:|
|根数確認|6.451|6.339|
|radial march|7.212|7.260|
|外縁fold位置解決|5.208|5.218|
|観測した変化の確認・修復|6.219|5.630|
|積分setupの端点精密化|5.791|5.080|

集団中央値同士を引いて固定費を推定しない。前の約62 µs - 約25 µsという差は同じ中央値入力の内訳ではない。処理範囲の関係と入力ごとの値で評価する。

## 採用変更
DD coupled_fold_newtonは停止後に最終P、Pt、PR、Ptt、Ptrを既に計算し、residual gateに使っていた。その直後のtopology_event_estimateが同じ(r,t)のjetを再計算してradius residual/derivative gateを評価していた。

optional final_outを追加し、ordinary/reciprocal chartのDD topology-event callerがそのjetを再利用する。両方の受理判定、反復数、精度、uncertainty計算はそのまま。その他callerは既定のnullで同じ経路を使用する。追加profile timerはHOLO_FRONTIER_COST_PROFILE時だけ有効で、通常binaryに時計やnull branchを加えない。

## 数値検証
全14432条件でv34とmu、estimated_abs_error、nodes、value_converged、coverage_certified、stopが文字列レベルで完全一致。既存の精度超過3・未収束6はそのまま。weighted334/adaptive1614/frontier321、計2269 checksで失敗0。既存の微分・fold・arc・anchor・count・direct-transport auditsを含む。tolや数値gateの緩和なし。coverage_certified=falseを維持。独立精度参照は既存保存済みVBM RelTol=1e-6、今回はVBMを再実行していない。

## 時計なしの速度比較
v34/v35の同一入力・同一tol、fresh seed込み、caustic構築除外、one thread、5反復中央値、順方向/逆方向。compile/tests/profile runを完了してからserial timing。ステージ中央値同士は加算しない。

|条件|順序|seed込み全体 v34→v35 [µs]|入力ごとの比率の中央値による削減|
|---|---|---:|---:|
|uniform 1e-3|forward|62.264→61.099|2.50%|
|uniform 1e-3|reverse|63.048→61.185|2.02%|
|uniform 1e-4|forward|69.432→68.535|2.28%|
|uniform 1e-4|reverse|69.062→67.927|1.67%|
|LD 1e-3|forward|79.229→75.560|2.71%|
|LD 1e-3|reverse|78.092→76.421|1.51%|
|LD 1e-4|forward|84.090→80.404|2.85%|
|LD 1e-4|reverse|83.625→80.649|1.50%|

通常87組・rtol=1e-4、各組5反復中央値の平均:

|順序|v34 [ms]|v35 [ms]|
|---|---:|---:|
|forward|0.255128|0.253108|
|reverse|0.255105|0.253212|

これは小さい改善。全体timeの差が全て削った一式の費用とは限らない。profileでもsetup_eventの約0.7 µs減少が見えるが、clockの影響とrun間変動を含む。VBM timingは保存済みで再実行していない。低増光uniformはなお約61 µsで、保存済みVBM中央値46.85 µsには届かない。

## 再現・provenance
作業checkout /rogue1_8/nunota/lcbinint/experiments/seeded-boundary-events、branch experiment/seeded-boundary-events、base f3547723b79828d9c501182746028e99f7bad116、host rogue1。研究dirty sourceのv34を継承。outer master/defaultに変更なし。commit/pushなし。

```bash
bash evidence/holonomic/planning_cost_v35/build.sh
bash evidence/holonomic/planning_cost_v35/build_normal.sh
bash evidence/holonomic/planning_cost_v35/build_tests.sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 build-seeded-events/bench_planning_after_v35 evidence/holonomic/seeded_paper_qrho_20261013/input_snapshot.tsv evidence/holonomic/planning_cost_v35/full_after.tsv 1
python3 evidence/holonomic/planning_cost_v35/audit.py
bash evidence/holonomic/planning_cost_v35/build_profile.sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 build-seeded-events/bench_planning_profile_v35 evidence/holonomic/low_magnification_fixed_cost_v30/input.tsv evidence/holonomic/planning_cost_v35/profile_detail_after.tsv 3 > evidence/holonomic/planning_cost_v35/profile_detail_after.log 2>&1
bash evidence/holonomic/planning_cost_v35/run_timing.sh
python3 evidence/holonomic/planning_cost_v35/summarize.py
```

baseline用v34 binaryは保存済みを使用。profile_detailは最終jet再利用前、profile_detail_afterは再利用後。profile.cppは既存planning runnerのコピーに内訳出力だけを追加したもの。profile_beforeは初回の簡易profileで、profile_detailとは時計の範囲が異なる。

## 残り
marchの根数確認回数やrepair確認には費用が残るが、固定回数を単純に減らす変更は採用しなかった。境界の選択・coverage witnessを保持できる設計が必要。今回の成果は最終jetの二重計算を除いたものに限る。LDの動径積分則、既存精度超過/未収束、coverage未認証は未解決。
