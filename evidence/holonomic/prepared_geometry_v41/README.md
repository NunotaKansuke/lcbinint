# v41: 境界準備から積分への幾何状態の引き渡し

## 結果と採用範囲
要求rtol=1e-3/1e-4を維持。uniformではfold状態を最初の積分点へ渡し、uniform/linear LDのvalue-onlyでは境界準備と積分に同じ根数判定kernelを使う。外側master/defaultへ昇格しない。gradient/Jacobian経路では有効化しない。

## 情報の流れと変更
既存の流れはseed→根数probe→local fold→DD/qf端点精密化→積分点の根数確認→root-pair cold initialization→根の継続。fold位置・角度は準備されていたが、根の継続の初期状態として渡していなかった。

1. 端点精密化の最終DD jetから(m,dm/dR,dv/dR)を求め、EventLocationに保持する。mはordinary/reciprocal chartの精密化済み座標。新しいfold solve、polynomial reconstruction、求根は追加しない。非DD/利用不能なmotionは従来のcold経路を使う。
2. uniformの最初のcold積分点に、foldの座標・motionから予測したpairを渡す。previous-node predictionは同じRなので不要。既存corrector、root-count一致、inside arc、端点精度のgateで検証し、不適切なpairは既存cold fallbackへ戻る。他の点・refinement treeのanchor再利用は維持。
3. 根数確認そのものは各積分点で維持する。準備で使う既存boundary_quartic_countをPlainValueのnodeでも使い、符号が曖昧な場合は既存Sturmへ戻す。保存されたroot countを異なるRへ無検証でコピーする変更ではない。global coverage証明も追加していない。

prepared_fold_pair_seed / reuse_frontier_count_kernelはdefault false。研究frontierだけがvalue-onlyで有効化し、fold pairはu=0に限定する。HOLO_FRONTIER_DISABLE_FOLD_STATE_HANDOFF、HOLO_FRONTIER_DISABLE_SHARED_NODE_COUNTで独立に無効化可能。

## 数値検証
7216入力×2tol=14432条件、保存済みVBM RelTol=1e-6参照。新しい精度超過0、新しい未収束0。既存超過3条件・未収束6条件を維持。1e-3の収束7213条件は全て目標内。値4625条件、点数53条件が変更。点数中央値はuniform22、LD53を維持。rtol、ledger、acceptance、resource limitsは変更しない。

weighted334、adaptive1614、frontier321で計2269 checks成功。既存微分、fold/arc/anchor/count/direct-transport auditsを含む。既存classifierの独立binary128 Sturm比較は188407 checks / 0 failures、うち187906 accepted fast countsを独立に確認。計190676 checksで失敗0。

誤差推定はinterval enclosureではない。coverage_certified=false、既存精度超過・未収束を成功扱いしない。

## 最終速度比較
v35 dirty accepted source/binariesをbaselineにする。one thread、fresh seed込み、caustic構築除外、同じ684入力×2tol×5反復、順方向/逆方向。全build/tests/profile runを終了してからserial timing。低増光A<1.1のuniform248入力、LD251入力。ステージ中央値同士は足さない。

|条件|順序|積分 v35→v41 [µs]|seed込み全体 v35→v41 [µs]|入力ごとの全体比率中央値の削減|
|---|---|---:|---:|---:|
|uniform 1e-3|forward|24.224→20.594|61.547→56.027|4.25%|
|uniform 1e-3|reverse|24.620→20.576|61.059→56.063|4.22%|
|uniform 1e-4|forward|31.024→27.889|67.434→62.735|5.96%|
|uniform 1e-4|reverse|31.026→27.747|67.958→62.284|4.87%|
|LD 1e-3|forward|49.039→43.856|77.426→72.287|5.31%|
|LD 1e-3|reverse|48.712→43.714|76.918→71.477|3.60%|
|LD 1e-4|forward|56.407→50.282|82.313→77.272|5.25%|
|LD 1e-4|reverse|55.246→50.396|81.563→77.124|4.59%|

通常87組、rtol=1e-4、各組5反復中央値の平均:

|順序|v35 [ms]|v41 [ms]|
|---|---:|---:|
|forward|0.256267|0.241284|
|reverse|0.255948|0.241709|

通常行列は約5.6–5.8%短縮。low uniformはseed込み集団中央値で約61→56 µs、入力ごとの比率中央値では約4.2%削減。異なる集計を混同しない。

clock付きprofileは速度結果に混ぜず、cold/warm分岐の回数を確認するために使用。最終版のuniform root-pair cold経路中央値6→1、warm18→21。LDのcold9→9、warm41→41。cold経路はroot pairを構成し直すカウンタであり、fresh seed providerの点源solve数ではない。

## 切り分け
両profileでpair handoffを有効にした中間版について、baseline / pairのみ / count kernelのみ / 両方を独立binary、同じ入力、5反復、両順序で比較した。uniformはpairのみの全体効果が小さいが、count kernelと合わせた積分部分に追加効果がある。LDはpair handoffのcold数は減っても全体改善が確認できず、最終版ではpair handoffを無効化した。速度改善の主役はnode count kernelの共通化。

|profile|順序|baseline [µs]|pairのみ [µs]|countのみ [µs]|両方 [µs]|
|---|---|---:|---:|---:|---:|
|uniform|forward|62.244|61.990|56.480|56.096|
|uniform|reverse|62.990|61.682|57.171|55.589|
|linear|forward|76.457|78.059|70.885|71.264|
|linear|reverse|76.720|77.273|71.626|71.209|

both_profilesはLDにもpair handoffした中間full audit/timing/profile。ablation_*はその分解測定。最終timingとは混ぜない。v40は初期試作で、fold motionを積分時に再構築した費用とper-node sample構築が改善を相殺した。v41では最終jetを保持し、sampleは必要な場合だけ作る。

## 再現と来歴
checkout /rogue1_8/nunota/lcbinint/experiments/seeded-boundary-events、branch experiment/seeded-boundary-events、HEAD f3547723b79828d9c501182746028e99f7bad116、host rogue1、GCC 11.5.0。dirty sourceを含むsource snapshot、binaries/input/library hash、provenanceを保存。before_includeはrelative includeも同じbaselineへ解決する完全なheader treeで、v35に相当する3 headersを上書きしてある。*_before.hppはv40中間試作のsnapshotで、比較baselineとは区別する。

```bash
bash evidence/holonomic/prepared_geometry_v41/build.sh
bash evidence/holonomic/prepared_geometry_v41/build_normal.sh
bash evidence/holonomic/prepared_geometry_v41/build_tests.sh
bash evidence/holonomic/prepared_geometry_v41/build_count_test.sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 build-seeded-events/bench_prepared_geometry_v41 evidence/holonomic/seeded_paper_qrho_20261013/input_snapshot.tsv evidence/holonomic/prepared_geometry_v41/full_after.tsv 1
python3 evidence/holonomic/prepared_geometry_v41/audit.py
bash evidence/holonomic/prepared_geometry_v41/build_profile.sh
bash evidence/holonomic/prepared_geometry_v41/run_timing.sh
python3 evidence/holonomic/prepared_geometry_v41/summarize.py
```

比較用v35 release binariesをcurrentから再buildして上書きしない。最終profileはbefore/after各3反復、最終timingはその完了後の時計なしbinary。ablationのboth/pair binaryは両profile有効の中間版を保持したもの。最終sourceで再buildするとLD pair無効になるので、元のablationを再現する場合はboth_profiles/seeded_radial_frontier.hppの有効化設定も使う。

commit/pushなし。outer master/default・manuscript変更なし。VBMのtimingは保存済みで再実行していない。低増光uniformのVBM中央値46.85 µsに対しv41約56 µsで、差はまだ残る。
