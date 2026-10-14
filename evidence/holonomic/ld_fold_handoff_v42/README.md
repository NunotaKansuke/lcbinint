# v42: LDの後続点へfold状態を再利用

## 確認した原因
初期仮説は「LDの最初の点がfoldに近く、tiny-width gateからcoldへ戻る」だった。直接ログでは最初の受け渡しはLD1851/2091成功(約89%)、tiny-widthを伴うreject52件(約2.5%)。初期点の失敗だけでは改善が相殺された理由を説明できなかった。

初期7点の最初の位置は、linear LDの通常double-fold mapで端から区間幅の約0.00357、uniformのweighted linear mapで約0.03806。LDは次の点への移動が相対的に大きく、前の細いpairの線形予測が既存kTransportVJumpRel=4のgateを超える。最初の初期値は受け取れても、その後に根の組を作り直すcold経路へ戻っていた。

診断対象684入力×2tolでLD後続呼び出し50543、cold5948、そのうちpredictor rejectを伴う3735(約63%)。カウンタは原因が重なるため各reject列を足してcold総数にしない。rootpair_branch_rejectはpredictor/corrector失敗も含む最終fallbackカウンタ。

## 最終変更
- 積分spanごとに左右の精密化済みfold状態への参照を一度だけ用意。積分中のevent cacheは不変。参照はそのintegration callに限定し、epochを跨ぐcacheではない。
- LDでも最初のcold nodeにfoldからのpair初期値を渡す。
- LDの後続点で、radial expansionの大きいstepだけ既存predictorを確認し、同じrelative-width gateを超える場合に準備済みfoldから新しい初期値を渡す。root count、corrector、inside-arc、端点精度のgateは従来と同じ。
- 前のfull quartic stateも保持して、foldからのpairが受理されなければ既存fallbackに利用する。準備・求根を全てやり直すrestartは追加しない。
- uniformの後続点は既存continuationを維持。gradient/Jacobian/defaultやouter masterへの有効化はしない。

kTransportVFloor、kTransportVJumpRel、Newton回数・許容差、要求rtol、radial/inner/error ledger、全体acceptanceを変更していない。radial step条件は予測を調べる計算の選別で、数値受理条件ではない。

## 精度と回帰
全7216入力×2tol=14432条件、baselineは採用v41。新しい精度超過0、新しい未収束0。既存超過3条件と未収束6条件は維持。値4954条件が変わり、点数は全条件不変。1e-3収束7213条件は全て目標内。参照は保存済みVBM RelTol=1e-6、VBMは今回再実行していない。

weighted334/adaptive1614/frontier321、計2269 checksで失敗0。fold/arc/anchor/count/direct-transportと既存微分検証を含む。共有count classifier自体はv41から変更していない。coverage_certified=falseを維持。積分誤差推定や全source coverageの新しい証明は主張しない。

## cold経路
clock付きprofileは速度比較に使用せず、分岐回数のみ確認。低増光A<1.1・rtol=1e-3、LD251入力。v41のLD cold中央値9/warm41に対し、v42はcold1/warm53。uniformはcold1/warm21を維持。coldはroot pairの再構成経路の回数であり、fresh seed providerの点源solve数ではない。

## 時計なしの最終速度
one thread、fresh seed込み、caustic構築除外、同一684入力×2tol×5反復、順方向/逆方向。全build/tests/profile後にserialで測定。低増光uniform248、LD251入力、各入力5反復中央値を集計。ステージ中央値同士は加算しない。

|条件|順序|積分 v41→v42 [µs]|seed込み全体 v41→v42 [µs]|入力ごとの全体比率中央値による削減|
|---|---|---:|---:|---:|
|uniform 1e-3|forward|21.038→20.668|55.029→56.415|-0.40%|
|uniform 1e-3|reverse|21.087→20.242|56.368→56.191|2.32%|
|LD 1e-3|forward|44.044→42.430|71.668→69.467|1.71%|
|LD 1e-3|reverse|44.414→41.977|71.988→68.867|3.92%|
|LD 1e-4|forward|50.285→48.162|76.798→74.188|2.16%|
|LD 1e-4|reverse|50.504→47.240|77.544→73.156|3.08%|

LD1e-3で全体中央値約72→69 µs、1e-4で約77→73–74 µs。入力ごとの比率中央値では1e-3が1.7–3.9%、1e-4が2.2–3.1%削減。uniformの追加速度改善は主張しない。

## 通常87組の変動と限界
rtol=1e-4、各組5反復中央値の平均:

|順序|v41 [ms]|v42 [ms]|
|---|---:|---:|
|forward|0.241873|0.239574|
|reverse|0.243131|0.250149|

逆順では約2.9%の増加が出た。seedだけの平均でも+3.44 µs、planning+1.60 µsと、今回の変更が直接変えない処理にも差がある。個別case12/13のseedが約80–98 µs増えたため、負荷/計測変動と算法の費用を分ける追加確認を実施した。これを理由に遅い行を除外しない。

同じ87組、after→before、7反復の追加測定はv41 0.236572 ms、v42 0.237442 ms (約0.37%増)。通常87組で一貫した速度改善は確認できていない。追加測定はnormal_check_*、最初の両順序結果はnormal_{forward,reverse}_*に保存。改善の主張は低増光LDに限定する。

## 来歴・再現
checkout /rogue1_8/nunota/lcbinint/experiments/seeded-boundary-events、branch experiment/seeded-boundary-events、HEAD f3547723b79828d9c501182746028e99f7bad116、host rogue1、GCC11.5.0。dirty研究を継承。source/before snapshots、commands、binaries/input/library hashesを保存。commit/pushなし。外側master/default、manuscript変更なし。

```bash
bash evidence/holonomic/ld_fold_handoff_v42/build.sh
bash evidence/holonomic/ld_fold_handoff_v42/build_normal.sh
bash evidence/holonomic/ld_fold_handoff_v42/build_tests.sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 build-seeded-events/bench_ld_handoff_v42 evidence/holonomic/seeded_paper_qrho_20261013/input_snapshot.tsv evidence/holonomic/ld_fold_handoff_v42/full_after.tsv 1
python3 evidence/holonomic/ld_fold_handoff_v42/audit.py
bash evidence/holonomic/ld_fold_handoff_v42/build_profile.sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 build-seeded-events/bench_ld_handoff_profile_v42 evidence/holonomic/low_magnification_fixed_cost_v30/input.tsv evidence/holonomic/ld_fold_handoff_v42/reseed_profile.tsv 1 > evidence/holonomic/ld_fold_handoff_v42/reseed_profile.log 2>&1
bash evidence/holonomic/ld_fold_handoff_v42/run_timing.sh
python3 evidence/holonomic/ld_fold_handoff_v42/summarize.py
```

比較v41 binaryをcurrent sourceから再buildして上書きしない。first_onlyはLDにも最初のfold handoffだけを有効にした診断版のsource。profile.log/profile.tsvはその初期hand-offログ、transport_profile.*は後続点のfallback診断、reseed_profile.*が最終版。HANDOFF/TRANSPORT logはHOLO_PREPARED_HANDOFF_PROFILE時のみ出力し、release timingには含めない。

VBMの同一tol timingは保存済み参照で今回は再実行していない。LD1e-3はなお約69 µsで、保存済みVBM中央値約50 µsとの差が残る。
