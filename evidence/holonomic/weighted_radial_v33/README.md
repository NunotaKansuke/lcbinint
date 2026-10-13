# v33: 平方根の重みを使う動径積分

比較基準は採用v32、checkoutは `experiments/seeded-boundary-events`、branch `experiment/seeded-boundary-events`、HEAD f3547723b79828d9c501182746028e99f7bad116。未コミットの既存研究を保持。要求rtol=1e-3/1e-4は変更しない。

## 実装

uniform/value-onlyについて、occupied spanの両端がphysical fold、両隣がempty、既存の内部probeが全て2 crossingsという構造を持つ入力を選ぶ。初期区間が全てこの構造を持つ場合だけ新経路を選択する。追加のレンズ方程式の求根やepoch間cacheは不要。これは既存probeからの経路選択で、連続したsupportの完全性の新証明ではない。

Rを線形に[-1,1]へ写し、F(R)=sqrt(1-x²) g(x)としてGauss-Chebyshev第二種の重みで積分する。従来Fejerと同じdyadic-angle nodesなので7→15→31→…の点はそのまま再利用できる。node評価に渡すJacobianをsqrt(1-x²)で割り、値と既存のinner/geometry/roundoff ledgerを同じ規則で積分する。event端の見積もりではこの因子を戻す。

対称weightに対する奇成分の積分は厳密に0。fine/coarse双方の値を±xで平均して、補間誤差を偶成分について推定する。積分値の和の計算順序は変更しない。これは一般のembedded差で相殺を許す変更ではなく、線形座標と対称weightの恒等式を利用したもの。

最低7点は維持。weighted modeではradial errorに保守的なfull interpolation detailを使い、1/8 floorを持つ従来hybrid縮小は使わない。途中のtopology修復で普通の区間が生じた場合もこのledgerを維持する。weighted treeが存在しない場合は従来controllerを使用。

weighted parentがh-refinementを要する場合は、その区間を2つの普通のone-fold mapへ分ける。両childの値を得るまでparentを有効に保ち、partial failureで領域を失わない。旧nodeのroot stateと探査witnessは保持する。座標が変わるchildの関数値は必要な位置だけ新規評価する。support発見・seed・他の区間を従来法で一からやり直すfallbackではない。

LDはuniformと違い、端でlinear項も含む。この重みでまとめて扱うと高精度時にquotientに残る平方根を過剰に細分化するため、今回LDはv32経路を維持。微分経路も維持。外側masterのライブラリ/defaultには変更を入れない。

## 試作で見つかった問題

- 最初の混在weighted/hybrid版：代表1368条件で新しい精度超過2件。採用せず (`pilot_after.tsv`)。
- weighted区間だけfull detailにする版：1件残った (`detail_pilot.tsv`)。case82のweighted主像は正確だが、別の普通の小像の旧hybrid誤差が甘かった。主像のledgerが小さくなったことで全体が早く受理されていた。`diagnose.log`にpanel内訳。
- mixed全体のledgerをfull detailにする版：全14432条件の新しい精度超過0。ただし複雑な像の費用が増え、従来87組では全体が遅くなった (`mixed_mode`)。初期構造による経路選択でこの範囲への適用を除いた。
- 最低3点かつ偶成分だけで受理する版：全数比較で5件の新しい超過。高次の偶成分を少数nodeが見逃すため撤回 (`even_initial3`)。奇成分の相殺という恒等式自体は正しいが、それで有限nodeの完全性を主張しない。
- `initial3_full_detail` は奇成分もfull normに含めた中間版、`conservative_all` は最初の全区間full ledger版。中間計測を最終速度に混ぜない。

## 最終計測と判断

低増光A<1.1、uniform248入力、linear LD251入力。各入力5反復中央値の集団中央値。集団中央値はステージごとに異なる入力が中央になるため、ステージ中央値同士を足して全体中央値と比較しない。入力ごとの比率の中央値も併記する。

|条件|順序|積分 before→after [µs]|seed込み全体 before→after [µs]|点数中央値 before→after|
|---|---|---:|---:|---:|
|uniform 1e-3|forward|36.114→24.973|64.540→60.131|37→22|
|uniform 1e-3|reverse|35.125→25.009|62.996→61.876|37→22|
|uniform 1e-4|forward|46.867→31.161|74.181→68.891|61→38|
|uniform 1e-4|reverse|45.565→31.391|72.925→68.477|61→38|
|LD 1e-3|forward|50.699→48.797|80.030→76.627|53→53|
|LD 1e-3|reverse|49.566→48.741|78.545→76.968|53→53|

入力ごとの比率の中央値：uniform 1e-3の積分21–23%、全体6–7%削減。uniform 1e-4は積分35–37%、全体9–11%削減。集団中央値の積分比ではそれぞれ約29–31%、約31–34%削減。LDは値・点数とも変更0なので小さなtiming差を改善として主張しない。

保存済み同一入力・tol VBMとの低増光比較（最終逆順の5反復中央値を使用）：

|条件|VBM中央値 [µs]|VBMより速い入力 before→after|中央値 VBM/after 比|
|---|---:|---:|---:|
|uniform 1e-3|46.850|47→92 / 248|0.739|
|uniform 1e-4|80.817|133→139 / 248|1.144|
|LD 1e-3|50.209|55→57 / 251|0.730|

VBMは今回再実行していない。uniform 1e-3で勝つケースは47→92に増えたが、全局面で勝っていない。

従来87組のrtol=1e-4、各組中央値の平均：

|順序|全体 before→after [ms]|
|---|---:|
|forward|0.258586→0.262505|
|reverse|0.257484→0.260716|

この行列ではなお約1–2%遅い。低増光uniformの改善と汎用行列の小さなregressionを区別する。新経路は研究branchの候補として残すが、外側master/defaultへの昇格や全体で一律の速度改善は主張しない。今後のLD接続にはuniform平方根項とLD linear項を別の重みで扱う設計が必要。

最終全7216入力×2tol=14432条件：新しい精度超過0、新しい未収束0。既存の未収束6条件、rtol=1e-4の既存超過3条件は残る。1e-3で収束した7213条件は全て目標内。LDの値と点数は全て不変。従来87組でもresolved D14 referenceに対するcandidate failure0、D14 reference未解決14条件は精度成功の根拠に使わない。

## 検証

新規weighted testsは328 checks、既存adaptiveは1614 checks、frontierは321 checks、全て0 failures。weighted polynomial moments (degree 2N-1まで)、厳密なnode再利用、Beta関数の独立積分oracle、奇成分を足した場合の同一積分と最小7点保持、ordinary fold childへの分割と独立Simpson oracle・領域保存を確認。既存微分、resource budget、primal snapshotとfold/arc/anchor/count/direct-transport auditsも実行。

積分誤差推定そのものはinterval enclosureではない。奇成分の積分は厳密な恒等式だが、有限nodeからの補間誤差の推定は従来同様に数値的な推定。`coverage_certified=false` を維持。全数精度比較でも既存の未解決条件を成功に数えない。

## 再現

build/testを全て完了してからserial timingを行う。1 thread、fresh seed構築を含め、caustic構築は除外。684入力×2tol×5反復、順方向/逆方向。従来87組は各組の5反復中央値の平均。VBMは同一入力・tolの保存済みtimingで、今回再実行していない。

```bash
bash evidence/holonomic/weighted_radial_v33/build.sh
bash evidence/holonomic/weighted_radial_v33/build_normal_before.sh
bash benchmarks/holonomic/build_seeded_paper_qrho.sh
bash benchmarks/holonomic/build_seeded_radial_master.sh
bash evidence/holonomic/weighted_radial_v33/build_tests.sh
bash evidence/holonomic/weighted_radial_v33/run_accepted.sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 build-seeded-events/bench_weighted_radial_after_v33 \
 evidence/holonomic/seeded_paper_qrho_20261013/input_snapshot.tsv \
 evidence/holonomic/weighted_radial_v33/full_after.tsv 1
python3 evidence/holonomic/weighted_radial_v33/summarize.py
```

`before_include` は採用v32の変更前headers。`source`、SHA256SUMS、provenance.jsonは最終採用sourceとbinary/input/library identity。`HOLO_FRONTIER_DISABLE_WEIGHTED_RADIAL` で研究frontierの新経路を無効化可能。commit/pushはしていない。
