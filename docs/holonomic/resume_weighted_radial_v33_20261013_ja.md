# v33 引き継ぎ確認 (2026-10-13 client date)

## 目的と制約
要求精度を変えず、動径積分を削減してseed込みでVBMとの差を縮める。直前の未完了作業は通常87組の1–2%遅延の原因特定。既存dirty研究を保持し、外側master/defaultには昇格しない。LDと微分はv32経路を維持。coverage_certified=false。commit/pushの指示はない。

## 確認した実体
host rogue1。外側checkout /rogue1_8/nunota/lcbinint は feature/boundary-error-and-discriminant-support / a68bca9d0e8234ef499dee3d860d23afef6061d1。
対象は既存の内側worktree /rogue1_8/nunota/lcbinint/experiments/seeded-boundary-events、experiment/seeded-boundary-events / f3547723b79828d9c501182746028e99f7bad116。dev/holonomic は別の既存worktreeに登録されているが、本件の最新成果は内側checkoutにある。
v33 source控え4ファイルは現在の実装とbyte一致。SHA256SUMSの127件は全て一致。既存の保存済み全14432条件比較では新しい精度超過0、新しい未収束0、既存超過3と未収束6は残る。全数計算は今回再実行していない。

## 今回行った確認
既存のchecksum確認済みbinaryで weighted_radial 328、adaptive_radial 1614、seeded_radial_frontier 321 checksを再実行、全て0 failures。
保存済み通常87組の入力別・stage別分解を evidence/holonomic/resume_v33/normal_decomposition.json に保存。
通常87組の全体平均差はforward +3.919、reverse +3.232 µs。
case44/u=0/rtol=1e-4だけで +760.367 / +737.494 µs、点数571→1135。
この1入力を除いた86組の平均差は -4.877 / -5.306 µs。これは原因の寄与を分けた診断であり、case44を性能評価から除外する採用方針ではない。
case44を既存のbefore/after binariesで5反復再実行。uniform 1e-4は571→1135、1e-3も285→539を再現。LDの点数は不変。全4条件でresolved D14に対するcandidate failure0、reference unresolved0、local_repairs0。D14との一致は独立真値による検証ではない。新計測の絶対時間は元計測と異なるので混ぜない。

## 続ける作業
case44のweighted panel毎に誤差ledger、level、split、端のquotient形状を計測し、full interpolation detailによる過剰refinementと端の非滑らかさを切り分ける。local topology repairが原因という仮説は今回の再現では支持されない。
case IDや測定時間で経路選択しない。tolを緩めない。修正案は構造的な判定または積分則/誤差推定の数学に基づかせる。改善案が得られた場合は既存全14432条件と両順序timingで再確認する。
今回solver sourceは変更していない。新しい長時間jobは起動していない。commit/pushなし。

## 再現コマンド (対象checkoutから)
```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 build-seeded-events/test_weighted_radial_v33
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 build-seeded-events/test_adaptive_radial_v33
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 build-seeded-events/test_seeded_radial_frontier_v33
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 build-seeded-events/bench_normal_before_v33 evidence/holonomic/resume_v33/case44.tsv evidence/holonomic/resume_v33/case44_before.tsv 5 0
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 build-seeded-events/bench_seeded_radial_master evidence/holonomic/resume_v33/case44.tsv evidence/holonomic/resume_v33/case44_after.tsv 5 0
```
