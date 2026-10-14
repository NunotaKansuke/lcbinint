# Phase29: endpoint exact-argument cache（不採用）

基準1b1bb1c。adaptive endpoint Newtonで直近2つの角度とphi/dphiを保存し、角度とzero signが完全一致した場合だけ再利用。停止条件/更新回数を変えない。prototype.patchへ保存し、コードは戻した。

7216行warm stage profile、steady5412行。145571 cache hits、3793566実評価。mu完全一致、status/node差0。endpoint p50 23.2295→22.9875 µs、profile whole .3079395→.3079855 ms。全体の利益を確認できず不採用。全体の同方向改善がないため追加の独立reference/full benchmarkは行っていない。

再現: Phase28 profileのcompiler flagsにHOLO_ADAPTIVE_ENDPOINT_CACHEを追加。入力v2_adaptive_best_trajectory_20260913/input_snapshot.tsv、warm、taskset -c 0。candidate→baselineの順で実行。rawとsummary.json保存。profile timerを含むため最終whole性能とは区別する。

より大きい残存arc/root tracking費用はsteady p50約77 µs。次はこの中のquartic solve/continuation/certificateの内訳を調べる。
