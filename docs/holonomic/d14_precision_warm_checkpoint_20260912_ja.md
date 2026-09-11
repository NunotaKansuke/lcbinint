# D14専用精度カーネル / trajectory warm-start Phase checkpoint

実測日: 2026-09-12

仕様は [d14_precision_and_warm_design_20260911_ja.md](d14_precision_and_warm_design_20260911_ja.md)。作業対象は isolated holonomic 範囲だけとし、V2/V3 の production router、fixed-`n_r` API、PF6/GM 経路は変更していない。

## 判定

`D14Real` block kernel と mixed-precision root interaction は採用候補として残した。既存の qf residual、conjugacy、Vieta、completeness の検査と fail-closed fallback は維持している。

論文用 trajectory corpus の value-only 実測では、同一入力・同一 adaptive tolerance・同一 `reps=3` で、全 14,432 行が cold / warm / radial-only のすべてで `value_converged=1`, `status=OK` だった。全条件を混ぜた中央値は、従来 V2 に対して full-cold が 0.7951 ms から 0.6441 ms、full-warm が 0.6682 ms から 0.5624 ms になった。radial-only は 0.1618 ms から 0.1651 ms で、D14を含まない経路の差は測定ノイズ範囲で改善していない。

これは D14/topology 固定費を含む full epoch の結果であり、radial-only の速さを full epoch の速さとして扱っていない。

## 実装した範囲

Step A では `V2Profile` と `bench_v2_profile` に D14Real、mixed/dangerous pair、fold seed、warm screen、各 tier の計数・時間を追加した。採用速度の判断は profile の細粒度 clock ではなく、別に計測した unprofiled whole wall を使った。

Step B では [d14_real.hpp](../../src/lcbinint/magnification/holonomic/d14_real.hpp) に D14専用の twofold 型を追加した。[d14_structure.hpp](../../src/lcbinint/magnification/holonomic/d14_structure.hpp) では `C3/G4/Z3` からの D14 展開を固定 `std::array` 中間領域へ移し、D/D' の block evaluator と fixed-capacity root storage を追加した。係数は現在の `PrimaryFrame` を同じ入力モデルとして生成しており、物理座標の定義を変えていない。

Step C では既存 trajectory の qf root seed を double basin presearchへ渡し、その結果を D14Real correctorへ渡す warm pathを維持・接続した。直接の rootwise warm Newton を独立 A/B したが、対象 corpus で受理率と全体時間が安定せず、デフォルトには採用していない。`HOLO_D14_DIRECT_WARM=1` は研究用の opt-in のままである。

Step D では Aberth の `S_i` について、十分離れた pairは Kahan compensated double、危険な近接 pairは D14Realで再計算する。in-place update順は変えていないため、Jacobi更新への変更や逆数共有は入れていない。

Step E は qf を通常経路から完全に除去する段階までは進めていない。D14 blockからの係数生成と D14Real correctorを通常経路に入れた一方、最終 qf residual / root-set certificate と必要時の qf warm/cold fallback は残した。包含半径を含む独立した全根 enclosureが未実装なので、qf certificateを弱める変更はしていない。

Step F では [radial_events.hpp](../../src/lcbinint/magnification/holonomic/radial_events.hpp) に quartic degree-one subresultantからの fold stationary seed `t=-V/U` を追加した。`U` の小ささ、非有限、残差不成立などでは従来の derivative-cubic probeへ戻る。これは seedであって認証ではなく、既存の local classification / residual checksを通す。Sturm topology certificate、chart parity、physical/complex event処理は維持した。

matrix/QZ、lifted Newtonの production 化、全体 qf 化、L1 cell-plan verbatim reuse は実装・採用していない。

## D14専用 kernel A/B

`gm_coverage_cases.tsv` の108ケースに7 reference caseを足した115ケース、各ケース best-of-5、同一 build で測定した。legacy は `HOLO_D14_STRUCT_LEGACY=1`、block は変数未設定の既定経路である。

| 経路 | solve median ms | p90 | p99 | max | whole `radial_events` median ms | whole value+5Jac median ms |
|---|---:|---:|---:|---:|---:|---:|
| legacy expansion / DD | 0.4942 | 0.5473 | 2.5671 | 2.9102 | 0.5752 | 1.0690 |
| block + D14Real | 0.2734 | 0.4964 | 0.7005 | 0.7136 | 0.2868 | 0.7578 |

block / legacy の速度比は solve median 1.80x、p90 1.09x、p99 3.68xだった。blockと expansion の係数ベクトル最大相対差は `1.657e-29`、root-set最大相対差は `2.148e-9`、parity failure は `0/115`、qf residual は legacy `5.625e-14`、block `6.613e-14`だった。

profileでは cold 108ケースの D14Real callが108、mixed/dangerous pairが `155740/24336`、nonconverged 0、fold seedが `674` 回中 `652` 成功・`22` fallbackだった。warm steady 324ケースでは D14Real callが324、mixed/dangerous pairが `395684/51844`、nonconverged 2、fold seedが `2036` 回中 `2034` 成功・`2` fallbackだった。nonconverged flagが立ったケースも、最終 qf residual gateと既存 root-set validationを通す構造で、未認証 rootを返していない。

短い `bench_v2_profile` の synthetic warm lane は legacy / block の双方で `316/324` status OK（8件は従来からの同じ難例）だったため、warmのcoverage判定には14,432行の trajectory runを使った。

## full-cold / full-warm / radial-only

入力は既存の論文用 q–rho trajectory snapshot（1,804 trajectory × 4 epoch）で、uniform / linear、`RelTol=1e-3, 1e-4` を含む 14,432 行。`LensParams{time, y, rho, 1/q, s, true}`、adaptive `n_r`、`mu_atol=1e-16` を使った。VBMの速度比較側と `RelTol=1e-6` referenceは既存 artifactを再利用し、今回追加計算はD14変更後のV2だけである。

表の値は all epoch の `p50 / p95 / p99 / max` ms。`full-cold` は各 epochで D14/topologyを新規生成、`full-warm` は trajectory内の前 epoch D14 rootsを L2 warm seedとして使い、L1 plan reuseは無効、`radial-only` は topology/D14を timer外で生成して adaptive bodyだけを測った。

| profile | RelTol | lane | p50 | p95 | p99 | max |
|---|---:|---|---:|---:|---:|---:|
| uniform | 1e-3 | full-cold | 0.5905 | 12.7826 | 17.4563 | 29.7053 |
| uniform | 1e-3 | full-warm | 0.4978 | 1.1151 | 13.3955 | 29.4415 |
| uniform | 1e-3 | radial-only | 0.1290 | 0.2772 | 0.3947 | 0.7134 |
| uniform | 1e-4 | full-cold | 0.6353 | 12.8042 | 17.7435 | 29.8042 |
| uniform | 1e-4 | full-warm | 0.5641 | 1.3012 | 13.5867 | 29.4174 |
| uniform | 1e-4 | radial-only | 0.1659 | 0.3959 | 0.6734 | 2.2821 |
| linear | 1e-3 | full-cold | 0.6463 | 12.8559 | 17.4763 | 29.9054 |
| linear | 1e-3 | full-warm | 0.5647 | 1.1999 | 13.5365 | 29.9268 |
| linear | 1e-3 | radial-only | 0.1735 | 0.3979 | 0.5290 | 0.8737 |
| linear | 1e-4 | full-cold | 0.7013 | 12.9652 | 17.8871 | 30.5629 |
| linear | 1e-4 | full-warm | 0.6342 | 1.4699 | 13.5469 | 30.5712 |
| linear | 1e-4 | radial-only | 0.2154 | 0.5686 | 0.8921 | 1.8911 |

steady epoch（trajectory初回を除外）の全条件混合集計は、full-cold `0.6408 / 12.8750 / 17.7608 / 30.5629`、full-warm `0.5326 / 0.8963 / 1.0763 / 15.7718`、radial-only `0.1623 / 0.3375 / 0.4308 / 2.1909` msだった。従来 trajectory artifactとの比較では、steady full-warm p50は `0.6289 -> 0.5326` ms、p95は `1.3556 -> 1.0763` ms、p99は `1.6338 -> 1.4480` msになった。full-coldの難しい p95/p99 は D14/topology の一部ケースが支配し、中央値ほどは動いていない。

warm reuseの合計は L1 `0`、L2 `10824`、L3 `3608`、rescreen fail `0`、warm seed used `10824` で、各 trajectoryの最初だけ coldになる期待値と一致した。

## 精度・status・timing sanity

cold / warm / radial-only の各14,432行で `value_converged=14432/14432`、`stop=Converged`、`status=OK` だった。入力に保存した既存 VBM `RelTol=1e-6` referenceとの差は全 laneで同一で、全条件混合の `p50 / p95 / p99 / max` は `4.222e-7 / 6.659e-6 / 1.413e-5 / 2.707e-4`。D14変更による新しい reference violationやstatus dropは観測していない。

stage timingは全14,432行で、全 laneについて nonfinite 0、負値 0、`whole < topology` 0だった。`whole - (topology + setup + physical + estimator + scheduler)` の p50 / p95 / p99 / max は、cold `0.0241 / 0.0526 / 0.0803 / 0.5772` ms、warm `0.0238 / 0.0516 / 0.0794 / 0.7143` ms、radial `0.0228 / 0.0495 / 0.0762 / 0.5210` msで、未分類分は有限の timer overheadとして残る。今回の D14比較で `setup_ms` を whole integrationまで含めて計上する旧バグは変更対象ではなく、既存 adaptive benchmarkの出力境界をそのまま使った。

## 図

既存の論文用 2×3 レイアウト、軸、速度比の境界値・色、誤差カラーバーを変更せず再生成した。上段は `R=t_VBM/t_V2`、下段は VBM `RelTol=1e-6` referenceとの差の cell p95。以下は代表として uniform full-warm、`1e-3` と `1e-4` を示す。全8図は trajectory evidence の `figures/` に保存した。

- [uniform / 1e-3 / full-warm PNG](../../evidence/holonomic/d14_precision_warm_20260912/trajectory/figures/q_rho_uniform_1e-3_v2_adaptive_full-warm_vbm1e-6.png)
- [uniform / 1e-4 / full-warm PNG](../../evidence/holonomic/d14_precision_warm_20260912/trajectory/figures/q_rho_uniform_1e-4_v2_adaptive_full-warm_vbm1e-6.png)
- [全図の一覧](../../evidence/holonomic/d14_precision_warm_20260912/trajectory/figures.json)

図の入力は standard script [make_figures.py](../../evidence/holonomic/d14_precision_warm_20260912/trajectory/make_figures.py) で固定し、図だけの都合でケースや精度を除外していない。

## 未実装・不採用

- qf係数生成と qf global certificate の完全除去。独立した outward enclosure がまだないため保留。
- D14Real hi/loを `PreparedEpochGeometry` の trajectory stateとして完全保持し、qf castをwarm pathから無くす変更。
- direct warm rootwise Newton、matrix/QZ seed、lifted Newton、cluster-local座標、active mask、Jacobi pair-sharing。いずれも今回の安全な既定経路には採用していない。
- D14 topology/cell-planの epoch間 verbatim reuse。現在の比較は L2 D14 warm seedのみで、各epochの topology authorityは再実行している。

## 再現手順と evidence

現在の D14 candidateを既定環境（`HOLO_D14_REAL` 未設定、既定 tol `1e-14`、`HOLO_D14_DIRECT_WARM` 未設定）で使う。legacy比較は `HOLO_D14_STRUCT_LEGACY=1`。

```text
cmake --build build-holonomic-m7 -j4
ctest --test-dir build-holonomic-m7 --output-on-failure -j4
taskset -c 0-7 ./build-holonomic-m7/bench_d14_structure evidence/holonomic/gm_coverage_cases.tsv 5
HOLO_D14_STRUCT_LEGACY=1 taskset -c 0-7 ./build-holonomic-m7/bench_d14_structure evidence/holonomic/gm_coverage_cases.tsv 5
/usr/bin/c++ -O3 -DNDEBUG -std=gnu++17 -I/rogue1_8/nunota/lcbinint/src -march=native -funroll-loops -ffp-contract=fast -fno-math-errno evidence/holonomic/d14_precision_warm_20260912/trajectory/runner.cpp -o /tmp/d14_precision_warm_trajectory_runner -lquadmath
taskset -c 0-7 /tmp/d14_precision_warm_trajectory_runner evidence/holonomic/d14_precision_warm_20260912/trajectory/input_snapshot.tsv evidence/holonomic/d14_precision_warm_20260912/trajectory/results.tsv 3
python3 evidence/holonomic/d14_precision_warm_20260912/trajectory/summarize.py
MPLCONFIGDIR=/tmp/mpl-d14-precision-warm-20260912 python3 evidence/holonomic/d14_precision_warm_20260912/trajectory/make_figures.py
```

raw全行は [trajectory/results.tsv](../../evidence/holonomic/d14_precision_warm_20260912/trajectory/results.tsv)、機械可読な trajectory summary は [trajectory/summary.json](../../evidence/holonomic/d14_precision_warm_20260912/trajectory/summary.json)、既存 V2 との比較は [trajectory/comparison.json](../../evidence/holonomic/d14_precision_warm_20260912/trajectory/comparison.json)、D14 A/Bと profile raw は [micro](../../evidence/holonomic/d14_precision_warm_20260912/micro/) にある。全体の index は [evidence summary](../../evidence/holonomic/d14_precision_warm_20260912/summary.json) とした。
