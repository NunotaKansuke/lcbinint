# D14 root-work scheduling — Phase 2 (2026-09-12)

## 結論

D14Real polish にroot単位のactive/frozen schedulingを追加し、研究用環境変数でA/Bした。全rootをAberth interactionに残したまま更新だけを止め、近接clusterはまとめて再活性化する。最終のqf residual/completeness判定は変更していない。機能はデフォルトOFFで、production routerにも昇格していない。

全14,432行のadaptive trajectory比較ではstatus・value収束判定を維持したが、whole-epochのp50はcold/warmとも0.4〜0.7%遅く、明確な速度勝ちは出なかった。D14Realのroot update数は16.4%減った一方、schedulingの制御費用とD14以外の時間を相殺できていない。したがって現状の設定は研究候補として保持し、production defaultには採用しない。

case 92 / d_bin 0 の約28 ms tailは別原因だった。double presearchが14個の有限seedを返せず、D14Real/DDを通らずにqf cold Aberthへ進み、198 sweepのうち約27.7 msをqf polishで使う。root schedulingではこのtailを短縮できない。

## 実装と安全条件

`aberth_d14_real_mixed()` に次を追加した。

- Newton correctionの絶対値と最近傍root separationに対する相対値を使うfreeze条件とpatience。
- frozen rootも全Aberth interaction sumに残す。deflationやrootの除外はしない。
- 距離が数値的cluster条件を満たすrootは一つのcomponentとして扱い、active memberの動きが閾値を超えたときfrozen memberを再活性化する。
- per-root updates、freeze/reactivation、cluster、最終Newton correction、separation、用途別position-error estimateをprofile capture時に記録する。通常のscheduled solveでは診断専用の最終root再評価を行わない。
- 最終qf residual、root count、conjugacy、Vieta/completenessの既存gateは変更していない。

有効化は `HOLO_D14_REAL_ACTIVE=1` のときだけ。測定した設定は次の通り。

```text
HOLO_D14_REAL_ABS_TOL=1e-10
HOLO_D14_REAL_REL_TOL=1e-8
HOLO_D14_REAL_PATIENCE=2
HOLO_D14_REAL_CLUSTER_REL=64*sqrt(binary64 epsilon)  # default
HOLO_D14_REAL_REACTIVATE_RATIO=0.25                  # default
```

既存のsafe presearch設定 `HOLO_D14_LOCAL_PAIRS=1`, `HOLO_D14_ACTIVE_PRESEARCH=1`, `HOLO_D14_ACTIVE_TOL=1e-12`, `HOLO_D14_ACTIVE_PATIENCE=2` はbaselineとcandidateの両方に同じく適用した。

## whole-epoch A/B

同じ `input_snapshot.tsv` に対してV2 adaptive value-onlyを測定した。`GradientPolicy::None`、`mu_atol=1e-16`、targetごとの `mu_rtol=1e-3` / `1e-4`、fold mapとsample reuse有効、`n_r=adaptive`。1,804 trajectories・14,432 timed output rows・3 repeats。coldは毎epoch D14/topologyを生成、warmはL2の前epoch D14 root seedを使いながら毎epoch topologyを再計算、radial-onlyはtopologyを事前生成する。表はmsで、`baseline → scheduled`。

| target | lane | p50 | p90 | p95 | p99 | max | p50時間差 |
|---|---|---:|---:|---:|---:|---:|---:|
| 1e-3 | cold | 0.498696 → 0.500649 | 9.540388 → 9.507721 | 12.463632 → 12.434967 | 17.256332 → 17.249315 | 29.397703 → 29.170871 | +0.39% |
| 1e-3 | warm | 0.455698 → 0.459054 | 0.817016 → 0.813675 | 1.037099 → 1.015385 | 13.295894 → 13.213076 | 29.341049 → 29.048078 | +0.74% |
| 1e-4 | cold | 0.549765 → 0.552439 | 9.585065 → 9.553382 | 12.541100 → 12.502271 | 17.308332 → 17.334723 | 29.959758 → 29.865004 | +0.49% |
| 1e-4 | warm | 0.522023 → 0.524259 | 0.962764 → 0.956665 | 1.266920 → 1.242226 | 13.269462 → 13.217463 | 30.043036 → 29.722755 | +0.43% |

medianは4条件すべてでわずかに悪化した。高percentileは混在し、小さな改善があるものの、15%のwhole-epoch改善目標には届かない。radial-only p50はほぼ同じで、D14を含むcold/warmではscheduler固定費が見える。

同一行のpaired outputでは、各target・laneの7,216行が双方でvalue-converged。value convergence mismatch 0、numerical status mismatch 0。muの相対差はp99で最大 `3.62e-14` 以下、全行最大は `5.08e-11`（warm, 1e-3）。多くの値はbinary64で一致した。

topology status/cell数/event数と、各laneのadaptive node・evaluation・panel・split数も14,432行すべてで一致した。

## root単位の仕事量と条件

root diagnosticは同じtrajectory snapshotの7,216 epoch-rowをcold/warm各1回測定し、202,048 root、14,432個の14-root setを記録した。

| 指標 | baseline | scheduled |
|---|---:|---:|
| D14Real root updates合計 | 556,836 | 465,564 |
| rootあたり平均 | 2.756 | 2.304 |
| rootあたりp50 / p90 / p99 | 2 / 5 / 14 | 2 / 3 / 10 |
| freeze回数 | 0 | 150,586 |
| reactivation回数 | 0 | 320 |
| qfへ昇格したroot | 15,904 | 15,904 |

updateは16.4%減った。final useによる平均updatesは、positive-real候補 2.872→2.323、complex soft-cut 2.731→2.350、その他 1.789→1.671。baselineのpositive-real候補ではp90/p99が6/14、scheduledでは3/10。cluster sizeはsingleton 196,864、2-root 4,944、3-root 144、4-root 96だった。

全root setで14個の根を一対一対応でき、root-set欠落0、input-key不一致0。正規化root displacementはp50/p90が0、p99 `4.91e-20`。ただし最大は `1.98e-8` で、case 0 / linear / d_bin 2 / epoch 23 / cold の近接complex soft-cut rootに集中した。`v≈0.018536+9.48e-6 i`、nearest separationは約`2.5e-6`。これはconditioningの悪い根でのforward-position差であり、既存のqf residual gateだけから小さいroot-position errorを保証できないことを示す。該当行は両方式とも13 cells / 12 events、1e-3と1e-4のadaptive muはbinary64で一致した。独立の厳密root-location oracleによる検証ではないため、最大差は隠さず記録する。

最終Newton correctionを`|D/D'|`、位置誤差推定を`|D/D'|/(2 sqrt(Re(v)))`として記録したが、これはnear-multiple rootでforward error certificateにはならない。特にsoft-cut rootはposition error estimatorだけでprecisionを決めない。全体のroot parityと下流value/statusも合わせて評価する。

同じoutlierに限定しcluster半径を`1e-5`へ広げた追加probeでは、該当rootが2-root clusterに分類されたが、root位置差と更新回数は変わらなかった。この単一case probeでは全体のspeed/accuracy効果を評価できないため、cluster閾値変更は未採用。

root updateの減少はwhole-epoch速度に直結しなかった。root-work diagnostic stage p50はcold `0.0374→0.0414 ms`、warm `0.0377→0.0418 ms`でやや増えたが、p90/p99は短くなった。これはper-root診断captureを有効にしたstage harnessの値であり、whole-epoch判断には上表のproduction-control-flow trajectory timingを使った。qf polishと残りのtopology/physics workはほぼ不変である。

## consumer-driven precision A/B

role-specific budgetsと強いfreeze条件はresearch profileだけで試した。qf gateは緩めていない。

- `HOLO_D14_ROLE_PRECISION=1`、physical `1e-8`、soft-cut `1e-6`、other `1e-8`ではD14Real nonconvergedが108 cold中66、324 warm中192。qf warm callsもcold 4→12、warm 2→24に増えた。warm value+Jac profileの最大gradient-reference差は`1.379e-3`から`3.590e-1`へ悪化したため棄却。
- `ABS_TOL=1e-8`, `REL_TOL=1e-6`, patience 1では108 cold / 324 warm全solveがD14Real nonconvergedとなり、qf warm callsはcold 4→6、warm 2→50へ増えたため棄却。
- relative-only `REL_TOL=1e-10`はwhole-pathの一貫した改善を示さなかった。

用途はseed時点のdouble root分類を含むため、consumer-specific停止条件を使うには最終useとの対応・cluster conditioning・下流のevent error budgetをより厳密につなぐ必要がある。現状では用途別precisionを採用しない。

## case 92 / d_bin 0 tail

raw detailではcold solveの全8 repeatでdouble presearchは3 sweep、finite double seedは0/14、D14Real call 0、DD sweep 0。そのためqf cold Aberthに進み、198 sweep、qf polish中央値`27.725 ms`、D14 solve中央値`27.989 ms`、whole classify中央値`28.372 ms`だった。qf expanded residualの最大は`2.16e-27`、root-count/conjugacy/Vieta/completeness failureは0。warm trajectoryの後続epochでは前epoch rootをseedとして使い、qf cold sweepは0だった。

観測できた直接原因は「D14Real schedulerに届くfinite double seedがない」こと。なぜこのケースでdouble presearchが全root seedを失うかというdouble presearch内部の根本原因は、今回の解析範囲では未確定である。従って本変更でこのtailを解消したとはしない。

## 採否、reproduction、検証

freeze/update schedulingはresearch-onlyとして採用し、production defaultは変更しない。残した変更はroot単位の仕事量計測、opt-in scheduler、回帰テスト、trajectory/root-work harnessと集計器。consumer precision、より強い停止budget、および一時的なfast-gate試験は本線に採用していない。qf gateやD14 solve routerの変更はない。

初期試作では、通常solveでも未使用の最終per-root診断再評価を実行しており、whole-epochを数%遅くしていた。診断capture時だけ評価するよう直してから最終A/Bを取り直した。`scheduled_abs1e-10_rel1e-8.tsv.gz`と`summary_before_diagnostic_elision.json`はこの棄却前段階のevidenceとして保存し、最終値には`scheduled_after_diag_elide.tsv.gz`を使った。

ビルド・テスト:

```sh
cmake -S tests/holonomic_cpp -B build-holonomic-m7 -DCMAKE_BUILD_TYPE=Release
cmake --build build-holonomic-m7 -j2
ctest --test-dir build-holonomic-m7 --output-on-failure
python3 -m py_compile benchmarks/holonomic/summarize_d14_root_scheduling_phase2.py
git diff --check
```

whole trajectory benchmarkの再現コマンド:

```sh
g++ -O3 -march=native -funroll-loops -ffp-contract=fast -fno-math-errno \
  -DNDEBUG -std=gnu++17 -I src \
  benchmarks/holonomic/bench_d14_positive_trajectory.cpp \
  -o /tmp/bench_d14_root_schedule -lquadmath

taskset -c 0 env \
  HOLO_D14_LOCAL_PAIRS=1 HOLO_D14_ACTIVE_PRESEARCH=1 \
  HOLO_D14_ACTIVE_TOL=1e-12 HOLO_D14_ACTIVE_PATIENCE=2 \
  /tmp/bench_d14_root_schedule \
  evidence/holonomic/d14_root_scheduling_phase2_20260912/input_snapshot.tsv \
  /tmp/d14_baseline.tsv 3

taskset -c 0 env \
  HOLO_D14_LOCAL_PAIRS=1 HOLO_D14_ACTIVE_PRESEARCH=1 \
  HOLO_D14_ACTIVE_TOL=1e-12 HOLO_D14_ACTIVE_PATIENCE=2 \
  HOLO_D14_REAL_ACTIVE=1 HOLO_D14_REAL_ABS_TOL=1e-10 \
  HOLO_D14_REAL_REL_TOL=1e-8 HOLO_D14_REAL_PATIENCE=2 \
  /tmp/bench_d14_root_schedule \
  evidence/holonomic/d14_root_scheduling_phase2_20260912/input_snapshot.tsv \
  /tmp/d14_scheduled.tsv 3

python3 benchmarks/holonomic/summarize_d14_root_scheduling_phase2.py \
  evidence/holonomic/d14_root_scheduling_phase2_20260912 \
  --output /tmp/summary_phase2.json
```

入力snapshot SHA256は`6646492658ed416e75cb8c6c821ee853f79cdf7761fc41441186aa67b81fa05b`。この測定環境はIntel Xeon Gold 6530、GCC 11.5.0で、実行をCPU 0に固定した。raw TSV/TSV.GZ、stage/root-work profile、case92 raw、whole-epoch summary、実行metadataと全raw artifactのSHA256 manifestを[Phase 2 evidence](../../evidence/holonomic/d14_root_scheduling_phase2_20260912/)に保存した。

full build成功、CTest 18/18 pass。whole-epoch値/status、root-setの一対一対応と既存qf gateは確認した。performance adoption criterionは未達であり、near-cluster rootのforward-position差も残るため、production adoptionはしない。
