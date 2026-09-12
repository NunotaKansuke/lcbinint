# D14 root work scheduling 実験 checkpoint（2026-09-12）

このcheckpointは、`dev/holonomic` の `3d30124` を起点に、D14の全根を保持したまま、根ごとの仕事量と混合精度の昇格単位を変える実験を記録する。変更範囲はisolated holonomicに限定し、`.claude/` と既存の未追跡成果物は変更していない。

## 結論

現行のbalanced double Aberth → D14Real/qf polish → residual/completeness certificateという保証経路は維持した。今回の実装は次の二つのopt-in候補である。

* `HOLO_D14_LOCAL_PAIRS=1` は、近接pairがある根でも安全な相互作用をKahan補償doubleで保持し、危険なpairだけをD14Realで再計算する。既存のD14Real残差、根数、共役性、Vieta、global certificateは変更していない。
* `HOLO_D14_ACTIVE_PRESEARCH=1` は、doubleのbasin presearchで小さい補正が続いた根を一時停止する。停止した根も他の根の相互作用和には残し、DD/qf polishとglobal certificateを必ず通す。非有限または不正な作業結果は既存Aberth presearchへ戻す。

最も安全性が確認できた組合せは、`HOLO_D14_LOCAL_PAIRS=1`、`HOLO_D14_ACTIVE_PRESEARCH=1`、`HOLO_D14_ACTIVE_TOL=1e-12`、`HOLO_D14_ACTIVE_PATIENCE=2` である。ただしp90/p99の全条件で安定した非回帰をまだ証明できていないため、production defaultへは昇格せず、research A/B flagのまま残した。

同一の1,804 trajectory（14,432 rows、reps=3、adaptive `n_r`）でのvalue-only whole-epochは次のとおりである。数値は `baseline_current` と `active_1e-12_p2_local_current` の比較で、列は p50 / p90 / p95 / p99 / max（ms）。

| RelTol | lane | baseline | active + local-pair |
|---|---|---:|---:|
| 1e-3 | cold | 0.6350 / 9.5054 / 12.4031 / 17.1281 / 29.3540 | 0.5422 / 9.5011 / 12.4193 / 17.2869 / 29.3734 |
| 1e-3 | warm | 0.5401 / 0.9388 / 1.1818 / 13.2464 / 29.2936 | 0.4910 / 0.8693 / 1.0974 / 13.1904 / 29.2986 |
| 1e-4 | cold | 0.6860 / 9.5564 / 12.4873 / 17.2307 / 30.2821 | 0.5976 / 9.5558 / 12.4804 / 17.4617 / 30.1182 |
| 1e-4 | warm | 0.6099 / 1.1065 / 1.4546 / 13.2599 / 30.0072 | 0.5574 / 1.0259 / 1.3520 / 13.1962 / 30.0538 |

whole trajectoryのvalue convergenceは候補でcold/warmとも4条件すべて `7216/7216`。baselineはwarmだけ各tolで `7214/7216` で、既存の`EventLocationLimited` 2 rowsが残った。候補のwarm statusは全row `OK` だった。candidateのp50はwarmで約1.10倍、coldで約1.15〜1.17倍速いが、cold p99は約1%の測定揺らぎを含むため、これだけでproduction採用とはしない。

同じ収束rowをbaselineと候補で対応させたmu差は、safe候補の相対差p99がcold/warmとも約`8e-14`、最大でもcold `3.51e-11`、warm `7.27e-11`だった。warmの2 status差はbaselineが失敗した2行を候補が収束させた差である。入力reference（VBM RelTol=1e-6）に対する相対誤差の分布は、同一rowでbaselineと候補が一致した。

遅いtailは今回の変更ではほぼ縮まらなかった。最遅行群には`case_id=92, d_bin_index=0`が繰り返し現れ、`RelTol=1e-3/1e-4`のcold wholeは約29–30 ms、そのうちtopologyが約28.2–28.4 ms、adaptive本体は約1.1–1.9 msだった。warmでも同じgeometryが約28 msのtopology時間を要した。このbenchmarkの`topology_ms`はD14 solve、real-positive isolation/classification等をまとめた値で、内訳のcase別profileはまだない。

このcaseをtrajectory入力から抽出して単独の`bench_d14_structure`に追加したところ、expanded-coefficient経路とblock D14経路のpositive-real root-set比較は`1/8`で不一致（最大相対差`1.026e-6`）だった。一方、両者のworst qf residualは`9.066e-18`と`4.943e-17`だった。これは今回のactive/local-pair最適化による差ではなく、非常に小さいrhoの既存hard geometryにおけるrepresentation parityの未解決点である。したがって、tail短縮やdefault昇格の根拠にはせず、raw診断を残して別途調べる。

108-case profile（trajectory_epochs=4、`bench_v2_profile`）でも、value+5Jacのp50は baseline → safe candidate で、cold `0.7405 → 0.6727 ms`、warm `0.6724 → 0.6598 ms` だった。value-onlyは cold `0.6354 → 0.5925 ms`、warm `0.5475 → 0.5280 ms`。profileは候補探索の内訳を見るための補助測定であり、採否は上の全trajectoryと独立root parityを優先した。

## 実装した候補

`d14_structure.hpp` の `aberth_d14_real_mixed` に、危険pairのindexを固定配列へ収集する経路を追加した。従来経路では危険pairが一つでもある根の13相互作用をすべてD14Realで再計算する。local-pair経路では安全pairをKahan補償doubleで和に加え、危険pairだけをD14Realで加える。D14RealのD/D'評価と反復、最終残差gateは共通である。

`radial_events.hpp` にroot-wise active presearchを追加した。`patience` 回連続して補正が指定値以下の根をpresearch中だけ休止する。休止根をdeflationしたり、相互作用から除いたりはしない。active結果のサイズ・有限性を検査し、異常時は同じseedから通常double Aberthを再実行する。このfallback件数も`V2Profile`へ記録した。

`bench_v2_profile.cpp` と `v2_profile.hpp` には次を追加した。

* active presearchの呼び出し数、sweep数、skip update数、fallback数
* D14Realのmixed/dangerous pair数、row全体再計算数、local-pair call数と危険row数

## A/B結果

独立root benchmarkは115 cases（108 benchmark + 7 reference）、reps=15で行った。safe設定 `active_tol=1e-12, patience=2` は、local-pair併用時にparity failure `0/115`、root-set最大相対差 `3.548e-10`、worst qf residual baseline `6.519e-14` / candidate `6.694e-14` だった。`active_tol=1e-11, patience=1` もこのroot benchmarkでは`0/115`だったが、p99の余裕が小さいためsafe設定を主結果にした。

`1e-10, patience=1` は失格である。trajectoryのstatusだけなら全rowが通ったが、独立root benchmarkで`rand028`の2条件が既存parity gateを落とし、candidate residualは`1.153e-13`になった。これは「後段のstatusがOK」をroot精度の代用にできない例である。

pair-local単独では、108-case profileでbaselineのD14Real `dangerous pairs=24336, full-recompute-rows=1872` に対し、local経路は`dangerous pairs=1854, full-recompute-rows=0`となった。しかしwhole wallの差は測定ノイズを明確には超えず、単独採用の根拠にはしなかった。activeとの併用では、safe設定のcold presearch sweepは`21600 → 約21026`、warmは`19440 → 約18742`へ減った。

## 不採用にした実験

`HOLO_D14_PRESEARCH_MAX` を固定的に短くする方法は、root basinを見つける仕事を後段へ押し付けるだけだった。

* `max=20`: 108-case profileのcoldでD14Real非収束106件、qf warm 104 calls、qf cold 12 calls、cold value p50 `3.7628 ms`。
* `max=40`: D14Real非収束62件、full-recompute-rows 1276、cold value p50 `0.7761 ms`。
* `max=80`: 短いprofileでは良く見えたが、全trajectoryではcold value convergenceが各tol `7188/7216`、warmが`7208/7216`に低下した。従って採用しない。

warm seedをそのままpolishへ渡してpresearchを省く案も不採用である。`HOLO_D14_SKIP_WARM_PRESEARCH=1` はwarm value p50 `0.6071 ms`、D14Real非収束6件、full-recompute-rows 6786となった。direct warm Newtonはattempt 345に対してsuccess 6、reject 339、D14Real非収束82、qf cold 71、warm value p50 `0.6522 ms`、p90 `10.3114 ms`だった。

degree-14固定バッファの小実験も行った。作業配列をstackへ移しても、同じtrajectoryでp50/p99が一貫して改善せず、`mu`差は0だったため、実装から戻した。rawは`evidence/holonomic/d14_root_optimization_20260912/rejected/`に残している。

Bernstein–Descartes、companion QZ、全根を捨てる正実根専用solverは今回実装していない。現行solverが一度に提供しているcomplex soft-cut、warm seed、全根certificateの情報を別計算で再構築する費用を、今回の実測だけでは回収できないためである。

## 再現方法

入力snapshotはこのevidence directoryの`input_snapshot.tsv`に保存した。元入力と同一で、SHA-256は`6646492658ed416e75cb8c6c821ee853f79cdf7761fc41441186aa67b81fa05b`である。実行環境は`rogue1`、Intel Xeon Gold 6530、GCC 11.5.0、single core `taskset -c 0`、`-O3 -DNDEBUG -std=gnu++17 -lquadmath`。rawと主要ソースのSHA-256、reproduction metadataは同じdirectoryの`provenance.txt`に記録した。

```text
c++ -O3 -DNDEBUG -std=gnu++17 -I src \
  benchmarks/holonomic/bench_d14_positive_trajectory.cpp \
  -o /tmp/d14_root_optimization_runner -lquadmath

taskset -c 0 /tmp/d14_root_optimization_runner \
  evidence/holonomic/d14_root_optimization_20260912/input_snapshot.tsv \
  evidence/holonomic/d14_root_optimization_20260912/baseline_current.tsv 3

HOLO_D14_LOCAL_PAIRS=1 \
HOLO_D14_ACTIVE_PRESEARCH=1 \
HOLO_D14_ACTIVE_TOL=1e-12 \
HOLO_D14_ACTIVE_PATIENCE=2 \
taskset -c 0 /tmp/d14_root_optimization_runner \
  evidence/holonomic/d14_root_optimization_20260912/input_snapshot.tsv \
  evidence/holonomic/d14_root_optimization_20260912/active_1e-12_p2_local_current.tsv 3

python3 benchmarks/holonomic/summarize_d14_root_optimization.py \
  evidence/holonomic/d14_root_optimization_20260912 \
  --baseline baseline_current.tsv.gz \
  > evidence/holonomic/d14_root_optimization_20260912/summary.json
```

profileの再現には、上記の候補環境変数を付けて次を実行する。

```text
cmake --build build-holonomic-m7 --target bench_v2_profile bench_d14_structure -j4
taskset -c 0 build-holonomic-m7/bench_v2_profile \
  evidence/holonomic/gm_coverage_cases.tsv 3 4

HOLO_D14_LOCAL_PAIRS=1 HOLO_D14_ACTIVE_PRESEARCH=1 \
HOLO_D14_ACTIVE_TOL=1e-12 HOLO_D14_ACTIVE_PATIENCE=2 \
taskset -c 0 build-holonomic-m7/bench_v2_profile \
  evidence/holonomic/gm_coverage_cases.tsv 3 4
```

## 検証と残存コスト

通常のCTestは17/17 passした。候補環境変数を付けた`test_holonomic_m7`、`test_d14_positive`、`test_adaptive_radial`、`test_quartic_sturm`、`test_root_pair`もすべてpassした。独立D14 root parity、trajectory coverage、qf residual/completenessを緩める変更は入れていない。

現在の大きな費用は、profile上ではD14の構造評価・D14Real polish・残差確認を含むtopology固定費である。active候補でD14 presearchの仕事は減るが、難しいtailはquartic/root trackingと既存のD14/qf certificationが支配する。warmでp50は改善したものの、p99/maxの難例を消すには、根を捨てずにclusterの状態を保持して次epochへ渡す設計が次の課題になる。

raw benchmark、profile、root parity結果、失敗候補は`evidence/holonomic/d14_root_optimization_20260912/`に保存した。機械可読な集計は`summary.json`、集計スクリプトは`benchmarks/holonomic/summarize_d14_root_optimization.py`である。
