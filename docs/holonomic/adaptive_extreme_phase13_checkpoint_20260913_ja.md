# Phase 13: D14演算組合せ、warm費用の確認

基準b6967c0。INTERLEAVED_HORNER + RESIDUAL_MAX_NORMの組合せ。
既定経路はまだ変更していない。CPU0、Phase10 runner、既存input_snapshot、
全14,432行、3 repeats、同flags。比較baselineはPhase11 whole_endpoint。

| RelTol | cold p50 baseline→candidate ms | warm p50 | cold p99 | warm p99 |
|---|---|---|---|---|
| 1e-3 | .483153→.469304 | .433158→.425369 | 3.269904→3.220302 | 1.254289→1.239364 |
| 1e-4 | .526939→.513095 | .493633→.486406 | 3.290688→3.265957 | 1.575077→1.544414 |

全laneで値差0、node/status差0、7216/7216 convergence/tol。
VBM1e-6要求tol超過は1e-3で0、1e-4で既存の3件。
1e-3 cold max56.081→56.072、warm55.963→55.895ms。
異なる日時の逐次測定なのでsub-percentの差は断定しない。
残差reductionは実際の根に2^-80〜2^80の摂動を加えたqfテストでも旧値と完全一致。
root scheduling単体テストを両flag付きでpass。

再現:
```sh
c++ -O3 -DNDEBUG -std=gnu++17 -fext-numeric-literals -Isrc -march=native -funroll-loops -ffp-contract=fast -fno-math-errno -DHOLO_D14_INTERLEAVED_HORNER -DHOLO_D14_RESIDUAL_MAX_NORM evidence/holonomic/adaptive_estimator_phase10_20260913/runner_hybrid.cpp -o /tmp/p13_combined -lquadmath
taskset -c 0 /tmp/p13_combined evidence/holonomic/v2_adaptive_best_trajectory_20260913/input_snapshot.tsv evidence/holonomic/adaptive_extreme_phase13/whole_combined.tsv 3
python benchmarks/holonomic/summarize_adaptive_reciprocal.py --baseline whole_endpoint.tsv --candidate ../adaptive_extreme_phase13/whole_combined.tsv --output ../adaptive_extreme_phase13/whole_summary.json
```

## Warm内部profile

bench_adaptive_stage_profileにwarmモードを追加。同一case/config/profile/db内だけ
PreparedEpochGeometryを使い、trajectory初回1804行を集計から除いた5412行。
これはprofile付き単回測定で、全件whole採用ベンチとは区別する。
列のtimerは包含関係があり、中央値同士を足してexclusive overheadと解釈しない。

presearch p50 .035627ms、real .022912、residual .026505、
cell probes .036820、expand .010208、setup .009695。
profile付きwhole .471485ms、topology .236389ms、physics .165871ms。
前回から追加したroot-work診断の費用も含むため、非profile時間と混ぜない。
次はcell probeとqf残差の費用を検討する。VBM全面勝利は未達。

## 採用監査

全14,432 root setsでexport座標差0、role/physical差0、event/cell/status差0。
presearch/DD/qf-warm/qf-cold sweeps差0、fallback差0、qf-cold総数100のまま。
独立root集合の証明を新設したわけではなく、既存最終gateを通る候補の
比較である。raw roots/eventsはgzip保存。
この結果から両最適化を既定ONとする。旧経路へのA/Bスイッチは
HOLO_D14_DISABLE_INTERLEAVED_HORNERとHOLO_D14_DISABLE_RESIDUAL_MAX_NORM。
production routerとfixed-n_r積分設計は変更していない。
