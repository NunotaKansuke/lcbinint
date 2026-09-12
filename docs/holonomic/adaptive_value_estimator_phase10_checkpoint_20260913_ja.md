# Adaptive V2 value estimator Phase 10 checkpoint

## 結論

D14 + symmetric root pair `(m,v)` + K-rule の経路を維持し、adaptive radial value の既定error estimatorをhybrid embedded estimatorへ変更した。従来のweighted interpolation-detailだけを積分誤差とみなすと、符号相殺後の積分誤差に比べて過大になりやすかった。新しいvalue用推定値は

```text
E_radial = max(detail / 8, min(detail, 2 * abs(Q_fine - Q_coarse)))
```

である。`detail`の減衰確認、inner/geometry/event/roundoff ledger、topology/branch certificateは従来通り維持する。gradientは今回変更せず、従来のweighted detailを使う。

embedded差だけを使う案は棄却した。100 trajectory probeの`RelTol=1e-3`で、1行のrelative reference errorが約`1.15e-2`へ跳ねた。低次数embedded差の偶然の相殺が原因であり、weighted detailの1/8 floorを加えると同じprobeの最大誤差は`6.97e-5`となった。

## 14,432-row matched A/B

入力は既存q--rho trajectory snapshot（1,804 trajectory、各4 epoch、uniform/linear、`RelTol=1e-3/1e-4`）。value-only、同一compiler、CPU 0固定、各lane 1 repeatで、weighted baselineとhybridを別binaryとして測定した。full-coldはD14/topologyを毎epoch含み、full-warmは前epochのD14 rootsをseedとして再認証し、radial-onlyはtopologyをtimer外で共有する。

| tolerance | lane | p50 baseline → hybrid ms | p90 | p99 | node p50 |
|---|---|---:|---:|---:|---:|
| 1e-3 | full-cold | 0.548 → 0.525 | 0.887 → 0.844 | 3.482 → 3.413 | 83 → 69 |
| 1e-3 | full-warm | 0.493 → 0.456 | 0.822 → 0.770 | 1.487 → 1.291 | 83 → 69 |
| 1e-3 | radial-only | 0.150 → 0.128 | 0.302 → 0.253 | 0.504 → 0.410 | 83 → 69 |
| 1e-4 | full-cold | 0.594 → 0.572 | 0.984 → 0.920 | 3.539 → 3.507 | 114 → 97 |
| 1e-4 | full-warm | 0.559 → 0.520 | 0.952 → 0.865 | 2.245 → 1.618 | 114 → 97 |
| 1e-4 | radial-only | 0.190 → 0.167 | 0.409 → 0.335 | 0.878 → 0.576 | 114 → 97 |

両方式・両tol・全laneでvalue coverageは`7216/7216`。hybridとbaselineのmu差をreference scaleで規格化した最大値は、`1e-3`で`2.07e-4`、`1e-4`で`2.02e-5`であり、いずれも要求tol未満だった。

VBM `RelTol=1e-6` referenceに対するhybrid誤差は、`1e-3`でp50/p90/p99/max = `6.15e-7 / 6.86e-6 / 3.09e-5 / 2.71e-4`、`1e-4`で`3.65e-7 / 3.39e-6 / 1.24e-5 / 2.71e-4`。最大値はbaselineにも存在するreference差が支配している。

既存の独立GL radial + direct angular reference auditは、reference判定可能458行で新規violation `0`。quartic Sturm、finite-source binary、adaptive radialの関連CTestも全passした。analytic Jacobian checkerも保存した。gradient estimator自体は変更していない。

## 判断と限界

hybrid estimatorをexperimental adaptive pathの既定として採用した。production routerとfixed-`n_r` APIは変更していない。whole-epoch中央値の改善はD14固定費によりcoldで約4%、warmで約7%に留まるが、radial bodyは`1e-3`で約15%短縮した。

timingは全corpus 1 repeatなので、数%差の厳密なCPU性能主張には追加repeatが必要である。一方、node数削減と全行accuracy auditはtiming noiseに依存しない。次の候補は、panelごとのdetail/embedded比から固定`1/8` floorをさらに鋭くできるcertificateだが、今回棄却したpure embedded gateへ戻してはならない。

## 再現

コマンド、runner source、raw TSV、summary JSON、hash、machine情報は `evidence/holonomic/adaptive_estimator_phase10_20260913/` に保存した。
