# D14 median / topology Phase 5 checkpoint（2026-09-13）

## 判断

中央値を0.x ms前半へ下げるため、D14の通常経路とD14省略trajectoryの両側を調べた。今回試したpair共有Jacobi、structured residual、active schedule中間設定はproductionへ採用しない。最新adaptiveでもprojected complex soft-cutを外すと11 target-rowでvalue coverageを失うため、旧cell/event planをそのまま使うD14省略も安全ではない。

現在のwhole中央値はRelTol=1e-3でcold約0.55 ms、warm約0.49 ms、radial-only約0.15 ms。warm D14/topology約0.34 msが最大の削減対象である。ただし、その全額を省くにはphysical eventの局所更新に加え、soft-cutが避けているnear-foldの不安定node配置を安価に認証・回避する必要がある。

## A/Bした候補

### unordered pair共有Jacobi presearch

14根の相互作用をordered 182除算からunordered 91除算へ減らし、全rootを同時更新する研究実装を試した。topology/event parityは保ったが、Gauss-Seidel型の現行in-place更新より収束域が悪化した。

- cold classify p50: 0.8464 → 0.9609 ms（+13.5%）
- warm classify p50: 0.7267 → 0.8416 ms（+15.8%）
- qf-cold: cold 58→218、warm 44→82
- cold p99: 3.70→18.93 ms、warm p99: 1.38→13.36 ms

一sweepの除算削減より、反復増加とqf fallbackが大きい。実装はrevertした。

### structured C3/G4/Z3 residual evaluator

正常行が毎回払うqf residual gateを、expanded degree-14 Hornerからexact structured evaluatorへ置換した。判定、event数、qf-cold回数は一致したが、複素block式の演算が多く遅かった。

- residual p50: 約0.0307 → 0.1022 ms（約3.3倍）
- cold classify p50: +8.2%
- warm classify p50: +9.9%

gateは緩めず、実装をrevertした。

### active schedule grid

既存safe設定 `1e-12 / patience=2` と、前checkpointの攻めた `1e-11 / patience=1` の間を調べた。`3e-12 / patience=1 + interval Rouché` は14,432行でcoverage/statusを維持したが、独立run間でradial-onlyも1〜2%動く測定変動があり、whole差はp50 -0.6〜+0.0%、p90以降は概ね回帰した。前checkpointの約3〜4%改善を超えず、採用しない。

### projected complex soft-cut省略の再監査

最新HEADと現行adaptiveで再実行した。all-soft baselineで収束していた7,216行/targetに対し、no-softは次を失った。

| RelTol | cold / warm / radial coverage | lost rows |
|---|---:|---:|
| 1e-3 | 7211 / 7211 / 7211 | 各5 |
| 1e-4 | 7210 / 7210 / 7210 | 各6 |

全て `TopologyUnresolved / GRADIENT_UNRELIABLE`。cold/warm/radialで同じ行が落ちるためD14 warm seedやtimerの問題ではなく、soft-cut除去後にadaptive nodeが不安定なnear-fold位置へ入る問題が残っている。失敗はcase 0/configuration 0/d-bin 0のepoch 0, 7, 15, 23に集中する。soft-cutを無条件に省略してcoverageを作る案は棄却する。

## 次の本線

大幅短縮の候補は、前epochからphysical foldを `P=P_t=0` で局所更新し、その局所enclosureとevent間の除外証拠を更新するtrajectory certificateである。ただし過去の全面stationary atlas（約42 ms）のように全domain tensor coverを毎epoch再構築しない。

次のprototypeでは、旧soft-cut回帰11行を設計入力として、各updated physical eventの近傍だけをDD/qf Krawczykで認証し、adaptive schedulerがfold enclosureへ危険に接近する場合だけ安全な追加panel cutを置く。event間は前epochの除外boxを係数差boundで一括再認証し、通らない局所区間だけ分割する。これで全event identity/completenessを証明できたepochだけD14を省略し、失敗時は現在のL2 warm D14へ戻す。目標は認証費p50 0.05 ms以下、D14省略率90%以上、14,432行のcoverage/status/value parity維持である。

production router、fixed-n_r API、PF6/GM経路、既存qf/root certificateは変更していない。今回のコード試作は全てrevertし、checkpointとrawだけを残す。
