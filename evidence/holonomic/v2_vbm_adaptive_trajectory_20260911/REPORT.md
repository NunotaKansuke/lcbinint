# V2 adaptive trajectory benchmark — D14 cold / trajectory warm

実行時の `dev/holonomic` HEAD は `26593fa005ac0c82eb13c11dd7c7ef79dd976867`。既存の q–rho corpus と同じ入力変換、同じ adaptive value-only 条件、同じ `RelTol=1e-3, 1e-4` を使い、D14/topologyを含むfull epochを追加測定した。

## 測定レーン

| lane | topology/D14 | epoch間state | timer boundary |
|---|---|---|---|
| `radial-only` | 事前生成 | なし | `flux_adaptive_integrate` のみ |
| `full-cold` | 毎epoch `classify_cells` | なし | production `epoch_adaptive` 全体 |
| `full-warm` | 毎epoch `prepared_topology` | L2 D14 warm seed | `prepared_topology` + adaptive全体 |

`full-warm` は `allow_topology_reuse=false`, `allow_warm_d14=true`, `l2_drift=1e18`。したがってL1のcell-plan verbatim reuseは使わず、各epochで `classify_cells` がtopology authorityとして実行される。trajectoryの最初のepochは必ずL3 cold build、後続3 epochはL2 warm-seeded D14を試す。4 epochを同じ順序で処理し、各lane・targetごとに3回測定してepochごとの中央値をrawへ保存した。

入力は1,804 trajectory、各4 epoch、uniform/linearの2 profile、合計14,432 rows（targetを含む）。`LensParams` は既存q–rho runnerと同じく

```text
LensParams{time, y, rho, 1/q, s, true}
```

で作った。radial-onlyのtopology/D14事前生成は意図的にtimer外なので、これは旧benchmarkと同じpure adaptive-kernel比較であり、full epochの速度とは別の量である。

## Whole epoch timing

値は `p50 / p95 / p99 / max` ms。`all` は4 epoch全部、`steady` はtrajectoryの最初のepochを除く後続3 epochである。

| profile | RelTol | scope | radial-only | full-warm | full-cold |
|---|---:|---|---:|---:|---:|
| uniform | 1e-3 | all | 0.1262 / 0.2717 / 0.3951 / 0.7063 | 0.6014 / 1.3819 / 13.6052 / 28.9283 | 0.7366 / 12.7485 / 17.6642 / 29.4225 |
| uniform | 1e-3 | steady | 0.1235 / 0.2674 / 0.3951 / 0.7063 | 0.5605 / 1.2328 / 1.4741 / 15.9600 | 0.7329 / 12.7500 / 17.9282 / 29.4225 |
| uniform | 1e-4 | all | 0.1628 / 0.3778 / 0.6479 / 2.0735 | 0.6568 / 1.5153 / 13.6049 / 29.4527 | 0.7852 / 12.9686 / 17.4542 / 29.5717 |
| uniform | 1e-4 | steady | 0.1600 / 0.3677 / 0.6480 / 2.0011 | 0.6213 / 1.3631 / 1.6522 / 15.9124 | 0.7831 / 12.9688 / 17.4879 / 29.4186 |
| linear | 1e-3 | all | 0.1709 / 0.3936 / 0.5207 / 0.8905 | 0.6707 / 1.5108 / 13.6843 / 29.7103 | 0.7983 / 12.8952 / 17.7419 / 30.1378 |
| linear | 1e-3 | steady | 0.1688 / 0.3886 / 0.5196 / 0.8905 | 0.6343 / 1.3282 / 1.6023 / 15.9418 | 0.7947 / 12.9817 / 17.9443 / 30.1378 |
| linear | 1e-4 | all | 0.2134 / 0.5689 / 0.8749 / 1.8925 | 0.7486 / 1.6233 / 13.8092 / 30.3995 | 0.8655 / 12.9065 / 17.9665 / 30.8928 |
| linear | 1e-4 | steady | 0.2104 / 0.5561 / 0.8521 / 1.6126 | 0.7112 / 1.4679 / 1.7709 / 16.1447 | 0.8618 / 12.9273 / 18.1130 / 30.8928 |

全条件を混ぜた参考値は、`full-cold` が `0.7951 / 12.8936 / 17.6710 / 30.8928` ms、`full-warm` が `0.6682 / 1.5216 / 13.6866 / 30.3995` ms。最初のepochを除くsteady stateでは、`full-cold` が `0.7927 / 12.9293 / 17.7504 / 30.8928` ms、`full-warm` が `0.6289 / 1.3556 / 1.6338 / 16.1447` msとなった。中央値ではwarmがcoldより約1.26倍速く、D14が難しいtailではsteady p95/p99も大幅に下がる。

warm全体のmixは、L3 cold build 3,608、L2 warm recompute 10,824、warm seed使用10,824、L1 reuse 0。これは1,804 trajectory × 2 targetに対して、各trajectoryの初回だけcoldであることと一致する。

## Stage split

全14,432 rowsを混ぜた `p50 / p95` msは次のとおり。

| lane | whole | topology/D14 | adaptive body | setup | physics | estimator | scheduler |
|---|---:|---:|---:|---:|---:|---:|---:|
| full-cold | 0.7951 / 12.8936 | 0.5713 / 12.6227 | 0.1681 / 0.4526 | 0.0101 / 0.0183 | 0.1254 / 0.3745 | 0.0047 / 0.0132 | 0.0012 / 0.0024 |
| full-warm | 0.6682 / 1.5216 | 0.4068 / 1.1257 | 0.2191 / 0.5371 | 0.0099 / 0.0181 | 0.1740 / 0.4671 | 0.0046 / 0.0123 | 0.0011 / 0.0023 |
| radial-only | 0.1618 / 0.4272 | 0 / 0 | 0.1618 / 0.4272 | 0.0092 / 0.0169 | 0.1218 / 0.3579 | 0.0043 / 0.0106 | 0.0011 / 0.0021 |

full-coldのp95/p99は主にD14/topologyの難しいtrajectoryに支配される。full-warmでは初回epochにその費用が残るため、all-epoch p99には難例が残るが、steady p99は1.6338 msまで下がる。warmのadaptive bodyがradial-onlyより重いのは、warm topologyの結果を受けたcontinuation/certificate処理を含むためで、D14 warm seedの利益はtopology stageに現れる。

wholeと同一repから選んだstageの差 `whole - (topology + setup + physics + estimator + scheduler)` は、全laneでp50約0.023 ms、p95約0.05 ms、最大約0.50 ms。負値、非有限値、`whole < topology` は0 rowsだった。従って今回のfull timingは、D14をtimer外へ逃がした値ではない。

## 2×3 whole-epoch figure

既存の標準図と同じカラーバー、境界値、色、レイアウトを使用した。上段は `R=t_VBM/t_V2` のruntime map、下段はVBM `RelTol=1e-6` referenceとの差のcell p95 mapで、列は `q×rho`, `A_fs×rho`, `A_fs×q`。`A_fs` は従来図と同じく入力に保存されたreference magnificationを座標に使う。`full-cold` と `full-warm` は別図に分け、V2側の分母だけをそれぞれD14/topology込みの `cold_ms` / `warm_ms` にした。灰色の斜線は、その射影binの行数が8未満、またはvalue coverageが不完全な箇所であり、今回のq–rho mapに非収束の穴はない。

速度比の上段は既存のVBM `selected_seconds` artifactを比較側に使い、V2側は今回のwhole-epoch timingを使う。従って、これはD14込みV2の位置を旧図の速度比表現で確認する図であり、pure-kernel同一timerの比較は前節のradial-only laneで分離している。

`uniform` / `linear`、`RelTol=1e-3` / `1e-4`、`full-cold` / `full-warm`の8図を生成した。

- [uniform / 1e-3 / full-cold PNG](figures/q_rho_uniform_1e-3_v2_adaptive_full-cold_vbm1e-6.png) / [PDF](figures/q_rho_uniform_1e-3_v2_adaptive_full-cold_vbm1e-6.pdf)
- [uniform / 1e-3 / full-warm PNG](figures/q_rho_uniform_1e-3_v2_adaptive_full-warm_vbm1e-6.png) / [PDF](figures/q_rho_uniform_1e-3_v2_adaptive_full-warm_vbm1e-6.pdf)
- [uniform / 1e-4 / full-cold PNG](figures/q_rho_uniform_1e-4_v2_adaptive_full-cold_vbm1e-6.png) / [PDF](figures/q_rho_uniform_1e-4_v2_adaptive_full-cold_vbm1e-6.pdf)
- [uniform / 1e-4 / full-warm PNG](figures/q_rho_uniform_1e-4_v2_adaptive_full-warm_vbm1e-6.png) / [PDF](figures/q_rho_uniform_1e-4_v2_adaptive_full-warm_vbm1e-6.pdf)
- [linear / 1e-3 / full-cold PNG](figures/q_rho_linear_1e-3_v2_adaptive_full-cold_vbm1e-6.png) / [PDF](figures/q_rho_linear_1e-3_v2_adaptive_full-cold_vbm1e-6.pdf)
- [linear / 1e-3 / full-warm PNG](figures/q_rho_linear_1e-3_v2_adaptive_full-warm_vbm1e-6.png) / [PDF](figures/q_rho_linear_1e-3_v2_adaptive_full-warm_vbm1e-6.pdf)
- [linear / 1e-4 / full-cold PNG](figures/q_rho_linear_1e-4_v2_adaptive_full-cold_vbm1e-6.png) / [PDF](figures/q_rho_linear_1e-4_v2_adaptive_full-cold_vbm1e-6.pdf)
- [linear / 1e-4 / full-warm PNG](figures/q_rho_linear_1e-4_v2_adaptive_full-warm_vbm1e-6.png) / [PDF](figures/q_rho_linear_1e-4_v2_adaptive_full-warm_vbm1e-6.pdf)

## Accuracy and status

入力に保存された既存 VBM `RelTol=1e-6` referenceとの相対差は、全14,432 rowsで全lane同じ分布だった。

```text
p50  4.221611e-7
p90  4.010123e-6
p95  6.659467e-6
p99  1.413752e-5
max  2.706967e-4
```

`radial-only`, `full-cold`, `full-warm` のすべてで `value_converged=14,432/14,432`, `stop=Converged=14,432`, `status=OK=14,432`。coldとradialの値はこのrawで全行一致し、warmとradialのreference scale相対差はp99 `5.41e-15`、最大 `8.04e-11`。status mismatchとvalue convergence mismatchは全比較で0である。

このreference差はV2 adaptive値と既存VBM referenceの比較であり、VBM自身の誤差を含む。速度・精度の最終判断では、旧pure-kernel結果とfull epoch結果を混ぜていない。

## Reproduction and artifacts

- [command.txt](command.txt): compile、CPU pin、runner、summary生成コマンド
- [v2_adaptive_trajectory_runner.cpp](v2_adaptive_trajectory_runner.cpp): benchmark source
- [input_snapshot.tsv](input_snapshot.tsv): 4 epoch trajectory input snapshot
- [results.tsv](results.tsv): 14,432-row raw, 73 columns
- [summary.json](summary.json): machine-readable coverage/timing/parity summary
- [make_figures.py](make_figures.py), [figures.json](figures.json): 2×3 figure source and manifest
- [figures/](figures/): standard-style cold/warm whole-epoch PNG/PDF figures
- [runner_stderr.log](runner_stderr.log): progress and completion log
- [binary.sha256](binary.sha256), [results.sha256](results.sha256): hashes

旧pure-kernel artifactの同一3,608×4条件における参考値は `p50=0.155496`, `p95=0.411062`, `p99=0.661292`, `max=2.046770` ms。これは別runの既存artifactなので、今回の同一run内のlane比較を置き換えるものではない。
