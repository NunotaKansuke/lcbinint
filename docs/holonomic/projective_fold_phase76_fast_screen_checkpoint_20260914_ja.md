# Phase 76: projective foldのno-hit fast screen

- 実施日: 2026-09-14
- 基準HEAD: `cbbc5725e0d5dc45a8b71beb2401f8a3fd78ab80` (`dev/holonomic`)
判定: **no-hitのqf probe費用は減ったが、whole-epochで安定した勝利は確認できない。research gateのまま保持し、production defaultには昇格しない。**

## 目的と変更

Phase 75では7,216 physical rows中の`chart_p4` 3,376件がすべてno-hitであり、projective-fold候補判定のqf局所Newtonが無駄になっていた。Phase 76では、qf `P=P_t=0` probeの前に、chart-p4半径とD14 eventの近接性、およびreciprocal quarticの`Q_u(0)`を調べるfail-open screenを加えた。

[`projective_fold_screen.hpp`](../../src/lcbinint/magnification/holonomic/projective_fold_screen.hpp)は、chart-p4因子cubicの根をbinary64 outward-widened intervalで局所囲い込みし、D14 eventの`radius + radius_lo`と既存`radius_uncertainty`を使った幅で重なりを確認する。D14候補と重ならなければqf probeを省略する。重なる場合はreciprocal係数`p3=-Q_u(0)`と係数scaleのinterval上界を作り、既存qf contact tolerance `8192*FLT128_EPSILON`の外側にあるときだけrejectする。それ以外・演算不能・囲い込み不能はすべて既存qf probeへ渡す。screenはeventをaccept/promoteせず、chart event identity、cell plan、topologyは変更しない。

実行条件は`HOLO_ADAPTIVE_PROJECTIVE_P4_FOLD_RESEARCH`と`HOLO_ADAPTIVE_PROJECTIVE_P4_FAST_SCREEN`のcompile defineである。production router/defaultと通常buildは変更していない。

このscreenが使うp4/p3区間演算はbinary64の`nextafter`拡張である。一方、D14側の`radius_uncertainty`は既存metadataで、形式的なroot interval証明ではない。したがって全入力に対する数学的な片側証明とは主張しない。既存qf判定とのparity、known reference、true-fold controlに対するfail-openを実測した研究ゲートである。

## screen scan

110-case reference corpusはchart-p4 70件だった。従来qf probeは8件をaccept、62件を`no_coincident_d14_event`でrejectした。fast screenはその62件を先にreject（D14 overlapなし60、qf contact tolerance外2）し、従来acceptの8件はすべて`possible_fail_open`としてqf probeへ渡した。従来acceptを落とした件数は0。

Phase 75の凍結trajectory snapshotではchart-p4 3,376件すべて従来probeがrejectし、従来acceptは0件だった。fast screenはD14 overlapなし3,232件、qf contact tolerance外144件を先にrejectし、false rejectは0だった。

screen単体scanの合計時間は、referenceでscreen全件0.428 ms（早期reject分0.358 ms）、従来qf probe 0.975 ms。trajectoryではscreen全件18.450 ms、従来qf probe 42.159 msで、診断測定ではprobe相当費用を約56%削った。ただしscanはtopology生成外のmicrodiagnosticであり、screenとqfの順次計時でもある。速度判断には以下のwhole-epoch A/Bを使う。

## matched whole-epoch A/B

armは次の3つである。

- `baseline`: production path、projective-fold研究probeなし
- `qf`: projective-fold研究path、従来のqf probe
- `screen`: 同じ研究pathにfast screenを追加

Phase 75と同じ凍結7,216 physical rows（1,804 trajectory × 4 epoch、snapshot SHA-256 `6646492658ed416e75cb8c6c821ee853f79cdf7761fc41441186aa67b81fa05b`）を使い、arm/policyごとに3反復、各run 14,432結果行を作った。各runは`RelTol=1e-3`と`1e-4`を含む。GCC 11.5、`-O3 -DNDEBUG -std=gnu++17 -fext-numeric-literals -march=native -funroll-loops -ffp-contract=fast -fno-math-errno`、CPU affinity `taskset -c 0-7`、serial実行。arm順を反復ごとに回転した。

`full-cold`はD14/topologyとadaptive integrationを含む。`full-warm`はtrajectory内L2 D14 root warm-startを含み、topology reuseはOFF。`radial-only`はTopologyResult生成をtimer外に置いた。value-onlyは`GradientPolicy::None`、value+Jacは`ValueFirst` + analytic 5Jac。`mu_atol=1e-16`、値rtolは各target、Jac rtol `1e-3`でdefault grad atolを使った。LensParams生成は計時外。

表の各セルは`baseline / qf / screen`のp50 / p90 / p99 (ms)である。

| 経路 | RelTol | lane | baseline / qf / screen p50 | baseline / qf / screen p90 | baseline / qf / screen p99 |
|---|---:|---|---:|---:|---:|
| value-only | 1e-3 | full-cold | 0.4063 / 0.4140 / 0.4095 | 0.6993 / 0.7171 / 0.7086 | 3.1748 / 3.1894 / 3.1878 |
| value-only | 1e-3 | full-warm | 0.3338 / 0.3409 / 0.3379 | 0.6245 / 0.6381 / 0.6296 | 1.1452 / 1.1514 / 1.1275 |
| value-only | 1e-4 | full-cold | 0.4498 / 0.4559 / 0.4529 | 0.7667 / 0.7827 / 0.7755 | 3.1906 / 3.2206 / 3.1950 |
| value-only | 1e-4 | full-warm | 0.3830 / 0.3897 / 0.3868 | 0.6961 / 0.7115 / 0.7037 | 1.4043 / 1.4340 / 1.4165 |
| value+5Jac | 1e-3 | full-cold | 0.5745 / 0.5829 / 0.5871 | 1.0538 / 1.0749 / 1.0740 | 3.3648 / 3.3891 / 3.3678 |
| value+5Jac | 1e-3 | full-warm | 0.5063 / 0.5153 / 0.5196 | 0.9791 / 1.0010 / 0.9956 | 2.2763 / 2.2957 / 2.2896 |
| value+5Jac | 1e-4 | full-cold | 0.6095 / 0.6179 / 0.6238 | 1.2241 / 1.2483 / 1.2394 | 3.5338 / 3.5680 / 3.5340 |
| value+5Jac | 1e-4 | full-warm | 0.5422 / 0.5499 / 0.5553 | 1.1519 / 1.1806 / 1.1693 | 2.7240 / 2.7502 / 2.7272 |

radial-onlyも同じ`baseline / qf / screen`順で、p50 / p90 / p99は次のとおり。

| 経路 | RelTol | radial-only p50 | radial-only p90 | radial-only p99 |
|---|---:|---:|---:|---:|
| value-only | 1e-3 | 0.1211 / 0.1238 / 0.1232 | 0.2393 / 0.2570 / 0.2475 | 0.3784 / 0.4016 / 0.3861 |
| value-only | 1e-4 | 0.1569 / 0.1601 / 0.1589 | 0.3145 / 0.3327 / 0.3230 | 0.5434 / 0.5696 / 0.5559 |
| value+5Jac | 1e-3 | 0.2609 / 0.2673 / 0.2682 | 0.5753 / 0.6042 / 0.5937 | 1.3797 / 1.4293 / 1.4066 |
| value+5Jac | 1e-4 | 0.2911 / 0.2975 / 0.2991 | 0.7804 / 0.8094 / 0.7980 | 1.7223 / 1.7632 / 1.7458 |

per-row median timing-ratio（screen/qf）のp50 / p90 / p99は、value-onlyでcold/warmが`1e-3: 0.997/1.006/1.027, 0.995/1.007/1.040`、`1e-4: 0.997/1.006/1.022, 0.995/1.006/1.031`だった。中央値差は最大でも約0.5%の改善に留まり、p90/p99は非回帰でない。

value+5Jacではcold/warmのscreen/qf ratioが`1e-3: 1.007/1.024/1.049, 1.008/1.027/1.062`、`1e-4: 1.008/1.025/1.049, 1.008/1.029/1.065`だった。whole-epochで約0.7–0.8%遅く、tailも悪化した。screenはno-hit probe費用をmicrodiagnostic上で減らすが、そのままでは要求するwhole-epoch改善を満たさない。

全laneのp50/p90/p95/p99/max、arm別status・VBM reference誤差・行単位のparityは`summary.json`に格納した。

## correctnessとtrue-fold trajectory

whole corpusではqfとscreenのvalueが全lane・両targetでbitwise一致し、stop/status、value convergence、topology hash/event/cell数、node数の差は0。ValueFirstの5成分すべてでgradient値、quality、reasonの差も0。value-onlyは全armで両target 7,216/7,216収束。ValueFirstの`1e-4`は全arm 7,215/7,216で、残り1件も既存共通失敗。VBM `RelTol=1e-6` referenceに対する新規要求rtol超過は0で、既存の3件は各arm共通。

true-fold controlは`X=.12, Y=[-1e-6,0,+1e-6,0], rho=.02, q=.5, a=1`の48反復trajectory。各epochでchart-p4が2件あり、projective fold countは`0,2,0,2`と切り替わり、全armのtopology statusはOKだった。screenと従来qf候補は全lane・両targetでvalue、5Jac、quality、statusが一致した。

`u=0, Y=0`の`dmu/da`は、Phase 74で作った独立cold-topology value FD/GL reference `-2.4046891642370833`に対し、qf/screenともabsolute difference `1.3078441e-8`だった。baseline（projective fold mapなし）は差`2.2993991`で、既知のfold-map訂正をscreenが維持することを確認した。FD referenceは観測収束値であり、厳密な誤差包含ではない。

## 結論・残件

cheap screenは従来qf probeをknown corpus上で誤って早期rejectせず、true-fold controlも維持した。一方、whole-epochの速度はvalue-onlyでも小さく、p90/p99改善なし、value+5Jacでは回帰した。したがって**screenをproductionに昇格しない**。projective fold map自体のtrue-fold correctnessは維持する研究候補だが、通常経路へ常時追加する費用を正当化できていない。

次に性能目的で進める場合は、binary128 contact gateの前に行うD14 overlap screenをさらに安くするか、screenを呼ぶchart event候補の準備自体をtopology側で共有する必要がある。ただし、D14 uncertaintyが形式証明済みでない点を解消するまでは、どのscreenもfail-open research pathに限定する。

## 再現

```bash
bash evidence/holonomic/projective_fold_fast_screen_phase76_20260914/run_phase76.sh
cmake --build build-holonomic-m7 -j2
ctest --test-dir build-holonomic-m7 --output-on-failure
sha256sum -c evidence/holonomic/projective_fold_fast_screen_phase76_20260914/SHA256SUMS
```

raw 18 runs、targeted trajectory raw、scan TSV/log、machine-readable `summary.json`、compiler/input/affinity manifest、CTest log、checksumは[`evidence/holonomic/projective_fold_fast_screen_phase76_20260914/`](../../evidence/holonomic/projective_fold_fast_screen_phase76_20260914/)に保存した。
