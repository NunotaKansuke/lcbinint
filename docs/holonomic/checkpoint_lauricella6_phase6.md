# Phase 6 checkpoint: physical-fold regular packet

基準は `dev/holonomic @ 8960fb31001ddcacad1efcea5402048f80d7873b`
（Phase 5 `8960fb3`）とした。変更は isolated holonomic のPF6ヘッダ、専用
benchmark/test、`docs/holonomic`、`evidence/holonomic` に限定した。V2/V3
production router、既存V3 packet、16点K-rule経路は変更していない。

Phase 5でPF6 packet constructionの80.4%がfold近傍に集中し、whole epochの
最良値でもLD valueが約4.5 msだった。そこでPhase 6では、foldを通常の
Lauricella logarithmic stateで細分化する前に、physical half-fluxを直接輸送する
正則packetを実装した。目的の第一段階（fold起因のpacket生成と再帰の削減）は
達成したが、最終的なV2速度勝利条件は達成していないため、ここを追加micro
optimizationの停止checkpointとする。

## 実装

`src/lcbinint/magnification/holonomic/gm_lauricella6.hpp` に次を追加した。

- `P(t;R)` のroot pairを `t=m+sqrt(v)x` とし、偶奇成分
  `E(m,v;R)=0, O(m,v;R)=0` を明示的に解く。
- `m(R),v(R)` はこの2本の方程式のIFT係数再帰で生成する。`sqrt(v)`はstate
  にせず、packet係数は `v`の整数冪だけにした。
- V2のsmooth deflated integrand `H(t,R)`を `t=m`の周りでTaylor展開し、

  ```text
  J_half = (2/rho) v sum_n h_(2n)(R) v^n mu_(2n)
  mu_(2n) = pi Catalan(n) / (2*4^n)
  ```

  の固定偶数moment（n=0..4）でphysical flux Taylor packetを作る。fold packet
  自体にK-rule、quadrature、sample-fitはない。
- fold packetのTaylor tail、5次moment tail、root-pair residual、`v>0`を
  quality gateにし、失敗時はPF6の既存fail-closed経路へ戻す。physical re-seedを
  通常fallbackにはしていない。
- Dual5 laneでは同じ6状態相当のroot-pair/flux seriesからchain ruleで5Jacを
  作る。境界多項式はdouble laneと同じ固定次数構造を使い、FDはtest/reference
  だけに残した。
- routingはphysical-real fold eventをcell境界で確認し、sideの`v`とcenterの
  `v`、branch/chart certificateを使う。物理パラメータ値によるrouteはない。
  fold candidateは認証済みcell全体を一度だけ構築し、通ればそのcellのnodeを
  dense evaluateする。`GM6_FOLD_NODE_LIMIT`は候補数のA/B専用で、未設定時は
  cell全体を一候補にする。
- epoch costに `f0 arc geometry`、endpoint、arc preparation、PF6 cover、fold
  build/evaluation、fold pair/smooth/quality、未分類残差を追加した。既存の
  `connection_ms`はPF6 build/coverを含むため、未分類残差ではcoverを二重計上
  していない。

PF6 interiorは引き続き既存の6複素状態 logarithmic Pfaffian transportであり、
fold representationとのoverlapでは値・5Jac・独立referenceを照合する。

## fold候補数のA/B

入力は `evidence/holonomic/gm_coverage_cases.tsv` の108ケース（uniform 54、LD
54）、`n_r=64`、Release build。以下はOrder 20で、cold/warmの順に値だけ、次に
value+5JacのLD median msを示す。`full`は認証済みcell全体を一回で候補化した
設定である。

| fold候補の最大node数 | LD value cold/warm/V2 [ms] | LD value+5Jac cold/warm/V2 [ms] | PF6 constructions (value) | fold accepted/rejected (value) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4.949 / 4.805 / 1.192 | 5.632 / 5.450 / 1.296 | 6721 | 158 / 1 |
| 2 | 4.697 / 4.591 / 1.196 | 5.622 / 5.490 / 1.382 | 6205 | 158 / 1 |
| 4 | 4.560 / 4.358 / 1.200 | 5.232 / 4.959 / 1.378 | 5751 | 158 / 1 |
| full cell | 2.818 / 2.653 / 1.200 | 3.773 / 3.637 / 1.319 | 2831 | 115 / 44 |

小さい候補ではfold packet自体は通ってもPF6 coverが残り、wall timeの改善に
ならなかった。full-cell one-shotではfold accepted 115 arc相当がPF6 coverを
完全に置き換えた。quality rejectを再構築する指数的な候補木は廃止した。

## PF6 Order A/B

fold packetを含む同じhybrid routeで、PF6 interior Orderを比較した。cleanは値の
全108ケースで`status=OK`かつ全node true transportのケース数、rateは全nodeの
transport率である。低OrderはPF6 interiorまたはfold full-cell qualityの失敗を
減らせず、Order 20だけがPhase 5のclean/statusを維持した。

| PF6 Order | clean | transport rate | LD value cold/warm/V2 [ms] | PF6 constructions | fold accepted (value) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 85/108 | 0.884590 | 3.844 / 3.672 / 1.202 | 16147 | 63 |
| 10 | 93/108 | 0.929435 | 3.600 / 3.435 / 1.199 | 10716 | 81 |
| 12 | 95/108 | 0.977175 | 3.531 / 3.372 / 1.193 | 7077 | 94 |
| 14 | 100/108 | 0.992944 | 3.579 / 3.416 / 1.188 | 5673 | 100 |
| 16 | 103/108 | 0.998488 | 3.562 / 3.437 / 1.200 | 4433 | 106 |
| 20 | 104/108 | 1.000000 | 2.818 / 2.653 / 1.200 | 2831 | 115 |

同じOrder 20のfinal runでLD value+5Jacは`3.773/3.637/1.319 ms`だった。
Jac laneのcleanはLD 52/54で、残り2ケースのtopology uncertaintyはPhase 5から
変わらない。value/Jacとも全13888 nodeがtransportされ、fallback/reseedは0だった。

## final whole-epoch measurement

wall measurement（packet profile無効、fold候補は未設定でfull-cell one-shot）は次で
再現できる。

```text
env -u GM6_PACKET_PROFILE -u GM6_PACKET_PROFILE_PATH -u GM6_FOLD_NODE_LIMIT \
  ./build-holonomic-global/bench_gm_lauricella6 \
  evidence/holonomic/gm_coverage_cases.tsv 64 0 512 20 \
  > /tmp/phase6_final_unprofiled_angular.txt
```

結果は `cold / warm / V2` の順である。

```text
value LD                 2.818 / 2.653 / 1.200 ms
value+5Jac LD            3.773 / 3.637 / 1.319 ms
value clean              104/108, 104/108
Jac clean                52/54, 52/54
transported              13888/13888/13888/13888 (value cold/warm, Jac cold/warm)
transport rate           1.000000/1.000000/1.000000/1.000000
physical fallback/reseed 0/0
fold constructions       159/159 (value cold/warm), 159/159 (Jac cold/warm)
fold accepted/rejected   115/44 (value), 111/48 (Jac)
fold blocks/nodes        115/7360 (value), 111/7104 (Jac)
PF6 constructions       2831/2831 (value), 3002/3002 (Jac)
PF6 physical seeds      102/102 (value), 106/106 (Jac)
```

PF6 seed数がPhase 5の217から減っているのは、accepted fold cellではphysical
seedを使わず、equation-derived root-pair packetを直接構築したためである。残った
PF6側だけがphysical seedを使い、全てreferenceとして初期化された。physical
fallback/reseedは0である。

median cost（ms）は次の通りだった。

| lane | topology/D14 | arc | F0 arc | F0 endpoint | seed | PF6 connection | transport | fold build | fold eval | analytic Jac | unclassified |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| value cold | 0.851 | 0.014 | 0.868 | 0.041 | 0.056 | 0.027 | 0.029 | 0.643 | 0.005 | 0.000 | 0.053 |
| value warm | 0.702 | 0.013 | 0.864 | 0.041 | 0.057 | 0.028 | 0.029 | 0.632 | 0.005 | 0.000 | 0.049 |
| value+5Jac cold | 0.883 | 0.013 | 0.879 | 0.051 | 0.056 | 0.473 | 0.029 | 1.224 | 0.005 | 0.067 | 0.119 |
| value+5Jac warm | 0.701 | 0.013 | 0.870 | 0.051 | 0.057 | 0.437 | 0.029 | 1.187 | 0.005 | 0.068 | 0.126 |

`unclassified`は、topology、arc/F0、seed、PF6 connection、node evaluation、fold
buildを一度ずつ引いたepoch wallの残差である。従来の約1 msという見積もりは粗い
差し引きだったが、Phase 6ではこの残差も直接記録し、valueで約0.05 ms、Jacで
約0.12 msだった。value+5Jacの追加費用はfold Dual5 buildが主で、analytic
Jacobian algebraだけは約0.067 msである。

## construction profileと難例分類

profileの再現コマンドは次である。

```text
env GM6_PACKET_PROFILE=1 \
  GM6_PACKET_PROFILE_PATH=/tmp/gm6_phase6_packet_profile.tsv \
  ./build-holonomic-global/bench_gm_lauricella6 \
  evidence/holonomic/gm_coverage_cases.tsv 64 0 0 20 \
  > /tmp/gm6_phase6_profile.txt
```

保存した全ARC profileは
[gm_lauricella6_phase6_packet_profile.tsv](../../evidence/holonomic/gm_lauricella6_phase6_packet_profile.tsv)
にある。value coldのPF6だけの集計は以下の通りである。

```text
PF6 candidates/built       2831/2831
PF6 accepted/rejected      1354/1477
PF6 quality/build-failed   1477/0
PF6 max recursion depth    13
accepted node histogram    1:580 2:132 3:134 4:4 5:138 6:142 7:4
                            11:154 15:10 21:12 32:38 64:6
fold candidate/accepted    159/115
fold rejected              44 (all failure_reason=6 quality gate)
```

Phase 5の全arc近傍分類は、fold 149 arc / 5794 constructions、chart_p4 31 /
615、D14 soft 26/543、other 11/254だった。Phase 6のfold certificateが実際に
候補を作ったvalue-cold ARC分類は、fold 131/159 candidate constructions、chart_p4
14/159、D14 soft 10/159、other 4/159である。fold candidate reject 44件の内訳は
Taylor tail超過30、moment tail超過39、pair residual超過12（重複あり）だった。
最大PF6 constructionのarcは`resonant`, `arc_index=6`、nearest eventはphysical
fold、45 constructionsだった。chart_p4はphysical singularityとして扱わず、既存
chart/branch certificateを維持している。

## accuracy/reference

独立angular referenceをn_theta=512で最初の12行に適用し、全12/12がvalid、
value cold/warmとの最大relative errorは`1.285e-9`だった。`test_gm_lauricella6`
には次を含む。

- plan15、resonant、closeのfold packetを独立angular/referenceと比較。
- fold/PF6 overlapでvalueと5Jacを比較。
- fold analytic Jacをangular finite-differenceと監査（FDはtestのみ）。
- physical period相当のF0/F_halfからmuと5Jacまでのwhole epochを実行。

最終test出力はfold overlapのvalue relative error最大`3.0e-12`、fold analytic
Jacと独立Jacのrelative error最大`1.65e-10`、whole epochのvalue relative error
`4.84e-10`、gradient relative error`2.63e-9`で、`gm_lauricella6 PASS`だった。

## 判断

fold packetでPF6 constructionはPhase 5の7206から2831へ、58.0%削減した。fold
起因の再帰・packet生成も一候補化でき、value whole wallは約4.5 ms台から
`2.65--2.82 ms`へ下がった。これはfold正則化が効いた明確な結果である。

ただし、同等status/精度のvalue+5Jacは`3.64--3.77 ms`で、目標の2 ms未満にも、
V2の約1.2--1.3 msにも届かない。valueでもtopology/D14約0.85 ms、F0 arc geometry
約0.87 ms、fold series約0.64 msが残る。fold packetを入れてもwhole wallが2.5--3
ms帯から下がらないため、追加のPF6 micro optimizationやthreshold緩和はここで
停止する。新しいfold-adapted basisまたはD14/F0共有キャッシュを設計する場合は、
別Phaseの数学的checkpointとして扱う。

## 検証

```text
cmake --build build-holonomic-global \
  --target test_gm_lauricella6 bench_gm_lauricella6 -j2
./build-holonomic-global/test_gm_lauricella6
ctest --test-dir build-holonomic-global --output-on-failure
```

`test_gm_lauricella6`はPASS。全体のctestも最終build後に
`100% tests passed, 0 tests failed out of 11`となった。
