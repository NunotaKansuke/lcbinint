# Phase 5 checkpoint: Lauricella PF6 packet construction

基準は `dev/holonomic @ 781e0501754de8b362bb35a279200875ce6dbaf9`。変更範囲は
`src/lcbinint/magnification/holonomic`、PF6専用ベンチ、`docs/holonomic`、
`evidence/holonomic` に限定した。V2/V3 production routerと既存のV3 packetは
変更していない。

今回の判断は、Taylor transportそのものではなく、whole epochで支配的だった
packet constructionを削ることだった。最終的に、Order-20のtrue PF6はPhase 4の
connection construction 8030回から7206回、median LD value wall timeは
`9.196/9.059 ms`から`4.544/4.370 ms`（cold/warm）まで下がった。しかしV2の
`1.144 ms`を上回らず、同じ精度/statusを保ったままproduction採用できる性能には
届かなかった。この時点で細かなmicro optimizationは止めた。

## 実装した変更

`gm_lauricella6.hpp`のdouble Taylor laneに、次の固定構造最適化を入れた。

- 境界二次式の `R=R0+h` jetを専用生成し、`P(t)`の実際の次数6を
  `root_series`へ渡した。
- `R0+h`との積、affine積、低次数多項式積を固定化し、境界 `P` とgeometryの
  不要な高次数zero workを削った。
- double seriesのscalar/complex除算を、inverse生成後の再乗算ではなく直接係数
  recurrenceにした。
- log derivativeで定数係数の逆数を一度だけ作り、同一packet内の重複した除算を
  削った。
- 既存の安全なmidpoint recursive bisectionに、実際のGauss radial nodeを含まない
  子区間を構築しない処理を加えた。受理packetはdense outputで複数nodeを評価する。
- packet construction内を endpoint/root、algebraic geometry/cross-ratio、
  log derivative、Pfaffian recurrence、quality/tailに分解して計測した。
  accepted/rejected、depth、accepted packetのnode数、arc単位のconstructionと
  event/divisor情報も記録する。
- benchmarkにOrder `8/10/12/14/16/20`切替、cold/warm、value-only、analytic
  value+5Jac、独立angular referenceを追加した。引数の `angular_n=0` は本当に
  referenceを無効化するよう修正した。

係数とtransportは引き続き6複素状態の明示的 logarithmic Pfaffian系であり、
K-rule/Chebyshev sample-fit packet、15x15再還元、巨大scalar PF展開は使っていない。
physical seedの積分はpacket係数のfitではなく、各arcの初期値/reference生成である。
精度promotionはdouble seedが失敗した場合だけlong doubleへ進むが、今回の全runで
promotion、fallback、physical re-seedは0だった。

## packet construction profile

次のコマンドで、n_r=64、108 cases、LD 54 casesを単独processで測定した。

```text
cmake --build build-holonomic-global --target test_gm_lauricella6 bench_gm_lauricella6 -j4
env -u GM6_PACKET_GATE GM6_PACKET_PROFILE=1 \
  GM6_PACKET_PROFILE_PATH=/tmp/gm6_phase5_final3_profile.tsv \
  ./build-holonomic-global/bench_gm_lauricella6 \
  evidence/holonomic/gm_coverage_cases.tsv 64 0 0 20 \
  > /tmp/gm6_phase5_final3.txt
```

`value_cold`の集計は次の通りで、`ARC`行の全217 arc分の値は
[gm_lauricella6_phase5_packet_profile.tsv](../../evidence/holonomic/gm_lauricella6_phase5_packet_profile.tsv)
に保存した。

```text
candidates=7206 built=7206 accepted=3405 rejected=3801
quality_rejected=3801 build_failed=0 max_depth=13
endpoint/root ms       25.072950
algebraic geometry ms  51.472351
log derivative ms      38.795602
Pfaffian recurrence ms 43.486680
quality/tail ms          6.650025
accepted candidate ms  78.704940
rejected candidate ms  88.838784
```

stage値は54 LD cases全体のconstruction計測の合計であり、whole-epoch wall timeでは
ない。rejected candidateが全候補の52.7%を占め、rejectしてから子を再構築する現在の
方式が残存費用の中心である。construction数のarc分布は `min/median/mean/p90/max =
1/42/33.2074/43/45`。depth histogramは次の通りだった。

```text
0:217 1:422 2:716 3:690 4:668 5:668 6:660 7:648
8:640 9:636 10:628 11:312 12:280 13:21
```

accepted blockあたりのnode数は次の通りだった。

```text
1:1590 2:334 3:336 4:4 5:340 6:345 7:4
11:358 15:11 21:13 32:64 64:6
```

event分類は、最近傍eventの種別を `1=physical fold`、`2=chart_p4`、
`3=physical complex/D14 soft`、`4=other`としている。

| 最近傍分類 | arc数 | construction数 | accepted | rejected |
| --- | ---: | ---: | ---: | ---: |
| physical fold | 149 | 5794 | 2711 | 3083 |
| chart_p4 | 31 | 615 | 308 | 307 |
| D14 soft | 26 | 543 | 262 | 281 |
| other | 11 | 254 | 124 | 130 |

fold近傍はarcの68.7%、constructionの80.4%を占めた。最大45 constructionのarcは
`resonant cell=9`、`close-binary cell=11/12`、`caustic-cross cell=7`などで、
foldまたはD14 softに近かった。したがってfoldが主要因という条件は実測で満たした。
`xi_i=0,1`または`xi_i=xi_j`までの最小距離が`1e-8`未満のarcは37本で、内訳は
fold 31本、chart_p4 6本、D14 soft 0本だった。最小値は`1.782e-11`（`rand007`
のfold近傍arc）であり、xi divisorも独立のconditioning指標としてprofileに残した。

## A/B結果

同じcase set、n_r=64、同じdouble精度とstatus gateでOrderを比較した。値はLD
value-onlyのmedian cold/warm/V2、coverageはclean case数と全node transport率である。

| Order | clean cases | transport rate | LD value ms cold/warm/V2 | constructions |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 84/108 | 0.883356 | 4.162 / 3.988 / 1.139 | 22126 |
| 10 | 93/108 | 0.929435 | 4.151 / 3.956 / 1.143 | 16653 |
| 12 | 95/108 | 0.977175 | 4.076 / 3.754 / 1.145 | 12806 |
| 14 | 100/108 | 0.992944 | 4.130 / 3.949 / 1.146 | 10639 |
| 16 | 103/108 | 0.998488 | 4.272 / 4.004 / 1.161 | 8619 |
| 20 | 104/108 | 1.000000 | 4.544 / 4.370 / 1.144 | 7206 |

Order-16は`very-close`でtransport tolerance failureを出し、`rand008`と
`rand031`はtopology uncertaintyを維持した。Order-20でもこの2ケースは同じ
topology statusであり、成功に数えていない。低次数はconstruction数を減らしても
tail reject、子packet、または失敗処理が増え、総wall timeの勝利にならなかった。

### recenter方式

一度作ったpacketから半径を推定し、範囲内のnodeをまとめてdense outputする方式を
試した。しかし、親packetの局所tailから子centerで安全なradiusを保証できず、
親のcertified range外に出る候補で誤値またはcoverage failureが発生した。片側edge
recenterも同じ問題を解消しなかったため、未証明のradius reuseは戻し、安全なmidpoint
bisectionを残した。現行のnode-aware skipはnodeを含まない子区間だけを削るため、
coverageを変えずに8030から7206へ減らしている。

### projected quality gate

全6状態のOrder-N/Order-(N-2) strict tailの代わりに、physical Fと5個の
`dF/dxi`を投影するgateを試し、Pfaffian residual、divisor distance、branchの内部
certificateを併用した。結果は `7194 candidates / 3398 accepted / 3796 rejected`
で、strictの7206から12回減っただけだった。quality計測はstrictの約6.65 msから
約25.90 msへ増え、LD value wallもstrictの約4.5 ms台より遅くなった。独立reference
で速度だけを作るgateではないため、本線には採用していない。

### fixed recurrence/cacheの試行

Pfaffianの状態組合せをcacheする6x6固定行列版は、recurrence集計を約43.6 msから
約47.8 msへ悪化させた。xi inverseの共通分母化はalgebraic stageを約51 msから
約64 msへ、共役根を使う別のgeometry rewriteは約53 msから約57 msへ悪化させた。
いずれもrevertした。採用した低次数geometry、direct division、root degree固定は
accuracy/statusを保った上でconnection medianをPhase 4の約7.03 msから約2.18 ms
へ下げた。

## whole epochと独立reference

profileを無効化した本測定は次のコマンドで再現できる。

```text
env -u GM6_PACKET_GATE -u GM6_PACKET_PROFILE -u GM6_PACKET_PROFILE_PATH \
  ./build-holonomic-global/bench_gm_lauricella6 \
  evidence/holonomic/gm_coverage_cases.tsv 64 0 0 20 \
  > /tmp/gm6_phase5_final3_order20.txt
```

結果は以下の通りである。cold/warmは毎回D14/topology/physical seedを生成する
epochと、前runのD14 root setだけをwarm seedに渡すepochを分けた。cell/chart/period
stateは安全な対応付けとbranch certificateがないため再利用していない。このため
warm値はD14-only warmであり、steady-state state reuseの証拠ではない。

```text
clean value cases       104/108 cold, 104/108 warm
value transported       13888/13888 cold, 13888/13888 warm
value rate               1.000000/1.000000
value-only LD ms         4.544 / 4.370 / 1.144  (PF6 cold/warm/V2)
value+5Jac LD ms         4.631 / 4.487 / 1.316  (PF6 cold/warm/V2)
analytic Jacobian cost   0.166 / 0.160 ms       (cold/warm median)
Jac marginal wall        0.172 / 0.203 ms       (cold/warm median)
connections              7206 / 7206
physical seeds           217 / 217
fallback/reseed          0 / 0
```

whole epochのcost内訳（median ms）は、value coldで
`topology/D14=0.841, arc=0.014, geometry=0.001, seed=0.170, connection=2.179,
transport=0.069`、value warmで
`0.683, 0.013, 0.001, 0.168, 2.174, 0.069`だった。Jac coldのanalytic Jacobian
追加は`0.166 ms`で、Phase 5の主因ではない。

independent angular referenceは次で最初の12行だけを512点で監査した。

```text
env -u GM6_PACKET_GATE -u GM6_PACKET_PROFILE \
  ./build-holonomic-global/bench_gm_lauricella6 \
  evidence/holonomic/gm_coverage_cases.tsv 64 0 512 20 \
  > /tmp/gm6_phase5_final3_angular.txt
```

全12行でreferenceはvalid、PF6 cold/warmとの最大relative errorは
`1.716e-09`だった。短いlocal physical seed/analytic Jac probeは既存の
`test_gm_lauricella6`に含まれ、plan15 end-to-endでvalue relative error
`5.442e-10`、gradient relative error `2.880e-09`を再確認した。physical seedは
217 arcすべてで生成され、全PF6 nodeでfallback/re-seedなしに評価された。

## fold germと停止判断

fold分類が80.4%のconstructionを占めたため、Phase 4の`fold_germ`をsqrt(v) packetへ
拡張する条件自体は満たした。ただし既存の6D logarithmic connectionではfoldで
`xi_0-xi_1 -> 0`となり、各log derivativeが単純なR Taylorの正則係数にならない。
現存の`fold_germ`は正則な一点のFD5 stateと有限な`K0`を検査するhelperであり、一般
packetのseedとして渡すだけではPfaffian residueを消さない。sqrt(v)を本線にするには
root pairの対称/反対称成分を分け、接続自体をその座標で変換し、branchとphysical
fluxを別に証明する必要がある。未検証の一点germを通常6D packetへ混ぜる実装は
fail-closedを壊すため、今回は採用せず、失敗理由をここに固定した。

現行実装でconnectionを完全に削除しても、value coldのwallからの差し引きは
`4.544 - 2.179 ~= 2.365 ms`であり、これはtopology/seed/transportだけの計測和
`0.841+0.014+0.001+0.170+0.069 ~= 1.095 ms`に未分類のepoch overheadが加わった
値である。V2の`1.144 ms`との差はこの周辺固定費だけでも残る。さらにfold germを
導入するには新しい正則化basisが必要で、今回の6D fixed-structure optimizationの
範囲を越える。したがって、現行basisでpacketを細かく削り続けてもV2勝利は見込めず、
Phase 5はここで停止する。

## 検証

```text
./build-holonomic-global/test_gm_lauricella6
gm_lauricella6 PASS failures=0

ctest --test-dir build-holonomic-global --output-on-failure
100% tests passed, 0 tests failed out of 11
```

`GM6_PACKET_GATE=projected`は診断用の実験フラグで、未設定時のstrict all-state
gateが通常動作である。V2/V3 production routerへの接続は行っていない。
