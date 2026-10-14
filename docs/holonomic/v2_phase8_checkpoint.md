# V2 Phase 8 checkpoint

日付: 2026-09-10
基準: `dev/holonomic` `62c28ec047b87f9d382657cc5119b7764005ab78`
対象: `src/lcbinint/magnification/holonomic/`, `tests/holonomic_cpp/`,
`benchmarks/holonomic/`, `evidence/holonomic/`, `docs/holonomic/`
測定条件: Release build、double V2、`n_r=64`、同一case set、同一ホスト、
cold/warm分離。

## 判定

V2の数値保証とfail-closedを維持したまま、現行のproduction routerを変更せずに
profileと研究用A/B経路を追加した。production defaultとして採用したのは、
structured D14と厳密な`chart_p4` factor pathのときに未使用のgeneric
`PolyFamilyR`を構築しない固定構造化だけである。これは式、root solver、certificate、
routerを変えない安全な固定費削減である。

reciprocal chart、Horner D14、lifted Newton、matrix-polynomial/QZ、三段K gateは
全てA/Bまたはresearch laneに留めた。reciprocal chartは難しいt-chart arcの
angular rescueを減らすが、whole epochの改善は15%に届かず、cold/warmの既定経路へ
昇格させる条件を満たさない。D14の新方式もend-to-endで現行経路を上回らない。
従ってV2/V3 production routerは従来のままである。

既存の未追跡`.claude/`には触れていない。

## 本番経路のprofile

`bench_v2_profile`はshared geometry harnessではなく、D14からcell classification、
radial node、root-pair、F0、K-ruleまでの本番epoch経路を呼ぶ。value-onlyは
timing boundaryを明確にするため`epoch_value(_prepared)`を直接測定し、
value+5Jacは既存のfinite-source Jacobian経路を測定する。finite-source routerの
実装自体は変更していない。

108ケース、cold、profile 1回の集計は次の通りである。各timerは呼び出し階層を
含むため、単純加算してwall timeと解釈しない。詳細なraw counterは
`v2_phase8_profile_direct_default.txt`に保存した。

| lane | node | D14 calls | D14 pre/DD/qf sweeps | topology cells | quartic warm/cold | root-pair warm/cold | K attempts/success/reject | angular rescue calls/nodes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cold value | 25,856 | 108 | 21,600 / 1,696 / 10 | 1,370 | 8,838 / 404 | 16,614 / 934 | 9,836 / 9,278 / 558 | 4,674 / 299,136 |
| cold value+5Jac | 25,856 | 108 | 21,600 / 1,696 / 10 | 1,370 | 8,838 / 404 | 16,614 / 934 | 19,672 / 18,556 / 1,116 | 9,348 / 598,272 |
| warm value steady | 68,352 | 324 | 19,440 / 4,830 / 14 | 4,000 | 18,692 / 1,068 | 48,592 / 2,422 | 27,157 / 26,112 / 1,045 | 8,512 / 544,768 |
| warm value+5Jac steady | 68,352 | 324 | 19,440 / 4,830 / 14 | 4,000 | 18,692 / 1,068 | 48,592 / 2,422 | 54,314 / 52,224 / 2,090 | 17,024 / 1,089,536 |

主要なwall固定費はD14 solve、topology/grid、arc geometryである。cold valueの
profile timerではD14 solve 54.41 ms、topology 81.66 ms、grid 20.11 ms、arc
19.75 ms、root-pair 5.56 ms、endpoint 8.37 ms、K 1.58 ms、angular rescue
5.33 ms（108 epochの集計）だった。warm value steadyではD14 solve 121.73 ms、
topology 208.16 ms、grid 65.73 ms、arc 63.06 ms、root-pair 33.67 ms、endpoint
19.12 ms、K 4.34 ms、rescue 9.82 ms（324 measured epochの集計）だった。

D14の既定経路では、global diagnosticのroot count/conjugacy/Vieta failureは
0件で、近接root clusterはcold 52、warm 106件だった。warm D14 seedは324/324
で利用された。topologyはcoldで512-grid 1,370回、3072-grid 138回、4096-grid
0回を含み、escalation 46、uncertain 4だった。warmでは512/3072/4096-gridが
4,000/528/0回、escalation 176、uncertain 10である。

K-rule reject理由は、profileした既定経路では全て8対16 node disagreementだった。
nonfinite、VFloor、TMax、S2、Bの直接rejectは0件である。arc側の内訳はcold valueで
order 1,536、TMax 2,256、VFloor 324、value+5Jacでは各々3,072、4,512、648
だった。reject率をゼロにすることは目的にせず、既存の64-node angular rescueを
正しいreference fallbackとして維持した。

## whole-epoch V2

次の値は`bench_v2_whole`のunprofiled wall timeで、108ケースはreps=8、
reference setは115ケースでreps=4、warmはepoch 0を除外している。

| lane | cases | p50 | p90 | p95 | p99 | max | status |
|---|---:|---:|---:|---:|---:|---:|---:|
| cold value, default | 108 | 0.880494 | 1.539505 | 1.684426 | 1.815794 | 1.825471 | 104/108 |
| cold value+5Jac, default | 108 | 1.038950 | 1.838637 | 1.944592 | 2.136173 | 2.140942 | 104/108 |
| warm value, default | 756 | 0.800960 | 1.340563 | 1.474118 | 1.908136 | 2.680696 | 728/756 |
| warm value+5Jac, default | 756 | 0.893731 | 1.534682 | 1.806424 | 2.467865 | 3.294787 | 728/756 |
| cold value, reciprocal A/B | 108 | 0.876877 | 1.437368 | 1.529694 | 1.604207 | 1.699727 | 104/108 |
| cold value+5Jac, reciprocal A/B | 108 | 1.018166 | 1.633586 | 1.690866 | 1.964545 | 1.985342 | 104/108 |
| warm value, reciprocal A/B | 756 | 0.796431 | 1.262284 | 1.422731 | 1.744757 | 2.634405 | 728/756 |
| warm value+5Jac, reciprocal A/B | 756 | 0.864643 | 1.414569 | 1.600454 | 2.405159 | 3.231893 | 728/756 |

reciprocal retryの中央値改善率はcold value 0.41%、cold value+5Jac 2.00%、
warm value 0.57%、warm value+5Jac 3.25%だった。p90/p99でも非回帰だが、目標の
15%に達しないため既定値にはしない。115-case referenceでもstatusは111/115で
一致し、defaultとreciprocalの最大値差はmuで2.044e-13、gradient相対差の最大は
1.218e-4（cold）/1.379e-3（warm）だった。

既定t-chartのK rescueに対し、reciprocal A/Bはcold valueで4,674回試行して
3,697回成功、残り977回をangular rescueへ送り、value+5Jacでは9,348回中
7,394回成功した。warm valueは8,512回中6,958回、warm value+5Jacは17,024回中
13,916回成功した。これはexact coefficient reversal、root mapping、5方向微分
変換を使うが、theta=0のもう一つのpoleやarc boundaryをfail-closedで拒否する。
root-pair全体をu-chart状態へ切り替える経路は、TMax拒否がcold 14/warm 42と
K arcに比べて少なく、今回のwall勝利に寄与しないため未採用である。

## D14 A/B

全候補はlocal convergenceだけで成功扱いにせず、finite root count、conjugacy、
Vieta、structured residual、coefficient reconstruction、root-set certificateを
通してから既存経路へ渡した。certificate failureは既存DD/qf経路へ戻るか、
fail-closedである。

同一108ケース・reps=4でのD14 A/B wall p50は次の通りである（matrixだけは
isolated solveの測定）。

| 構成 | cold value | cold value+5Jac | warm value | warm value+5Jac | 観測 | 判定 |
|---|---:|---:|---:|---:|---|---|
| current structured DD/Aberth | 0.877011 | 1.037701 | 0.795268 | 0.921235 | status 104/108、warm 308/324 | baseline |
| Horner/structured hybrid | 1.053063 | 1.175896 | 1.002188 | 1.064830 | warm mu差最大7.965e-11、gradient差最大7.525e-3 | revert |
| lifted cubic Newton | 0.994296 | 1.069842 | 0.888163 | 0.985297 | certificate pass 70/108 cold、544/756 warm、残りはfail | revert |
| matrix polynomial seed/QZ research | external QZ約0.054 ms/solve | external QZ約0.054 ms/solve | 同左 | 同左 | exact identity failure 0/108、finite root count 14 | production未採用 |
| presearch max=120 | 0.836804 | 0.989102 | 0.804627 | 0.915728 | coldだけ数%改善、warm非勝利 | default未変更 |

Horner候補はrootごとにcheap boundを作ったが、Aberth interactionを含む安全な
correction boundを満たすrootが少なく、構造評価へ戻る費用が支配的だった。候補の
実装にはcoefficient conversion、Horner/derivative、S、B形成の誤差項を含めたが、
全体で遅くなった。hybridはさらにrootごとの構造評価とallocationが支配的になった。

lifted Newtonは三次判別式の恒等式自体は使えるが、local Newton収束後にも
conjugacy/Vieta/reconstructionのglobal certificateで落ちる例が残った。coldの
failure内訳はnewton 16、conjugacy 10、Vieta 12、warmはnewton 102、scalar/判別
条件36、conjugacy 38、Vieta 36である。従ってfalse successを避けるため既定には
入れない。

matrix routeは`4 A E - B^2 = 3 D14/4096`のidentityを108ケースで検査し、
Python/SciPy QZのisolated research benchmarkもfinite root 14を確認した。ただし
QZ依存をproductionへ追加するwall根拠がなく、warmでは既存root seedがあるため、
matrix seedは凍結した。

## K gateとchartの安全性

24-nodeの三段gateは8対16 disagreementをcoldの1,116件から676件へ減らしたが、
whole wallの改善は再現せず、max/referenceも僅かに悪化した。常時32 nodeへ行く
変更は採用していない。

reciprocal primitiveは
`u=-1/t`、`u^4 P(-1/u)=p4-p3 u+p2 u^2-p1 u^3+p0 u^4`を使う。coefficients、
parameter derivatives、mapped root set、V2 angular referenceを検査した。
`test_reciprocal_chart`は108ケースでroot mapping差3.884e-16以下、derivative
identity差2.556e-16以下でPASSした。chart switch後もF0、Fhalf、value、5Jacの
独立reference比較でstatus非回帰だった。production defaultは`t` chartのままで、
`HOLO_RECIPROCAL_CHART=1`だけがこのretryを有効にする。

## 検証と再現コマンド

```sh
cmake -S tests/holonomic_cpp -B build-holonomic-m7 -DCMAKE_BUILD_TYPE=Release
cmake --build build-holonomic-m7 -j2
ctest --test-dir build-holonomic-m7 --output-on-failure

./build-holonomic-m7/bench_v2_profile \
  evidence/holonomic/gm_coverage_cases.tsv 1 4
HOLO_RECIPROCAL_CHART=1 ./build-holonomic-m7/bench_v2_profile \
  evidence/holonomic/gm_coverage_cases.tsv 1 4
./build-holonomic-m7/bench_v2_whole \
  evidence/holonomic/gm_coverage_cases.tsv 8 8
HOLO_RECIPROCAL_CHART=1 ./build-holonomic-m7/bench_v2_whole \
  evidence/holonomic/gm_coverage_cases.tsv 8 8
./build-holonomic-m7/bench_v2_whole \
  evidence/holonomic/v2_phase8_cases_with_reference.tsv 4 4
HOLO_RECIPROCAL_CHART=1 ./build-holonomic-m7/bench_v2_whole \
  evidence/holonomic/v2_phase8_cases_with_reference.tsv 4 4
```

D14 A/Bは`HOLO_D14_METHOD=horner|lifted|hybrid`、presearchは
`HOLO_D14_PRESEARCH_MAX=120`、三段K gateは`HOLO_K_GATE=strict`で再現できる。
これらは全てresearch laneであり、通常の環境では未設定でcurrent V2を使う。
matrix identity/QZは`bench_d14_matrix`と
`benchmarks/holonomic/v2_phase8_d14_matrix_qz.py`で再現できる。

全13 CTestがPASSした。chart_p4 legacy parity、strict gate、reciprocal V2 test、
independent angular/physical reference比較もraw evidenceに保存した。

## 残る最大costと次の判断

現時点のwhole epochではD14 solveとtopology/gridが最大costで、K-rule rejectを
さらに減らしてもangular rescue自体が小さいため、15%条件を満たす見込みはない。
新しいD14数学やthreshold緩和を追加せず、今回のD14/reciprocal研究経路はこの
checkpointで凍結する。将来続ける場合は、certified warm topology/root stateの
再利用、grid probeとD14のquantity共有、または全く別のD14 algorithmを、同じglobal
certificateとwhole-epoch基準で検討する。

### Evidence

主なraw outputは次の通り。

- `evidence/holonomic/v2_phase8_profile_direct_default.txt`
- `evidence/holonomic/v2_phase8_profile_direct_reciprocal.txt`
- `evidence/holonomic/v2_phase8_whole_direct_default.txt`
- `evidence/holonomic/v2_phase8_whole_direct_reciprocal.txt`
- `evidence/holonomic/v2_phase8_whole_direct_reference_default.txt`
- `evidence/holonomic/v2_phase8_whole_direct_reference_reciprocal.txt`
- `evidence/holonomic/v2_phase8_profile_d14_horner.txt`
- `evidence/holonomic/v2_phase8_profile_d14_lifted.txt`
- `evidence/holonomic/v2_phase8_profile_d14_hybrid.txt`
- `evidence/holonomic/v2_phase8_d14_matrix_qz.txt`
- `evidence/holonomic/v2_phase8_reciprocal_chart.txt`
- `evidence/holonomic/v2_phase8_summary.tsv`
