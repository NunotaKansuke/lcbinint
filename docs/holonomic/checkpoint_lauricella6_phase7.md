# Lauricella6 Phase 7 checkpoint: shared-geometry LD evaluator A/B

基準は `dev/holonomic @ 781e050` からPhase 6を進めた現在のHEAD
（実装時の基準 `f71d85e`）とした。変更は isolated holonomic のPhase 7
harness、専用test、`docs/holonomic`、`evidence/holonomic` に限定した。
V2/V3 production router、既存PF6 epoch経路、16点K-ruleのproduction wiringは
変更していない。

今回の目的は新しいholonomic数学を追加することではなく、V2とPF6/fold hybridが
同じgeometryを使ったときのLD evaluatorだけを比較することだった。各laneで一度
だけV2の `classify_cells`/D14、cell traversal、quartic/root-pair continuation、
endpoint polish、F0 accumulationを作り、その不変なplanをV2 K-rule evaluatorと
PF6/fold evaluatorへ渡した。PF6側は `classify_cells`、`arc_intervals`、endpoint
探索、F0 arc geometryを呼び直さない。既存のpolished pairはchart/branch
certificateに使い、PF6 Taylor packetのroot continuationをそのpairと照合する。

## 実装

追加した
[`phase7_shared_harness.hpp`](../../src/lcbinint/magnification/holonomic/phase7_shared_harness.hpp)
は、次の二段構成を持つ。

- `build_shared_plan`がV2のtopology/D14結果、cell境界、Chebyshev radial node、
  quartic/root-pair追跡、polished endpoint、`F0`と`dF0`を保存する。
- `evaluate_v2_k_rule`は保存済みの `(m,v)`、quartic、endpointだけを使って既存の
  16-node K-ruleを評価する。既存K-ruleの8-node quality gateが失敗したarcは、
  V2のfail-closed angular rescueへ送る。このため `V2_k_failed` と
  `V2_angular_fallback` は同数で、raw K-rule適用率とV2 status維持を分けて読める。
- `evaluate_pf6_fold`は同じnode/root-pair/cell planを受け、physical-fold cellでは
  Phase 6のequation-derived fold packet、それ以外では既存PF6 Taylor packetを
  構築する。PF6の初期physical seedはpacket初期化用であり、nodeごとのphysical
  re-seed fallbackではない。失敗時はfail-closedでstatusを落とす。
- valueとanalytic 5Jacは別laneで計測し、Jacobianは同じPF6 stateからchain ruleで
  取り出す。5Jac pathの追加費用を`analytic_jacobian_ms`とwhole-epoch差分に
  分離した。

追加した
[`test_gm_phase7_shared.cpp`](../../tests/holonomic_cpp/test_gm_phase7_shared.cpp)
は、cleanな`cusp-approach`でcold/warm、value/Jac、shared F0再利用、既存V2
K-rule routeとのparityを検査する。

## 再現コマンド

```text
cmake -S tests/holonomic_cpp -B build-holonomic-m7 \
  -DCMAKE_BUILD_TYPE=Release -DLCBININT_BUILD_HOLONOMIC_M7_PY=OFF
cmake --build build-holonomic-m7 -j2
ctest --test-dir build-holonomic-m7 --output-on-failure

./build-holonomic-m7/bench_gm_phase7_shared \
  evidence/holonomic/gm_coverage_cases.tsv 64 108 108 20 2048 \
  > evidence/holonomic/gm_lauricella6_phase7_shared_order20.txt
```

benchmark引数は順に`cases TSV`、`n_r`、ケース数、angular referenceを適用する
先頭行数、PF6 Order、reference angular node数である。したがって入力は108
ケース（uniform 54、limb-darkening 54）、`n_r=64`、LD 54ケース、Order 20、
独立angular reference 2048 nodeとなる。raw per-case outputとaggregateは
[`gm_lauricella6_phase7_shared_order20.txt`](../../evidence/holonomic/gm_lauricella6_phase7_shared_order20.txt)
に保存した。

## 実測

以下はRelease、同一プロセス、LD 54ケースのmedian ms。`shared_total`は同じplan
のtopology/root/F0を含み、V2とPF6で共通に加算される。`V2_K_LD`と`PF6_fold_LD`
はLD evaluatorだけの時間である。

| lane | shared topology/D14 | shared root tracking | shared F0/endpoint | shared total | V2 K LD | PF6/fold LD | whole V2 | whole PF6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| value cold | 0.883 | 0.323 | 0.047 | 1.393 | 0.075 | 0.932 | 1.421 | 2.493 |
| value warm | 0.713 | 0.338 | 0.047 | 1.235 | 0.075 | 0.936 | 1.315 | 2.347 |
| value+5Jac cold | 0.893 | 0.331 | 0.056 | 1.401 | 0.112 | 2.081 | 1.483 | 3.684 |
| value+5Jac warm | 0.727 | 0.353 | 0.056 | 1.244 | 0.112 | 2.073 | 1.357 | 3.420 |

PF6側の内訳は、value coldでfold packet build `0.648 ms`、fold evaluation
`0.009 ms`、PF6 seed/packet/evaluationは中央値がそれぞれ`0/0/0 ms`（foldを
使うcaseが中央値を占める）だった。Jac coldではfold build `1.334 ms`、fold
evaluation `0.016 ms`、physical seed `0.057 ms`、PF6 packet `0.170 ms`、PF6
node evaluation `0.089 ms`、analytic Jacobianのnode marginal `0.067 ms`だった。
whole時間にplanを共通加算しても、LD evaluatorの差がそのまま残っている。

### nodeとstatus

- LD nodeは全`13888` nodeをshared planから両方式へ渡した。
- PF6/foldは`13888/13888` nodeをtransportし、failed nodeは0、angular/reseed
  fallbackは0だった。clean statusは`52/54`で、残る2ケースは共有topologyの
  `TOPOLOGY_UNCERTAIN`である。
- fold packet候補はvalue/Jacとも184。valueは122 accepted/62 rejected、Jacは
  118 accepted/66 rejectedだった。
- PF6 packet constructionはvalue 2559、Jac 2730。valueのaccepted/rejectedは
  1223/1425、Jacは1303/1520である。
- V2 raw K-ruleは`9214/13888` nodeがsuccess、`4674` nodeがquality rejectと
  なり、同数を既存64-node angular rescueへ送った。これはV2のstatus/値を保つ
  ための既存経路で、PF6のreseedではない。従って「K-ruleだけでfallbackなし」の
  strict all-trueは`26/54`だが、V2 evaluatorのstatusはclean subsetで維持される。
- PF6のstrict all-trueは`52/54`で、topology uncertainty以外のclean caseでは
  全node true transportとなった。

### accuracy/reference

独立angular referenceは54/54で計算できた。topologyがcleanな52ケースだけを
数値比較の対象にすると、value coldのV2/PF6 reference相対誤差の最大はそれぞれ
`1.91e-8`/`3.25e-9`だった。shared V2とPF6/foldのvalue差は最大`2.087e-8`、
Jacのmu差も`2.087e-8`、5Jac component差の最大は`1.912e-3`だった。
uncertain topologyのangular numberはstatus検査には残すが、accuracyのclean
分母には入れていない。

## 判断

このA/BではPF6/foldがshared geometry上でもV2 K-ruleより遅かった。valueの
pure LD evaluatorは`0.932/0.075 = 12.4`倍（cold median）、value+5Jacは
`2.081/0.112 = 18.7`倍である。shared geometry込みのwhole epochでもPF6/foldは
V2よりvalue coldで約`1.75`倍、warmで約`1.79`倍、value+5Jac coldで約`2.48`倍、
warmで約`2.52`倍だった。

したがって、PF6/foldのLD evaluator単体をV2へ統合する速度上の候補にはならない。
同一geometryで勝てることを条件にしたPhase 7の判定は不合格であり、PF6/fold
production統合を行わず、この経路を研究成果として凍結する。V2/V3 production
routerは変更していない。Phase 7で新しいpacket数学やheuristic thresholdを
追加していない。

## 検証結果

```text
ctest --test-dir build-holonomic-m7 --output-on-failure
100% tests passed, 0 tests failed out of 12

./build-holonomic-m7/test_gm_phase7_shared
gm_phase7_shared PASS failures=0
```
