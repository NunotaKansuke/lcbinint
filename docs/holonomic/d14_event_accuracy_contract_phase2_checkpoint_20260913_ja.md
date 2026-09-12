# D14 event-accuracy contract Phase 2 checkpoint (2026-09-13)

## 結論

全14根の qf Aberth が global step 条件へ入らない場合でも、候補根の周囲に互いに交わらない円板を構成し、各円板で D14 の一次Taylor項が残りを十分なmarginで支配する場合は、Rouchéの定理から各円板に根がちょうど1個ある。この条件を研究経路として実装した。円板内でroot identityを保ったまま2回の独立Newtonを行い、その根集合から従来どおりphysical eventとcomplex soft cutを構築する。

14,432-row trajectoryでは qf-cold callを cold `58 -> 30`、warm `44 -> 12`、qf-cold sweepを cold `12498 -> 4956`、warm `12912 -> 2364`へ削減した。受理行のtopology中央値は cold `45.09 -> 4.99 ms`、warm `53.79 -> 4.90 ms`だった。

一方、この実装のRouché不等式はbinary128で大きなmarginを要求しているが、まだoutward-rounded interval boundではない。従ってproduction contractの置換には採用せず、`HOLO_D14_EVENT_CONTRACT_RESEARCH` と `HOLO_D14_EVENT_CONTRACT_ACCEPT=1` の両方がある研究経路に限定した。production routerと通常ビルドのD14受理条件は変更していない。

## 数学的な受理条件

候補根 `c_i` の周りで

```
D14(c_i + w) = a_0 + a_1 w + ... + a_14 w^14
```

と展開する。実根候補は円板中心を実軸上へ射影する。最近傍候補との距離を `s_i` とし、`r_i < s_i/3` の範囲で

```
|a_1| r_i > 16 (|a_0| + sum_{k=2}^{14} |a_k| r_i^k)
```

を満たす半径を探す。14個すべてが成立すれば、円板は互いに交わらず、それぞれがD14の根を1個含むためdegree 14の完全性を得る。実軸中心の円板に根が1個しかない場合、実係数多項式の共役対称性からその根は実根である。

受理後は各円板中心から2回だけstructured qf Newtonを行い、全iterateが元の円板内に留まることを要求する。既存のscalar residual gateとNewton-sum completeness checkも残している。deflationは使わず、complex rootも捨てずにsoft cutと次epochのwarm seedへ渡す。

## 既知反例

case 49 / d-bin 1 / epoch 0 は qf-warm residualが `4.63e-28`でも、oracleに対してpositive rootsが `5 vs 6`、physical eventsが `4 vs 6`となる既知反例である。この候補では円板を14個分離できず、新contractはrejectし、従来のqf-coldへfail closedした。残差だけで受理する経路は導入していない。

## 14,432-row correctness

RelTol `1e-3` と `1e-4`、full-cold / trajectory-warm / radial-onlyを比較した。

- topology status / cell count / event count mismatch: `0`
- physical event count mismatch: `0`
- complex soft event count mismatch: `0`
- value convergence / stop / numerical status mismatch: `0`
- coverage: 全lane・両toleranceで `7216/7216`
- baselineとの最大scaled mu差: cold/radial `9.43e-13`、warm `9.43e-13`
- 保存VBM `1e-6` referenceに対する新規violation: `0`
- 既存observed violation: RelTol `1e-3` は `0`、RelTol `1e-4` は各lane `3`。baselineとcandidateで同一。

受理された60行の最終root setをqf-cold oracleへ最小距離対応させた最大差は `8.34e-11`（v平面）だった。これはcandidate側のevent/value parityを壊していない。qf-coldが400 sweep上限で止まる行もあるため、qf-cold root座標そのものを絶対真値とはみなしていない。

## whole-epoch performance

同一binary、CPU 0固定、各row 3反復中央値。単位ms。

| RelTol | lane | p50 baseline/candidate | p90 | p95 | p99 | max |
|---|---|---:|---:|---:|---:|---:|
| 1e-3 | cold | 0.5492 / 0.5455 | 0.8806 / 0.8729 | 1.0334 / 1.0242 | 3.4481 / 3.4524 | 56.283 / 54.770 |
| 1e-3 | warm | 0.4908 / 0.4914 | 0.8096 / 0.8116 | 0.9641 / 0.9627 | 1.4783 / 1.4771 | 56.218 / 54.550 |
| 1e-3 | radial | 0.1490 / 0.1487 | 0.2985 / 0.2988 | 0.3565 / 0.3557 | 0.4961 / 0.4955 | 0.909 / 0.906 |
| 1e-4 | cold | 0.5965 / 0.5901 | 0.9902 / 0.9754 | 1.2123 / 1.1925 | 3.5004 / 3.5352 | 56.303 / 54.957 |
| 1e-4 | warm | 0.5563 / 0.5566 | 0.9379 / 0.9389 | 1.1296 / 1.1312 | 2.2174 / 2.2347 | 56.255 / 54.730 |
| 1e-4 | radial | 0.1886 / 0.1883 | 0.4038 / 0.4047 | 0.5119 / 0.5117 | 0.8551 / 0.8544 | 2.265 / 2.238 |

受理対象は全行の1%未満なので、p50からp99は計測揺れ程度である。残ったqf-cold行がmaxを支配するためmax改善も約2.4--3.0%に留まる。これは一般fast pathの高速化ではなく、一部の50 ms級restartを約5 msへ変えるtail rescueである。

## A/Bして不採用にした案

1. **exact dyadic rational Sturm**: binary64入力をexact rationalへ変換し、integer pseudo-remainder chainでpositive-root countを証明した。case 149では全14試行を証明できたが中央値 `94.47 ms`で、qf-coldより重いため不採用。
2. **既存positive-real interval Sturmのみ**: strong certificateは cold 8 / warm 2程度に限られ、`ChainPivotUncertain`が主な停止理由だった。
3. **event-contract rootを次epochのwarm seedから外す**: warm p99が約3.3%悪化したためrevert。認証済み全根状態はwarm seedとして保持する。
4. **6回の独立Newton**: 2回とevent/value parityが変わらず、受理行へ余分なstructured qf評価を課すため2回へ削減。

## 判断

「global Aberth stepが収束していない」ことと「event listが未確定」であることは同義ではない。分離円板を構成できる60/102候補では、全根完全性、実根性、root identityを局所条件で扱い、qf-cold restartを避けられた。一方、円板が重なる42候補はevent ambiguityが残るため従来経路へ落としている。

次にproduction候補へ進めるには、Taylor係数・絶対値・半径演算をoutward-rounded intervalで評価し、Rouché不等式を形式的な包含証明へ変える必要がある。また残る qf-cold 30 cold / 12 warm行は、重なるclusterのevent roleを調べ、positive-real orderingが曖昧なclusterだけ局所高精度化するのが次の対象である。

## 再現

raw、集計スクリプト、コマンド、provenanceは `evidence/holonomic/d14_event_contract_phase2_20260913/` に保存した。
