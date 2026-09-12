# D14 median/tail Phase 4 checkpoint（2026-09-13）

## 結論

easy trajectoryの通常control flowを`perf`で再調査し、double D14 presearchが約19%を占めることを確認した。Rouchéだけでは中央値を動かせないため、既存active presearchを `tol=1e-11, patience=1` まで進めた候補と、既存interval Rouchéを独立・結合A/Bした。

結合候補は14,432行でcoverage/status差0、reference violationはbaselineと同じ（1e-3: 0、1e-4: 3）。whole中央値は1e-3でcold `0.5453→0.5239 ms`（-3.9%）、warm `0.4948→0.4763 ms`（-3.8%）。1e-4はcold `0.5914→0.5692 ms`（-3.8%）、warm `0.5590→0.5432 ms`（-2.8%）だった。warm p99は1e-3で-2.6%、1e-4で-5.4%。cold p99は+0.9〜1.1%、maxは+1.5〜1.8%で、production採用基準にはまだ届かない。

active単独は中央値が約3〜4%改善した一方、qf-coldへ落ちる少数例でmaxが26〜41%悪化した。interval Rouchéを組み合わせるとその異常tailを救い、max悪化を約2%へ抑えた。したがって両候補は補完関係にあるが、現時点ではresearch opt-inを維持する。

## 試した案

- presearch固定上限120: qf-coldが増え、warm coverageが7214/7216へ低下し、p90/p99が大幅悪化。棄却。
- 固定上限160: coverageは維持したがcold/warm tailが大幅悪化。棄却。
- 前epoch binary128 rootへdouble presearch変位だけを加えるlow-limb保持seed: D14Real時間を減らさずwholeも悪化。実装から戻した。
- active `1e-11/patience=1`: medianに有効だが単独tailは不採用。
- Rouché difficult-root-first: 点prefilterのmargin順に正式interval shiftを行う案を試した。ループ適用位置を修正した再測定でも受理数cold 24/warm 22、topology差0を維持したが、通常root順に対する改善は測定ノイズを超えず、sort分だけ通常中央値を増やし得るためコードから戻した。

全根solver以外では、physical-fold分類のsubresultant seedは108-case profileでcold 652/674、warm 2034/2036成功しており、通常中央値の主犯ではなかった。残る大幅短縮には、warm event/cell planをD14なしで更新・認証する仕組みか、D14 presearchの相互作用計算自体を別表現で減らす必要がある。単純なsweep短縮は後段qfへ費用を移すため採用しない。

production router/defaultは変更していない。このcheckpointでは採用コードを増やさず、`tol=1e-11/patience=1` とRouchéの結合候補もresearch opt-inのままとした。既存certificate、不成立時qf-cold、fixed APIには影響しない。
