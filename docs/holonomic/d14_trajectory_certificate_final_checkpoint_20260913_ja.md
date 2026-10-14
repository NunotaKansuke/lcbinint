# D14 trajectory certificate 最終検討（2026-09-13）

## 結論

前epochの情報からD14/topologyを安全に省く案を、14,432-row trajectoryのwarm 5,412 epochで検証した。現時点の数値保証を維持したままwhole中央値を0.2 ms級へ入れるtrajectory shortcutは成立しなかった。production経路は変更しない。

局所`P=P_t=0` continuationは0.064 ms p50と速く、cell planがoracleと一致する行は3,734/5,412（69.0%）だった。しかし前plan一致gateには308件、event shift/gap < 0.1を加えたgateにも24件の偽受理が残った。eventの出生・消滅とreal/complexの入替えをendpoint情報だけでは証明できない。

全14根を保持する別案として、前epoch根へWeierstrass補正を行い、NeumaierのGershgorin型包含円板を構成した。14円板が互いに素なら各円板に1根ずつ存在するため、root identityとcompletenessを同時に扱える。ただしpoint-qf試作は3 sweepで0.510 ms p50、8 sweepで1.002 ms p50であり、現行warm whole約0.50 msより既に重い。real/complex分類可能な円板だけに限定しても、point演算では3 sweep 72件、8 sweep 18件のtopology不一致が残った。outward interval化は必要だが、追加費用を払う前から速度条件を満たさないためproduction実装へ進めなかった。

## 現行経路の下限

root-work記録を無効にしたexclusive profileでは、warm `classify_cells` p50は0.290 ms、`radial_events` 0.251 ms、D14 solve 0.148 msだった。D14 solveの主な内訳はdouble presearch 0.056 ms、D14Real 0.025 ms、qf residual評価0.029 ms、その他約0.037 ms。cell topology remainderは0.038 msである。

同じcorpusのadaptive radial-only p50はRelTol=1e-3で約0.152 ms、RelTol=1e-4で約0.191 ms。従ってwhole 0.2 msには、D14 root solveだけでなくevent生成とcell分類をほぼ全消去する必要がある。局所continuationの0.064 msを足した理想値でも1e-3は約0.216 msで、しかも31%のoracle fallbackと安全な出生・消滅certificateが未解決である。

## 試した案と判断

- 前eventの局所continuation: 速いが69%一致。cheap heuristic gateは偽受理を除去できず棄却。
- 前根からdirect D14Real/Aberth 2/4/8/16回: incumbent warmより約3%以上遅く棄却。
- Weierstrass/Neumaier包含円板: completenessの表現は適切だが、qf point試作の時点で0.5–1.0 ms。interval化前に速度条件を失い棄却。
- 単純warm-presearch省略、固定sweep短縮、独立Newtonは過去checkpointどおり後段補正・fallbackを増やすため再採用しない。

今回の結果は、D14を別solverへ置換すれば中央値が大幅に下がる、という仮説を支持しない。現行の全根状態はpositive-real event、complex soft cut、warm seed、completenessを一度に供給しており、それらを分離して再認証する固定費が現行solveの節約額を上回った。

## 維持するもの

`8852a28`までのsafe active presearch、local-pair precision、overflow-safe complex division、interval Rouché tail救済、lazy D14 precision blockを維持する。既存qf residual/completeness、topology authority、production routerは変更していない。研究benchmarkは候補をproductionへ渡さず、oracle比較だけを行う。

raw dataと再現コマンドは `evidence/holonomic/d14_trajectory_certificate_final_20260913/` に保存した。
