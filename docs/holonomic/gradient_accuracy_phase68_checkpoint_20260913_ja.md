# Phase 68: 微分の精度・独立refinementの小規模A/B

基準: dev/holonomic e941b91。production / fixed-n_r / safety=2は変更なし。
今回の目的は微分精度の原因調査であり、速度の採用試験ではない。

## 結果

3難例の選択微分について、4種類×72設定=288回を実行した。
全trialでvalueは収束。gradient-only変更では既に収束したprimal snapshotがbitwise一致。
選択微分は全trialでFiniteUncertifiedのまま。Invalidを格下げしていない。
本番変更の採用はなし。微分の独立meshは有望だが、一律の設定変更には根拠不足。

独立参照はPhase66のGL128/256・直接angular積分のvalue有限差分。刻み幅変化も確認済みの保存値を再利用した。今回は全5成分の独立認証ではない。

微分だけのp-refinementを最大15点で止め、以降h-splitする試作（64追加round上限）:

| 対象 | 既存最大255点: nodes / 相対誤差 | gradient最大15点: nodes / 相対誤差 |
|---|---:|---:|
| caustic-cross u=0, a |2650 / 0.515%|577 / 0.0881%|
| caustic-cross u=.5, X |2651 / 0.363%|577 / 0.496%|
| rand030 u=0, rho |203 / 約2.5e-6|511 / 同程度の小誤差|

数字は選択微分の観測誤差であり、全成分の保証ではない。点数削減は示せたが時間は測っていない。短い8/32roundでは誤差が大きく、64roundの結果を通常の4round budgetの性能として扱ってはいけない。LDでは一様に精度が上がらず、rand030では余計な点数を使う。

## 確認できた原因と未確定部分

1. valueに使われるHybridEmbedded誤差推定は微分には適用されず、微分はweighted interpolation-detail normのまま。微分の収束判定が保守的になる構造はコード上確認できた。ただし、実際の積分値の非単調性までこれだけで説明できない。
2. 微分は区間間で強く相殺する。rand030 rhoでは約69.798、-139.978、70.1867などの寄与から約0.013を作る。caustic-crossでも正負の大きな寄与がある。単純なrelative accuracy要求はゼロ付近で不適切で、absolute toleranceが必要。
3. 前Phase67のuniform aのnode auditでは解析微分と固定R finite difference、およびcold/warmの値が一致した。少なくともその例を微分式やwarm tracking全体の故障とみなす根拠はない。LDの全node・全成分については未検証。
4. grad rtolを1e-3から1e-2へ変更、atolを独立に1e-4としても、今回の難例では未認証のまま。radial detailやevent uncertainty、他の微分成分の条件が先に律速する。緩い要求値に変えるだけでは解決しない。
5. physical fold mapの有無をactive panelログへ記録した。これは全eventの正しさの独立証明ではない。

## 試作と不採用理由

- raw: max_level=4/5/6/8をvalueにも適用。primal meshを変えてしまうため診断用途のみ。
- gradient_only: primal stageは現状8固定、微分stageだけmax_level変更。primal不変。早いh-splitは一部有効だが非単調性を解消しない。
- frozen_priority: value snapshot取得後、現在のgradient meshのvalue推定にschedulerが引き戻されないよう研究用headerで変更。72行中4行の微分値が変化、primal差ゼロ。これが主要因ではない。
- hybrid_gradient: 微分にも値と同じnested-difference estimatorを研究用に適用。例えばLD max_level8/64roundで77.7342→78.0754へ悪化（参照77.4531）。推定誤差を小さくするだけでは観測誤差は改善しない。採用しない。

全て/tmpに作るheader mirrorで実験し、本番headerは変更していない。既存のdecay/invalid/event/inner gateは保持した。qf化や重いfallbackは追加なし。

## 次に狙う部分

微分は独立した緩めのabsolute+relative toleranceを設定する方針が妥当。ただし今回の値を推奨defaultとして採用する段階ではない。
まずLDも含む選択nodeの直接angular微分との比較を広げ、radial誤差とinner誤差を分離する。その後、gradient detailの減衰が遅いpanelだけ早めにh-splitする方法を検証する。全panelを常に15点上限にする方法は採用しない。
強い区間相殺については、moving-boundaryのchain ruleを含めた変換後微分の評価が次の仮説。ただし境界項の取り扱いを検証せず変更しない。

## 再現と検証の範囲

repo rootから順番に実行（mirrorを共有するので並列実行不可）:

```sh
bash evidence/holonomic/gradient_accuracy_phase68/run.sh
bash evidence/holonomic/gradient_accuracy_phase68/gradient_only.sh 2> evidence/holonomic/gradient_accuracy_phase68/panels.log
P68_FROZEN_PRIORITY=1 bash evidence/holonomic/gradient_accuracy_phase68/frozen_priority.sh 2> evidence/holonomic/gradient_accuracy_phase68/frozen_panels.log
P68_GRAD_HYBRID=1 bash evidence/holonomic/gradient_accuracy_phase68/hybrid_gradient.sh 2> evidence/holonomic/gradient_accuracy_phase68/hybrid_panels.log
python3 evidence/holonomic/gradient_accuracy_phase68/summarize.py
```

同じcompiler flags、CPU0、同じ入力で実施。入力は既存reference_cases.tsv、各設定は1epoch warm-up後の同パラメータwarm epochでありtrajectory性能測定ではない。
raw / audited TSV、panelログ、summary.jsonをevidenceに保存。288trialの集計sanity pass。solverコード変更がないため既存全corpus/CTestの再走は行っていない。誤差比較は3選択微分のみで、全5Jacのaccuracy/status非回帰を主張しない。
