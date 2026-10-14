# Phase66: 係数2のgradient収束性・軽量診断

基準18666fc。ユーザー指示に従い安全係数2と現行実装を維持。solver/default変更なし。大きく動いた3例だけを調べた。全corpus/HMCベンチは実施していない。

## 結論

係数2でもvalue mesh上のgradientは不足する例がある。value toleranceの収束はgradientの収束を意味しない。今回の対象はすべてFiniteUncertified（2）で、ToleranceMetとして誤認証されていない。ただし小さい追加gradient budgetでも常に改善するわけではない。

## 方法

Phase64最良フラグ、同じLensParams。対象はbarycentric=falseなので、内部(X,Y,rho,m0,a)のX/rho/a微分と入力のxs/rho/a微分が一致する。
- ValueFirst、gradient追加budget=0：安全係数1/2/4、value RelTol=1e-3〜1e-8。
- 係数2、RelTol1e-8：initial level5/6/7（31/63/127点）。
- 独立値積分（GL radial 128/256 + 直接angular 256/512）を中心差分。h=rho×{1e-3,3e-4,1e-4}。各perturbationでtopologyを再生成。Fejer/K-rule/analytic gradientと別の積分だがlens式・topologyは共有する。
- 最後に既存ValueFirstのdefault gradient budget（4096nodes/4rounds）、係数2で1e-3/1e-4だけ確認。

## 係数2・value RelTol1e-3・追加gradient budgetなし

|case|u|微分|analytic|独立参照|差|
|---|---:|---|---:|---:|---:|
|caustic-cross|0|a|-2.799596|-2.404688|約16.4%|
|caustic-cross|0.5|X|76.786017|77.453095|約0.86%|
|rand030|0|rho|0.00438157|0.01299866|絶対0.008617、相対66%|

rand030は微分自体が小さく、相対率だけで実用上の悪さを判断しない。

FDはh3段で安定。n128→256差はaで5.23e-6、LD Xで1.76e-4、rhoで3.80e-9。上のanalyticとの差より十分小さいが、数学的な包含証明ではない。

## 収束系列

- caustic-cross/u0/a：value tol1e-3→1e-8で -2.7996,-2.6026,-2.5037,-2.3070,-2.4026,-2.3941。単調でない。initial31/63/127では-2.403630となり、参照との差約0.044%まで縮むが未認証。
- caustic-cross/LD/X：76.7860,76.7195,77.0005,77.5866。1e-7以降はvalue自体がInnerAccuracyLimitedで止まり78.1441。これを収束値と扱わない。gradient ledgerはradialが支配（例えば1e-6でradial31.21、inner0.193）だが、value stageの内側精度制限がrefinementを止める。
- rand030/u0/rho：0.00438157→1e-5で0.012997→1e-7で0.01299866付近。こちらは参照へ近づく。
- 安全係数4にすれば一律改善するわけでもない。caustic-cross/LD/Xは1e-3で75.7302となり、係数2より参照から離れる。

## default小budgetの追試（重要）

|case|u|value tol|gradient|参照|
|---|---:|---:|---:|---:|
|caustic-cross|0|1e-3|27.75464|-2.404688|
|caustic-cross|0|1e-4|-0.10529|-2.404688|
|caustic-cross|0.5|1e-3|76.07596|77.453095|
|caustic-cross|0.5|1e-4|93.68726|77.453095|
|rand030|0|1e-3|0.012998677|0.012998658|
|rand030|0|1e-4|0.012998682|0.012998658|

すべてFiniteUncertified、reason BudgetExceeded。valueは収束を保持し固定される。budgetなしの前回benchmarkだけの問題と片付けることはできない。単に小budgetをONにする処方も採用しない。

## 解釈と未解決事項

radial value meshだけでgradientを返す条件での不足は確認できた。一方、caustic-crossで追加refinement後に悪化する理由が、特異なintegrandへの不十分なnode配置なのか、追加nodeのgeometry/root tracking/gradient評価に数値問題があるのかは、この軽量probeでは未特定。実装バグがないとは結論しない。

従って安全係数2を維持する判断と、gradient品質が十分であるという判断は別。未認証gradientを実用的に安定させるなら、次はcaustic-crossの追加nodeの寄与を追って独立fixed-R derivativeと照合するのが小さい次手。今回それ以上の実装変更は行わない。

## 再現

evidence/holonomic/gradient_convergence_phase66:
- bash run.sh → raw.tsv（81行）
- bash run_budget.sh → budget.tsv（6行）
- probe.cpp / budget.cpp、run.log / budget.log、summary.json。
compiler flagsは各scriptに記録。FDはtest/reference専用で本線に使用しない。
