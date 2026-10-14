# Phase 70: 同一7点の式由来R微分で積分誤差を予測できるか

基準b80c7e2。research probeのみ。本番controller、停止条件、fixed APIは変更なし。

## 数学的背景

VBMicroLensing §2は、境界点の微分を使うparabolic correctionを説明している。特に式(8)とその直後では、二種類の補正の平均を採り、その差をarc誤差推定とする。したがって「差分による推定は使わない」は正しくない。重要なのは追加のlens root solveなしで二種類の近似を作る点。

一次資料:
- https://arxiv.org/html/2410.13660v1 （§2、式5–8）
- https://dlmf.nist.gov/3.3 （補間）
- https://dlmf.nist.gov/3.5 （数値積分・剰余）

以下は本研究での導出。uniform observableをF(R)=R*sum(theta_right-theta_left)とすると、

`t_R=-P_R/P_t`, `theta_R=2*t_R/(1+t²)`

からF_Rが得られる。fold変数変換R=R(x)、J=R_x後のg(x)=F(R(x))*Jについて、

`g_x=F_R*J²+F*J_x`。

通常foldではF_R自体が発散的でもgは正則になり得る。ただし現probeは個々のendpoint微分を作るので極端なfoldの丸め安定性を証明したものではない。(m,v)で有限な積へ整理する拡張は未実装。

7個のFejer点x_iでgとg_xを使い、13次Hermite補間Hを作る。L_iを7点Lagrange基底とすると

`H=sum[(1-2 L_i'(x_i)(x-x_i))*L_i²*g_i + (x-x_i)*L_i²*g'_i]`。

積分の重みは固定定数として事前計算可能。probeはPythonで重みを作り、0–13次多項式の積分exactnessを1e-10以内で確認する。物理量をsample-fit packetへ置換するproduction経路ではなく、誤差予測用の独立研究。

g∈C14ならHermite剰余は `g^(14)(xi_x)/14! * product(x-x_i)^2`。従って全区間の14階微分上界Mが得られれば、積分誤差は `M/14! * integral product(x-x_i)^2` で囲える。しかし7点のg,g'だけではMは得られない。

明確な反例は `g(x)=A*product(x-x_i)^2`。全7点でg=g'=0なのに積分は非零でA倍できる。A=1の積分は約0.0002468018093。この事実はレンズ方程式由来の追加制約を使う方法を否定しないが、有限のsample/derivative情報だけを厳密保証と呼ぶことはできない。

## 実装・監査

既存110入力のうちu=0を使用。非empty・非degenerateの247cellに対し:

- 同一7点でboundary/root解と解析F_Rを計算。
- map後のg_xをx±1e-5の有限差分で検査（FDは診断専用）。
- Q7、同じ7点のHermite補正QH、Q15を比較。
- GL64/128を独立radial参照とし、差が1e-7*max(1,abs(reference))未満の区間をusableとする。
- endpoint reliability / nonfiniteは除外として記録。

参照はFejer/Hermiteと異なるquadratureだが、同じtopology/境界式を共有する。whole-epoch独立physical oracleではない。LD、5パラメータ微分への適用は未実装。

候補の誤差推定:
1. abs(QH-Q7)×係数1/2/4。
2. 補正多項式H-L7の絶対値積分をGL64で評価（物理node追加なし、ただし上界ではない）。
3. 補正多項式の単項式係数絶対値から積分上界を作る。これは補正多項式の上界であり、真のg-H剰余まで保証しない。

## 結果と採用判断

集計の正確な件数と値はevidence/holonomic/derivative_quadrature_phase70/summary.json、全区間はpanels.tsv。

7点Hermiteは多くの区間でQ7より数桁改善する。一方Q15には大半で届かない。つまり高階微分情報なしでも追加root solveを節約する可能性はあるが、これだけで安全な終了判定とはならない。

重要な反例: paper_highA panel12（両側fold map）。

- Q7 ≈10892.263171
- QH ≈10888.926277
- Q15 ≈10867.732610
- GL128 ≈10837.076301、GL64との差≈3.5e-5
- Q7実測誤差≈55.19に対し abs(QH-Q7)≈3.34

係数4でも過小評価。補正のsampled L1≈8.23も係数2では不足。係数上界≈5863なら検出するが極めて保守的。普通区間にも同じ保守性を課すと狙いを失う。単に係数を大きくして採用しない。

node別base_us / derivative_usは参考診断。微分追加費はroot solveを含むbase費用の数%程度に見えるが、各nodeの短いtimer、cold reference_arcs、重み適用費を含まないためwhole-epoch速度差を主張しない。warm trackingやLDで同じ比になる保証もない。

## 次の判断

現段階ではproduction acceptanceを変更しない。QH補正だけでQ7を安全に受理する案は不採用。

残る有望案は、式由来の局所高階情報/区間内conditioningを利用して「同一node予測の信頼できる範囲」を制限すること。D14 eventで実根数一定でもcomplex特異点までの距離や高階微分の大きさまでは保証しない。これを無視してfold mapだけで滑らかと決めない。

微分経路へ適用するにはg_pのR微分、すなわちmixed sensitivityが必要。既存5Jacをそのままg_xとして流用してはいけない。この費用とnear-fold正則性を評価してから進める。

## 再現

```sh
bash evidence/holonomic/derivative_quadrature_phase70/run.sh
python3 evidence/holonomic/derivative_quadrature_phase70/analyze.py
```

CPU0、既存research compile flags、入力reference_cases.tsv。rawとsummaryを保存。productionコードの変更がないため全CTest/trajectoryの再走は行わない。候補の不安全例が見つかった段階なのでrouterへの組込みとwhole-epoch勝利試験は行わない。
