# Phase67: gradient精度診断・途中split欠落と収束停止の修正

基準6d533a6。安全係数2、K-rule16点、初期7点を維持。変更はisolated adaptive control flowとunit。production router、fixed-n_r、PF6/GM、精度gateの閾値は不変。

## 特定した2件のバグ

1. h-splitで親を先にinactive、両子をactiveにしていた。左子を計算後、gradient round/node budgetによって右子が未計算のまま終了すると、右子(level0、q=0)を集計し、領域の半分を欠いたgradientを返した。valueはsnapshotなので欠落が表示上隠れる。caustic-cross/u0/aの4roundsで+27.754643になる直接原因。
2. all_gradient_pass/all_gradient_error_passのAND初期値が!gradients、つまり微分要求時false。全成分が収束しても集合の収束がtrueにならず不要なrefinementを続けた。

## 修正

子をinactiveで生成し、左右両方が成功したときだけ親→子を切り替える。失敗時は完全な親の積分とledgerを保持する。子での数値失敗は既存invalidフラグに残し、InvalidをFiniteUncertifiedへ格下げしない。split/discarded node統計もcommit成功時に記録。全成分ANDはtrueから開始する。

value snapshot、独立gradient tolerance、None/Strict/ValueFirstの意味は維持。予算増加や重量fallbackは追加しない。

## node単位の診断

caustic-cross/u0/a、value RelTol1e-3、係数2、追加rounds0/1/2/3/4/8。active sampleのanalytic gradientを、同じRとmap Jacobianでcold rootから再計算したgradient、および独立arc幅の固定R中心差分（2刻み幅）と比較した。

cold差の重み付き絶対和は最大約2.03e-9。coldは全node reliable。今回の一様光源例ではwarm trackingは大きな差を説明しない。有限なFDについて刻み幅半減差の重み付き和は約5.5e-6以下。極端にfoldに近い1〜2nodeはFD刻みを確保できずNaNとした。NaNをゼロにした積分値を参照にはしない。

最初のprobeはactiveなlevel0子に遭遇しFejer例外で終了した（run.log/before_partial.tsv）。未完成panelが集計対象だった直接証拠。probeはlevel0をスキップするよう修正し、修正後の完全rawをraw.tsvに保存。

## 残存精度

独立GL+直接angular値のFD参照はPhase66を再利用する。

|case / 微分|参照|4rounds|8rounds|64rounds|
|---|---:|---:|---:|---:|
|caustic-cross/u0/a|-2.404688|-2.870508|-2.380565|-2.417061|
|caustic-cross/u0.5/X|77.453095|76.075962|84.191726|77.734213|
|rand030/u0/rho|0.012998658|0.012998677|0.012998690|0.012998690|

+27.75の不完全領域の値は消えたが、正しい領域でもgradientのradial収束は非単調。aのdefault4roundsは依然約19%差、64roundsで約0.51%差。LD/Xは64roundsで約0.36%差。両例とも約2650unique nodes。全てFiniteUncertifiedで、精度達成とはしない。

rand030は参照へ近いが、8rounds以降はgradient側EventLocationLimited。実際の誤差と保守的なevent error estimateを分けて扱う必要がある。

node監査はu0/aについて実施。LDの全nodeでK-rule/直接angular derivativeを独立監査したわけではない。すべての微分にバグがないという結論ではない。

## 精度を分離する方針

値の安全係数を強めるだけではgradientは管理できない。valueは収束して固定し、gradientを独立のabsolute+relative toleranceとbudgetで判定する既存ValueFirst/Strictが妥当。今回その制御バグを修正した。

64roundsを全入力へ一律追加する案は採用しない。nodeが大幅増でも難例は未認証で、少数roundsの追加が常に改善するわけでもない。次の対象はgradient integrandの局所正則性、radial estimator、event uncertainty。valueのOKをgradientの精度達成と解釈しない。

## 検証

- 新unit: 全成分収束で初期7node終了、round/node予算の片側中断でも完全領域保持、数値失敗はInvalid。
- 修正前1495checks/7failures、修正後1495checks/0failures。
- CTest adaptive_radial/quartic_sturm: sourceから再buildして2/2pass。
- 独立reference550行中466有効、observed violation0。84行は参照未認証。
- whole value-only14432行、cold/warm/radial全て収束、Phase64最終rawとmu完全一致・status差0。1repeatのregression確認で性能主張には使わない。
- 全corpusの独立Jacobian oracle監査は未実施。

## 再現

evidence/holonomic/gradient_nodes_phase67:
unit.sh / old_unit.sh（旧ヘッダ6d533a6を/tmpへ復元）、run.sh（node probe）、
budget.sh（係数2固定、CSV safety列は追加round limit）、whole.sh、
raw/before_partial/budget TSV、node_summary/whole_parity/validation JSON、reference.csvと各log。

CTest:
cmake --build /tmp/lcbinint-holonomic-adaptive-build --target test_adaptive_radial test_quartic_sturm check_adaptive_reference -j2
ctest --test-dir /tmp/lcbinint-holonomic-adaptive-build -R 'holonomic_(adaptive_radial|quartic_sturm)$' --output-on-failure
