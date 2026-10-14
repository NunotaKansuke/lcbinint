# Phase65: LD誤差ビンの診断とadaptive仕事量のA/B

基準e16d1de、Phase64最良研究フラグ。production router、fixed-n_r、PF6/GM、solverの既定設定は変更していない。既存のユーザー変更も保持。

## 結論

LDのAfs×qパネルの最大p95ビンは15行。主因はcase115/d_bin2/epoch15であり、V2のradial収束不足ではなく、保存VBM参照との不一致だった。保存参照を黙って差し替えてはいない。

value-onlyではnested_difference_safety=1（従来2）を、検証済みの研究候補として残す。weighted detailの1/8 floor、detail減衰、inner/geometry/event/roundoff ledgerは保持。LD 1e-3でnode総数11%減、paired whole-warm約4%改善。微分込みの既定設定は2のまま。run_selected_candidate.shで候補を再現できる。これは全入力に対する新しい厳密誤差保証ではなく、既存accuracy corpusによる観測検証である。

## 赤いビンの切り分け

case115: q=0.00076065753405687151、rho=0.0004224285403319365、s=1.9599718681909375。
epoch15のLD、保存reference=1581.2061363808323。
- adaptive RelTol1e-4: 1580.861787（191nodes）、推定絶対誤差0.077998。
- adaptive 1e-5: 1580.861797（287nodes）。
- adaptive 1e-6/1e-7: 1580.861799（367nodes）。
- independent GL128 radial + angular256: 1580.861798618748。
- independent GL256 radial + angular512: 1580.8617986144282。
- 1e-4 V2と独立参照の相対差は約7e-9。図の保存参照との差は2.1778e-4。

独立積分はFejer/K-ruleを使わないが、lens式/topologyは共有する。n=512/1024のepoch15およびn=1024のepoch7ではNaNとなり、さらに高次数での認証はできていない。失敗はindependent.tsvにそのまま残す。保存済み旧cartesian/polar inverse-rayの収束列も1580.8616付近に近づき、V2側と整合する。独立topologyの新しい完全証明ではない。

元reference artifact d10_c2/results.jsonのaccuracy_reference_reltol=1e-6に同じ値があり、結合・描画の誤りではない。round-trip入力と新規VBMicrolensing objectによる直接BinaryMagDark再計算でも、1e-6で1581.2061363808323が再現した。1e-4では1580.8782106393978、1e-7/1e-8では1581.2066702589213/1581.206716616568となる。tolを厳しくすれば独立参照に近づく挙動ではない。

最初のPython probeはpandas既定parserを使用したため、入力に最下位bit差があった。この結果は正式参照比較に用いず、vbm_probe.jsonに区別して保存。正式比較はfloat_precision=round_tripのvbm_roundtrip.json。同じ微小入力差でVBM 1e-6のepoch15 LD値は1580.863679717554に変わる一方、V2の対応比較の最大相対変化は5.046e-13。このVBM側の数値感度と、V2/独立積分の一致が原因判定の根拠。

**特定できたのは保存VBM参照/直接LD contour呼出し側に局在する数値不安定性まで。VBM内部のroot tracking・annulus誤差配分など、どのC++行が原因かまでは未特定。** 単にVBMのRelTolラベルを正解の保証と解釈しない。referenceを自動修正するコードは入れない。

## 全14432行A/B

CPU0固定、同入力・compiler最適化・3repeat中央値。value-only、Tol=1e-16、RelTol1e-3/1e-4。coldは毎epoch D14込み、warmはL2 seed更新/D14/topology込み（trajectory先頭cold含む）、radial-onlyは診断用。実行順はbaseline→safety1→initial15→k12→initial3。交互再実行ではないためsub-percent差は強い採用根拠としない。

|候補|profile|RelTol|warm p50 base→candidate ms|paired speedup|node総数 base→candidate|新規参照超過|
|---|---|---|---|---:|---|---:|
|safety1|uniform|0.001|0.233305→0.229185|1.0062|256075→242667|0|
|safety1|uniform|0.0001|0.274662→0.266728|1.0094|363091→341083|0|
|safety1|linear|0.001|0.285758→0.274641|1.0414|287499→255971|0|
|safety1|linear|0.0001|0.333382→0.321816|1.0093|389459→367515|0|
|initial15|uniform|0.001|0.233305→0.253681|0.9571|256075→322835|0|
|initial15|uniform|0.0001|0.274662→0.282853|0.9858|363091→406211|0|
|initial15|linear|0.001|0.285758→0.310416|0.9521|287499→350211|0|
|initial15|linear|0.0001|0.333382→0.343799|0.9813|389459→431779|0|
|k12|uniform|0.001|0.233305→0.233203|1.0004|256075→256075|0|
|k12|uniform|0.0001|0.274662→0.274606|1.0010|363091→363091|0|
|k12|linear|0.001|0.285758→0.284924|1.0045|287499→287499|0|
|k12|linear|0.0001|0.333382→0.330725|1.0068|389459→389459|0|
|initial3|uniform|0.001|0.233305→0.234847|0.9905|256075→248903|0|
|initial3|uniform|0.0001|0.274662→0.278242|0.9903|363091→358755|0|
|initial3|linear|0.001|0.285758→0.286842|0.9913|287499→280667|0|
|initial3|linear|0.0001|0.333382→0.336573|0.9920|389459→385271|0|

全候補で14432行のvalue収束、warm status変更0。保存reference超過は1e-3=0、1e-4=既存3のまま。全lane rawを保存。p99は一部増加しているため全tail非回帰とはしない。

- safety1: value-only候補として保持。LD1e-3 radial paired約9.8%改善、whole約4.1%。uniformのwhole利益は約0.6%に留まる。
- initial15: 7点で足りるpanelにも15点を払うためnode約11〜26%増。wholeで約1〜5%遅く不採用。
- initial3: 7→3点開始。node約1〜3%減でもwhole約1%遅く不採用。初期3点は従来と同じ2段のdetail履歴を持たないため、単に既定のmin levelを下げる変更は採用しない。
- k12: K-rule high16→12、low8と相対gate1e-8を維持。LD mu差最大2.314e-11、node数同一。whole約0.5〜0.7%で利益が小さく、今回は既定16を維持。試作overlayのみ保存。全node角度rescueの増加等は行っていない。

## 独立検証とgradientの制限

safety1のcheck_adaptive_reference由来テスト550行中466有効、observed violation0。84行はreference未認証。reference_safety.csvを保存。

ValueFirst、gradient追加budget=0の220行A/Bでは、両方ともvalue未収束20行、Invalid gradient成分100で同一、新規Invalidなし。finite gradientでも28成分がToleranceMetからFiniteUncertifiedへ変わった（逆方向0）。最大scaled gradient差0.98346。例rand030/u=0/RelTol1e-3のrho微分は0.00438→-0.97908、双方FiniteUncertified。これをgradient parity passとはしない。value meshを粗くすれば未認証gradientは変わるため、safety1を微分込みの既定へ一括採用しない。qualityを偽って昇格する変更なし。

新しいsolver変更を入れていないので既定unit全体の再実行はしていない。今回変更した設定そのものは上の独立参照・whole・gradient A/Bで検証した。gradientを含む全面的な既定昇格は未達。

## 再現と失敗記録

evidence/holonomic/adaptive_accuracy_phase65:
- probe.sh: 最悪ビン由来4caseのtol系列。
- reference.sh: GL/直接angularの独立積分。
- build_ab.sh / run_ab.sh: baseline、safety1、initial15。
- build_k12.sh: isolated include overlay、high12/low8。
- build_initial3.sh: header mirrorを/tmpへ作りminimum initial levelだけ研究変更。
- run_next.sh: K12/initial3 full A/B。最初のinitial3 overlayはrelative includeにより同じ型が二重定義されcompile失敗した。next.logに残し、全header mirrorに修正して正常実行した。
- vbm_roundtrip.py: 正式な入力一致VBM probe。vbm_probe.pyはpandas既定読み込みによる感度診断。
- run_validation.sh: roundtrip VBM、独立reference、ValueFirst比較、微小入力差V2比較。最初のrounded_inputは2epochだけだったためrunnerが拒否した。既存4epoch/trajectoryに直して再実行済み。
- summarize_ab.py / finalize.py: machine-readable集計。raw、compiler/HEAD/hashはprovenance.json。
- run_selected_candidate.sh: **value-only**最良候補safety1。生成先selected.tsvはまだ別名で再実行していない。対応済み実測rawはsafety1.tsv。

今回の結論は「value-onlyには小幅な削減余地あり、赤いbinをV2のradial不足とみなす根拠なし」。K-rule点数・初期点数の変更は費用対効果が弱い。D14やbranch/topologyのcertificateを弱めず、gradient既定も維持した。
