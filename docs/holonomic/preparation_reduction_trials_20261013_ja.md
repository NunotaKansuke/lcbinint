# 準備費用の削減試作と採否 (2026-10-13 client date)

結論: 今回試した4系統は低増光の準備費用を大きく削る採用条件を満たさなかった。solver sourceは検証済みv35へbyte一致で復元済み。未達の速度改善を成果として数えない。commit/pushなし。長時間jobなし。

checkout /rogue1_8/nunota/lcbinint/experiments/seeded-boundary-events、branch experiment/seeded-boundary-events、HEAD f3547723b79828d9c501182746028e99f7bad116、host rogue1。要求rtol=1e-3/1e-4と既存dirty研究を維持。

## 試作と反証

|案|数値検証|判断|
|---|---|---|
|v36: bracketでNewton foldを求め、二分探索を早期終了|最初のinside-edge版は新規超過92、physical-edge版112、6回の二分探索を保った版16。全て全14432条件で比較|撤回。点の減少で細い像・内部の根数変化のwitnessを失う|
|v37: degree-six radial polynomialを入力ごとに組み立ててHorner評価|全14432条件のmu不変、新規超過/未収束0。quad直接式との係数・根数比較1148 checks成功|準備時間が約10–12%増え、低増光全体約2–3%悪化。撤回|
|v38: exteriorのcount bracketと精密foldの対応を保持|全14432条件mu不変、新規超過/未収束0。根数probe変化は9条件、合計112回だけ減少|低増光の大きな費用削減を狙う案として採用しない。速度改善を主張しない|
|v39: 元のlens mapの逆三角不等式で円全体をsource外と除外|全14432条件mu不変、新規超過/未収束0。直接lens mapの19761 checks成功、既存2269 checks成功|対象低増光群で省ける四次式評価数の中央値0。全体timeはほぼ横ばい～微小悪化。撤回|

v36の全数未収束は各版とも既存6条件のまま。新規精度超過を持つ版は採用しない。既存超過3条件は全版の結果から除外せず追跡。v36 pilotを含むraw resultsとsource snapshotsを保持。

## 最終時計なし速度比較
各入力5反復中央値、one thread、fresh seed込み、caustic構築除外。compile/tests完了後にserial測定、順序を逆転。v37の最初の測定は他のbuildと重なった可能性があるためinitial_timingに分離し、全build後に測り直した最終値を使用。

|案|条件|順序|v35→試作 seed込み [µs]|入力ごとの比率中央値による削減|
|---|---|---|---:|---:|
|v37|uniform_0.001|forward|62.054→63.391|-1.53%|
|v37|uniform_0.001|reverse|62.166→62.059|-1.24%|
|v37|linear_0.001|forward|76.142→78.385|-0.87%|
|v37|linear_0.001|reverse|75.988→77.547|-0.77%|
|v39|uniform_0.001|forward|61.761→61.971|-0.15%|
|v39|uniform_0.001|reverse|61.457→61.608|-0.60%|
|v39|linear_0.001|forward|76.289→76.526|-0.62%|
|v39|linear_0.001|reverse|76.405→76.765|-0.09%|

## なぜ単純な短縮が効かないか
12回の二分探索は端点の初期値を狭めるだけでなく、途中のroot-count witnessも作っていた。physical foldのNewton収束と、その両側でinside/outsideを観測したことだけでは、そこまでの区間に細い別の像や根数変化がないことを言えない。v36の反証がこの制約を示す。

v37は実数上で同じ多項式でも、準備の配列操作と依存したextended-precision Horner評価が費用を増やした。v39の幾何学的boundは安全なempty判定の十分条件だが、対象の狭い境界近傍ではほとんど決定できない。数学的に正しいshortcutでも有効な適用率が必要。

## 成果物
- evidence/holonomic/fold_bracket_v36/{interior_edge_pilot,coarse_fold_pilot,six_bisection_pilot}: 各不採用版のsource/full result/audit。
- evidence/holonomic/count_polynomial_v37: 不採用版source、独立quadテスト、両順序の最終timing、audit。
- evidence/holonomic/exterior_witness_v38: full result/audit。sourceはgeometric_empty_v39/seeded_radial_frontier_before.hppに保存。
- evidence/holonomic/geometric_empty_v39: 不採用版source、直接lens mapテスト、全数比較、両順序timing。

## 続ける場合の設計上の課題
既存witnessを減らすには、隠れたroot-count変化を除外できる追加の情報が必要。次はseed・caustic情報を持つproviderから、source diskとcritical curveの関係や各像の接続を確認できる情報を渡せるか調べる必要がある。単なるfold Newtonの成功を完全性の証明に置き換えない。この境界のAPI/contract変更は今回実装していない。

v35は低増光uniform約61 µs、LD約76 µsの到達点で、保存済みVBMとの差は残る。coverage_certified=false、既存精度超過3・未収束6も維持。
