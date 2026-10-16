# 追加バグ・性能スクリーニング (2026-10-16)

## 対象と再現条件

- checkout: `/rogue1_8/nunota/lcbinint/experiments/algebraic_boundary`
- branch / HEAD: `experiment/algebraic_boundary` / `65faef7a3f5ade838119c39b77634a5ccbdf1ee3`
- 勾配 extension SHA256: `d610da31602364566a2870e0f7dd60031eb994eae1d1006a9f3f09b45e5c8d2a`
- 単一 CPU に固定。探索は既存の stress/contact ケース9件と決定論的 stratified ケース24件、計33ケース・873 epoch。
- 基本許容誤差は `tol=retol=1e-3`。一部は `1e-5` への感度を確認。性能比較は401 epochでVBMicrolensing 5.5と計測し、`Tol=RelTol=1e-3`。

結果の集計は [summary.json](summary.json)、個々のscreen結果は [post_patch/](post_patch/) にある。

## 確認できた修正

二重レンズの実軸対称性により、光源中心が実軸上にあり微分可能なら `dmu/dy=0` になる。近共鳴・軸上ケース (`s=1.000001, q=1e-6, rho=1e-3`) では、値が収束し勾配も有効扱いなのに、`tol=1e-3` の `du0` が `t=-0.025 d` で `-24.8`、`+0.025 d` で `+24.9` と大きな奇対称残差を持っていた。同一エンジンの反射有限差分は中央差分がゼロ近傍で、左右差分は符号反転した。

[holonomic_fused.py](../../../python/lcbinint/holonomic_fused.py) で、入力 source-y が厳密にゼロの行だけ、source-plane の法線勾配を対称性の値 `0` に射影する。値や既存の勾配有効フラグは変更しない。`symmetry_before.json` と `symmetry_after.json` に前後・刻み幅別の有限差分を保存した。

## Screen結果と残る問題

- 873/873 magnification値が有限、871/873でvalue convergence。NaN勾配は0。10 epochで勾配を無効扱いにした。全ケースの時刻順反転差は最大 `5.9e-12` 相対、`info()` と通常呼び出しは全ケースで完全一致。parameter再利用を調べた8ケースもfresh instanceと完全一致。
- 以前のstress/contact 9件では `tol=1e-3` の全勾配が有効扱い。今回の追加24ケースで、次の2ケースは値の成功と勾配の失敗が分かれた。
- `stratified_02`: 値経路は全25 epochでhexadecapole、値とvalue convergenceは正常。8 epochの勾配が有限値を返す一方で全成分無効、stop=`TopologyUnresolved`。同じ時刻で `tol=1e-3,1e-5,1e-7` のいずれも無効のまま。値経路だけが通り勾配用トポロジーが解決できない経路の差が残る。
- `stratified_09`: `q=1e-7, rho=1e-6, u0≈0.98465 rho` の接触近傍。`tol=1e-3` では全27値が有限だが2接触epochのvalue convergenceと勾配が失敗。許容誤差を `1e-5` または `1e-7` に下げても、接触と中心の選択時刻3点すべてで値またはトポロジーが未解決。stopには `EventLocationLimited` と `TopologyUnresolved` が残る。失敗は無効フラグで表に出ており、NaNや有効扱いの誤値ではない。
- `tol=1e-3` と `1e-5` の勾配差が0.32%を超えたケースは14件。case117系列では70/175〜189成分、`stratified_00` では9成分。多くは `fully_converged=false` またはsupport未証明のまま。これは許容誤差への感度を示すが、独立した正解との比較ではない。
- `stratified_00` のq/s有限差分を複数刻み・3許容誤差で計算した。選んだ摂動点の値は `value_converged=true` でも全て `fully_converged=false` で、value error estimateを刻み幅で割った値が勾配差より何桁も大きい。よってこの有限差分は勾配の正誤を決める参照にならない。詳細は `gradient_fd_tolerance_probe.json`。

`gradient_valid` は有限・利用可能な勾配を示すもので、勾配誤差の認証ではない。今回の448 finite-source積分epochでは `support_certified` は0件だったため、精度保証として扱わない。

## 速度の観測

[gradient_cost_profile.json](gradient_cost_profile.json) は、geometry準備が約0.02–0.20 ms、seeder初期化が約1.4–1.7 msに対して、特定epochの勾配評価が約71–132 msになることを示す。27 epoch全体の勾配呼び出しは、選んだ4ケースで約149–1,036 ms。対応する値呼び出しは約1.6–12.3 msだった。全epochが一様に遅いのではなく、中心・軸・臨界幾何に当たった少数epochが山になる。

401点のLightCurve対VBMicrolensing比較では、大きい一様光源 `rho=0.05` が148 ms対31 ms、大きい一様光源 `rho=0.5` が141 ms対18 msで、lcbinint側が約4.8倍・7.8倍遅かった。一方、同じ大きい光源のlinear LD `c=0.9` は155 ms対740 msでlcbinint側が約4.8倍速かった。普通のpoint-sourceは両者約0.36 ms。条件を合わせたこの行列でも、一様光源の速度差は残っている。相対値差は別実装間の比較値であり、VBMを厳密解とみなす精度認証ではない。

厳密な軸上でgradient shortcutを避ける候補も比較した。単一epochのfresh evaluatorでは短縮したように見えたが、4ケースの全LightCurve反復は0.2–0.7%差で、計測変動と区別できなかった。さらに `legacy_049` のrho勾配1成分が1.1%変化したため、候補は採用せず戻した。`post_axis_skip/` と `baseline_gradient_repeat.json` / `axis_skip_gradient_repeat.json` は比較記録で、現行ソース・buildには候補変更は残っていない。

## 次に掘る場所

1. `stratified_02`: 値経路hexadecapoleの勾配を得られる一般的なルート、または `TopologyUnresolved` の原因を調べる。
2. `stratified_09`: radial event locationの限界を、source-limb接触でどう回避できるかを調べる。
3. case117を含むtol感度: `fully_converged=false` を解消し、同一許容誤差内で独立参照を作ってから勾配精度を判定する。
4. 高コストepoch: gradient専用経路の点画像・トポロジー処理を分解し、普通のepochには追加負荷を出さず、該当epochだけを改善する。
5. 大光源一様光度の値経路: radial quadratureの分布とseed preparationを分けてprofileする。

この調査ではcommit/pushしていない。既存のdirty notebookと既存の未追跡生成物は維持した。
