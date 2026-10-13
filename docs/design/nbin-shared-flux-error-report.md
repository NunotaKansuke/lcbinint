# Uniform / linear LD 共通の軽い誤差モデル

追補：過小評価4件の原因と修正は [nbin-shared-flux-underestimate-report.md](nbin-shared-flux-underestimate-report.md)。以下の数値は修正前v6の記録。

2026-10-13。実験入口 `binary_flux_estimate` に統合した。
旧名 `binary_ld_flux_estimate` は同じ関数へのalias。通常master経路は変更しない。

## 結果

同じ20条件のlinear LD (c=.5) で、指標/実誤差の中央値は **8.73→2.06**、最大は **261→32.4**。
Uniformの20条件では中央値 **2.22→1.78**、最大 **58.9→32.9**。
独立参照付きの別geometry・c=0,.2,.8を加えた計58条件で、通常の増光率とbit単位で一致、誤差推定用の追加rayは0。

58条件中4条件は過小評価で、最小の指標/実誤差は0.683。
この標本で `indicator<=1e-5<actual_error` は0件だが、上限保証・一般的なcoverage・生産用の受理判定は未達。
`certified=false`、`complete_error_model=false` を維持し、target_metは返さない。

## 過大評価の原因と修正

1. **端セルの残差だけを積算していた。**
   平方根型端点では内側のmidpoint積分の尾部も同じ誤差次数で寄与する。
   端点と尾部を合成した単位格子のmomentを共有関数にした。
   同じmomentをuniformの像端・LDの輝度端に使う。
2. **辺・行ごとに絶対値を取っていた。**
   格子位相で符号が変わるmodel residualを像成分ごとに集計してから絶対値を取る。
   独立した像成分間の相殺に頼って指標をゼロへ縮めない。
3. **採用しなかった格子も数えていた。**
   通常component refinementが細かいfluxを採用するとき、粗い格子の指標を置換する。
   試行だけで不採用になったfine fillの指標は捨てる。投入した計算量counterとは区別する。
   試験matrixでは12条件で実際の置換が起きた。
4. **滑らかな行と非滑らかな接合部を同じ二階差分の絶対値で数えていた。**
   四行の端点modelと、内部の符号付き高階defectを分ける。
   run数が変わる分岐・合流付近には局所的な二階差分の指標を残す。
5. **テーブルの丸めをLD曲率としてfitしていた。**
   フィット用の境界近傍2点だけ、既存の写像距離から正確な輝度則を評価する。
   実際の増光率では従来の輝度テーブルとBennett補正をそのまま使う。

## 共通構造

- 既存のrun fill、seed、Bennett補正、component refinementはuniform/LDとも共通。
- `same_grid_flux_error.hpp`: sqrt(t), t^(3/2), t^(5/2) の格子端点moment。
  有限和とmidpoint Euler--Maclaurinの尾部から構築し、一度だけ生成する小さな普遍係数表を補間する。
  レンズgeometryのキャッシュではない。表はすべてのepoch・geometryで共通。
- `cartesian_run_fill.hpp`: 行の端点fit、内部接合部の指標、componentごとの集計。
- `bennett_integration.hpp`: 同じmomentを使った内側方向の輝度model。
  linear LDの正則端では定数+sqrt(t)+t^(3/2)をfitする。
- `finite_source_magnifier.cpp`: 採用したcoarse/fine representationだけを集計。
- Uniformは輝度一定なので、輝度fitと隣接輝度保持を省く。行方向のmodelとfallbackはLDと同じ。

四行fitの曲率が強く、sqrt二次式の展開範囲を外れる場合、既に得たその像成分の行差分へ戻す。
短いcomponentやmodelが作れない場合も同じ。別の像探索や積分器を起動しない。
これはmodelの妥当性を見た軽いfallbackであり、rational lens map全体に対する厳密な剰余評価ではない。

## 速度

`shared-flux-error-v6`、同一build、同じ5geometryのscalar batch、1 thread、交互順の7回中央値。
コンパイル・初回の普遍係数表生成とwarmupは含めない。
通常component refinementは有効、追加support保証のflagは無効。

| source | nbin | 通常計算 [s] | 指標付き [s] | 追加費用 |
|---|---:|---:|---:|---:|
|uniform|64|0.00890|0.00914|2.7%|
|uniform|128|0.01306|0.01351|3.4%|
|uniform|256|0.02708|0.02795|3.2%|
|uniform|512|0.07826|0.08034|2.7%|
|linear LD|64|0.00953|0.01025|7.5%|
|linear LD|128|0.01534|0.01681|9.6%|
|linear LD|256|0.03605|0.03949|9.6%|
|linear LD|512|0.11378|0.12289|8.0%|

LDの初期cheap経路(約4〜6%)より指標計算は少し増えた。原因は精確な近傍輝度評価、端点fit、採用格子のmetadata集計。
半径別のfill・境界再求根は復活させていない。通常側は同一overlay buildの既存scalar経路であり、別buildのmasterとの厳密な同一性は主張しない。

## 検証artifact

`tests/diagnostics/results/recal2026/` 以下。

- `shared-flux-error-v6`: uniform/LD各20条件、bit一致・追加map0のassert、7batchのpaired timing。
- `shared-flux-error-holdout-v1`: 別geometry3組とc=0,.2,.8の独立参照。
  uniform VBMとSciPyの半径積分。積分精度の変更は参照の検証専用で、実行時受理には使わない。
- `shared-flux-error-holdout-v3`: その保存参照と安定性記録を再利用し、最終nativeを18条件で再評価。
- `shared-flux-error-v6/combined-validation.json`: 58条件の集計。過小評価も除外しない。
- `source_snapshot/manifest.json`: relevant dirty source、診断script、reportのhashとコピー。
- `unit-tests.log`: unit_core / unit_boundary_support 2/2。

C++単体試験では有限格子の解析的なpower積分からmomentを検証し、uniformの残差0、既知のsqrt/t^(3/2)の係数fitを確認する。
検討中のv1〜v5も保存。v5の大きなLD余剰はテーブル丸めを曲率へfitしたことによるもので、最終方式の成績として使わない。

## 残る限界

過小評価4条件の内訳は、pilotのLD nbin128が2条件、別geometryの小qでuniform/c=.2のnbin128が2条件。
過大評価は複数の像成分・内部gapの寄与が相殺する配置でまだ残る。
brightness table全体の誤差、未発見の像、modelの一般的剰余は上限保証していない。
平方根LD(d!=0)、polar、実行時のnbin受理への接続は今回の検証対象外。
