# 細分化の仕事量調査と最適化試行 — 2026-10-15

## 結果

**期待した「もう一段の大幅高速化」は未達。** 両経路を測り、7案を実装・比較した。大きな変更は管理費用や追加評価が勝ち、採用しなかった。採用したのは、数値を変えない小さな算術処理の削減と、両経路共通の診断traceだけ。

| 輝度 | 要求精度 | Cartesian速度比 | Polar速度比 | 精度を満たす共通ペア数（Cartesian、Polar） |
|---|---:|---:|---:|---:|
| Uniform | 1e-03 | 0.9919 | 1.0081 | 118/128、128/128 |
| Uniform | 1e-04 | 0.9968 | 1.0095 | 116/128、128/128 |
| Linear LD | 1e-03 | 1.0044 | 1.0099 | 112/128、128/128 |
| Linear LD | 1e-04 | 1.0032 | 1.0072 | 109/128、128/128 |

速度比は **今回開始時/採用版** のケース別比の中央値。Cartesianは±1%以内で横ばい、Polarも0.7〜1.0%程度の微小差。この結果を「爆速」「大幅改善」と呼ばない。基準は既にCartesianの精度修正を採用したv7、Polarはv43であり、以前の408→61件という修正を今回の成果として数え直していない。

256入力（Uniform/LD各128）、同一入力・要求精度、CPU0/1 thread、変更前後を交互に実行。初期Cartesian16、Polar2、半径配分4。warmup2回後、seed cache missを含むcold3回中央値。中心像・caustic・LDテーブルの準備は計時外。[全件計時rawと失敗](v3-paired256/summary.json)。8群とも、変更前に正確に受理できた標本からの新しい失敗0。失敗・未達・精度外受理は保存し、速いケースとして統計へ混ぜていない。

## 調査で分かったこと

### 最後の一段は両経路で違う

128入力×2精度×2手法を同じ形式で記録し、traceを入れた512条件の数値結果は変更前と一致した。[診断結果](trace-analysis.json)

- Cartesian1e-3: 有限結果のうちUniform **49/62**、LD **48/63** が細分化なし。初期値が実際の精度を満たしたのはそれぞれ60/62、62/63。後の段で一時的に精度を外すケースもあるため、「最初に参照値へ近づいたら止める」は採用しない。
- 細分化するCartesianでは、最後の試行が細分化時間の中央値約70〜79%。ただし、これは**呼び出し全体の70〜79%ではない**。大半が細分化しない1e-3の中央値を順序変更だけで大きく改善することはできない。
- Polarは中央値23〜32.5試行、最後の試行は細分化時間の約2%。Cartesianと同じ「最後の大きな一段」を仮定できない。
- 参照値に対して安定して精度を満たした後にも計算を続ける割合は残る。LDでは細分化時間の約47〜60%。これは**事後診断**で、運用上の停止条件や誤差保証ではない。

### Pythonの準備費用を除いてnative内を測定

計時runnerと同じpreplanned境界で、診断を有効にした128入力をCPU0で実行。受理された486条件を集計。[native内訳](native-cost/summary.json)、[raw](native-cost/raw.tsv)、実行コードは `native-cost/runner.cpp`。

- Cartesianは初期fillが呼び出し時間の中央値 **約73%**、全fill約83%、誤差集計約7%、細分化管理約0.5%。LDの初期fillは1e-3で約75%、1e-4で約68%。
- Polarの細分化時間は全体で約57%、LDでは約43〜48%。初期処理と細分化の比率もCartesianと異なる。
- seed構築は通常約1〜2%。速いseed呼び出しを省いて得た改善ではなく、既存接続を維持した。
- これらは診断有効時の同一呼び出し内の時間比。最適化の速度主張には診断無効のpaired計時を使う。Pythonラッパーの準備時間をnativeのボトルネックとして解釈していない。以前の`row_boundary_seconds`はcompile時無効の0だったため使わず、新しい出力からも外した。

**測定からの判断:** 共通の細分化管理を磨くだけでは、今回の中央値を大きく動かす余地は小さい。優先順位はCartesianの初期行走査・保持処理、Polarの列評価とLD/角度局所誤差であり、同じ変更を両方へ入れればよい、という仮説は支持されなかった。

## 試行と採否

| 案 | 観測 | 採否 |
|---|---|---|
| v1 共通の「残り誤差に対する期待改善/評価費用」 | Cartesian横ばい、Polar速度比0.96〜0.97。候補管理の費用が増える | 不採用 |
| v2 不要方向の倍増制限を緩和、Polar9/13点比較 | 評価点も速度もほぼ変わらず | 不採用 |
| v3 算術処理削減 | 上表の微小差。100万組のbrightness/角度比較でbit差0 | 採用、全体高速化の主張は小さく限定 |
| v4 32点の画像内部を曲率上界で判定しLDを実点で積分 | 総評価数はUniform約7.5〜11.7%、LD最大約3.2%減るが、短い行には追加費用 | v5へ試行 |
| v5 隣接行のhintでv4を絞る | 実点shadowで全32点を確認、追加失敗0。しかしCartesian速度比約0.98 | 不採用 |
| v6 AVX2/AVX512で8光線を先読み | 100万光線のbit差0。1点ごとのbuffer管理と余分な光線で速度比0.83〜0.88 | 不採用 |
| v7 SIMDを行の内部単位にまとめる | pilot最大倍率差2.22e-16、新しい精度外受理0。それでもLD速度比0.962/0.986 | 不採用 |

pilot128は各輝度64入力。v3のみ最終256入力と全件へ拡張した。不採用案について全コーパスの精度保証は主張しない。source、raw、計時、単体検証、失敗は各 `candidate-v*-source/` と各出力へ保存した。[試行履歴](EXPERIMENTS.md)

v5の独立した実点確認では、採用された各内部ブロックを全32点再計算し、積分差/許容値の最大は0.1925。多項式の離散和の単体10,000件は失敗0（最大7.19e-15）。この確認が成功しても、速度改善を意味しなかった。

v7では既存の未使用block hookの「反対側境界へ渡す2点目」が違う場所になり得る問題も専用2ケースで確認・修正した。**修正を含む試作経路全体を採用していない**。再開する場合は `candidate-v7-source/src/lcbinint/magnification/adaptive_cartesian_run_fill.hpp` の修正を先に適用する。対応単体は `check_batch_neighbours.cpp`。最終productionではinterior callbackが未設定で、この経路は使わない。

## 最終コードに残した変更

- 両経路の試行に、追加評価数、経過時間、全体の誤差・fluxの前後を記録する共通sidecar。`LCBININT_DIAGNOSTIC_REFINEMENT_WORK_TRACE=1` のときだけ記録し、requestのABIは変更しない。
- 線形LDで不要な2回目の平方根を省く。Cartesianの既知の線形LD分岐も式を直接使う。
- Cartesianの収束次数1/2での `pow` を対応する定数へ置換。
- Polarの既に正規化済みの角度への `fmod` と、有界な隣接角度差への一般的な `remainder` を省く。

**誤差係数、要求精度、初期格子、受理条件を緩めていない。** `1.09e-4`向けの調整、ケースID救済、楕円画像の差し引きモデル、実行時手法ルーターを追加していない。SIMD/内部block/高次則/新スケジューラの実験フラグはproductionから削除した。

## 最終検証

- Cartesian、Polarそれぞれ **14,432条件** を再実行。今回開始時と**全resultフィールドおよび例外が一致、変化0**。[Cartesian](final-cartesian-parity.json)、[Polar](final-polar-parity.json)
- Cartesianの既知の精度外受理61（12/26 Uniform、4/19 LD）、例外312形状/624条件、未達865条件は残る。Polarは全件受理・精度外受理0。
- Cartesian独立216条件は今回開始時と主要数値/例外一致。114受理・72例外・30未達、精度外受理0。[独立比較](independent-cartesian-parity.json)
- Polar独立882条件を新旧native両方で再実行し、全result一致。881受理・精度外受理0、既知の1e-5難例1件は未達のまま。driverは以前のdatasetに対するこの未達で両版ともexit1になる。**882全成功とは数えない**。[独立比較](independent-polar-parity.json)
- 最終nativeでCartesian132、Polar189内部チェック通過。過去の計測を再利用する前に、再ビルドのnativeが全件検証版とbyte一致、最終archiveから再リンクしたrunnerが計時版とbyte一致することも確認。[native](selected-native-parity.json)、[runner](selected-runner-parity.json)
- `certified=false`, `complete_error_model=false`を維持。基準参照は保存VBM RelTol1e-6と既存の独立裁定3件であり、全件の新しい高精度参照を生成したわけではない。

## 再現・引き継ぎ

`final-identity.json` にsource/native/input/reference/CPUを記録。sourceは `final-source/`、開始時のdirty sourceとの差分は `before-this-turn.patch`。開始時のstage済みindexはhashで不変を確認。未コミット・未push。6パネルは今回再生成していない。

project rootから、新しい出力名で:

```bash
OUT=tests/diagnostics/results/recal2026/refinement-work-optimization-20261015
TMPDIR="$PWD/build-adaptive-production/task-tmp" cmake --build build-adaptive-production -j2
python "$OUT/probe_work.py" --name NEW_TRACE --trace --cost
TMPDIR="$PWD/build-adaptive-production/task-tmp" python "$OUT/paired.py" --name NEW_PAIRED --count 256
python "$OUT/audit_full.py" --method cartesian --initial-bins 16 --output "$OUT/NEW_CART_FULL"
python "$OUT/audit_full.py" --method polar --initial-bins 2 --output "$OUT/NEW_POLAR_FULL"
```

不採用SIMD/内部和の単体を再コンパイルする場合は `-I "$OUT/candidate-v7-source/src" -I src`、離散和は `-I "$OUT/candidate-v5-source/src" -I src` を使用する。該当prototypeのheaderはproductionから取り除いている。
