# Polar誤差推定・積分経路の再設計（2026-10-15）

## 結果

本ターン開始時の保存版と同じ入力・初期格子数で比較し、採用版v43はPolarのケース別速度比中央値で **uniform 1.238 / 1.335倍、LD 1.262 / 1.358倍**（要求1e-3 / 1e-4）。LDの時間中央値は **0.525→0.357 ms、0.648→0.456 ms**。単なる安全係数変更ではなく、角度積分の比較対象、LD端点重みの積分、根の証明計算とデータ配置を変更した。

独立294条件/要求の誤差推定/実誤差中央値は **150.79→28.59、143.24→11.62**。実誤差/要求中央値は **0.00435→0.0213、0.00538→0.0527**。過大評価は大きく減ったが、まだ残る。これを数学的な誤差上限証明とは主張しない。

**LD1e-3でVBMに勝つ目標は未達。** 同じ128 LD入力を再計時したVBM/Polarケース別比中央値は **0.8705（1e-3）、3.1578（1e-4）**。Polar単独の1e-3はまだ遅い。精度や時間を見て入力ごとに手法を選ぶルーターは今回追加していない。

## 採用した設計

1. 通常の角度パネルは、分割の前に既存9点で比較則を更新する。返す値は正の重みのBoole5点、参照は9点の高次則。返す値との差を直接推定し、既知の非入れ子観測点も全て整合性検査に残す。古い低次差分を高次補正後まで加算し続けない。
2. 孤立した角度端点は変数変換後、9点則と13点Stieltjes拡張で比較する。低次則は9次、高次参照は17次まで厳密。符号付き重みの半径誤差には重みの絶対値を使う。
3. LD半径積分は実際のレンズ式の六次多項式から物理的な両端根を因数として除き、端点の平方根重みを解析的に分離する。楕円画像近似やモデル差し引きではない。Gauss-Chebyshev Uを1→3→7→15…と増やし、物理レンズ写像による独立評価点で因数化・端点誤差を監視する。初期段階の不要な導関数計算を除く。
4. 物理境界は符号の異なるブラケットとNewton仕上げを保持。既知のNewton点をブラケット片端に再利用し、境界位置の不確かさを積分誤差に加算する。
5. 定常点・画像境界のBernstein符号判定をdoubleで試し、曖昧な係数だけlong doubleで再評価する。解の個数を証明できない場合は元の定常点ソルバーを使う。Newton収束だけで完全性を判断しない。
6. 角度順の観測索引を平坦化し、小さいLD則のノード・作業配列を固定化する。実体の列キャッシュは既存の安定した参照を保つ。

flood fill、実像seed、画像成分の発見、境界・接触処理を保持。ケースIDによる救済分岐や楕円フラックス近似は追加していない。Cartesian積分アルゴリズムには本ターンの最適化を混ぜていない。

## 速度測定

正本: `../optimization-implementation-20261015/selected-redesign-final-paired256/summary.json`。

|光源|要求|ケース別 基準/採用版 中央値|基準ms中央値|採用版ms中央値|精度達成組|
|---|---:|---:|---:|---:|---:|
|uniform|1e-3|1.2383|0.458286|0.331887|128/128|
|uniform|1e-4|1.3349|0.562429|0.363633|128/128|
|LD|1e-3|1.2623|0.525116|0.356544|128/128|
|LD|1e-4|1.3579|0.648168|0.455756|128/128|

CPU0、1 thread、同一256入力（各profile128、seed202610151718）、実行順を入力ごと交互にした。Cartesian初期16、Polar初期2、半径配分4は両版同じ。warmup2回後、seed cache missを含むcold3回中央値。中心像・caustic形状・LD表は事前準備し計時外。ケース別速度比の中央値と、時間中央値の比は別の統計量。失敗・誤受理もJSONに全件保存し、精度達成組だけの速度統計と区別する。Cartesianは対照群であり、既存の失敗を今回修正したとは主張しない。

VBM正本: `selected-current-vbm-ld128/summary.json` と `identity.json`。同じ128 LD、warmup2・median3、入力ごとエンジン順交互、Tol1e-12・RelTol=要求、BinaryMagDark。VBMのPython binding呼び出しを含むがオブジェクト構築は除く。両者全128入力で要求精度達成、除外なし。Polarはnative C++呼び出しでありbinding境界は厳密に同一ではない。VBM/Polar中央値0.8705 / 3.1578、Polar時間中央値0.351482 / 0.453703 ms、VBM0.264553 / 1.089018 ms。

## 精度と残る制約

- 保存コーパス **14,432条件、全件受理、独立裁定後の要求精度外受理0、未達0、例外0**。`v43-full-shadow/adjudicated-summary.json`。保存VBM参照には既知の誤り3件があり、既存の独立裁定値を明示して使用。全域の高精度参照を新しく作ったわけではない。
- 独立参照882条件: **881受理、誤受理0、例外0、契約違反0、未達1**。未達は極端ケース `fresh-1-0.0`, 要求1e-5、実誤差8.71e-6だが指標4.29e-5で角度深さ上限。実誤差が小さくても受理扱いに書き換えない。ケース専用の救済は追加していない。
- 新しい16形状×3LD×3要求=144条件: 指標上は全件受理。独立参照が安定した **138条件で誤受理0**。6条件は高精度VBM参照が時間切れ。別LD layer-cake法の30秒再試行も時間切れのため、この6条件を検証済み数へ含めない。正本 `../polar-quadrature-rework/estimator-redesign-v43-new-holdout/validation.json`、`fresh-ld-layer-cake-status.json`。
- 内部契約 **189/189**。`v43-contracts.json`。
- 画像を横切るランダム10,000光線、9,069区間を証明、1,569区間で高精度係数へ昇格、物理的な根の個数不一致0。`wide_partition_check.txt`。
- ソルバー比較shadowは完全な定常点求解・物理残差と比較。計時時は無効。数値契約の `certified=false`, `complete_error_model=false` を保持。

## 採用しなかった案

全試行は `EXPERIMENTS.md`、candidate-*、個別監査JSONに残した。角度Hermite導関数モデル、親補間誤差の無条件継承、初期安全係数だけの縮小は未検出の角度変化を見逃した。画像に合わせた初期メッシュは遅く、定常点の角度帯キャッシュも証明費用が高かった。

最後のv47境界移動予測とv48微小LD寄与の上下限省略は、全14,432条件と独立882条件では同等の精度を保ったが、同じ256入力で追加の速度改善を確認できなかった。v48のケース別速度比はLD1.253 / 1.337、VBM/Polar0.884 / 3.217。単独測定間の小差を改善と解釈せず、より簡単で十分測定済みのv43を採用した。v46の初回コンパイル失敗直後に旧バイナリで走った監査は `*-invalid-build` として隔離し、結果に使っていない。

## 次に残る本質的な問題

独立ケースの中央値でも推定/実誤差が29 / 12残る。角度方向の各局所誤差を絶対値で合算する点、接触・成分出現を少数点だけでは見抜けない点、離散的な二分による行き過ぎが残る。一方、初期比較を無根拠に緩めるとコーパスに多数の誤受理が出る。安全係数だけの変更では埋まらない。LD1e-3のVBMとの差を消したとは主張しない。

本体ソース・ライブラリ・拡張・入力ハッシュは `final-identity.json`、採用ソースは `final-source/` に保存。未コミット。元のstaged変更、他のdirty作業を保持。

## 再実行コマンド

リポジトリ直下から実行。`OUT=tests/diagnostics/results/recal2026/angular-estimator-redesign-20261015`、`TMPDIR=/rogue1_8/nunota/lcbinint/build-adaptive-production/task-tmp`。新しい出力名を使い、計時と他のCPU処理は重ねない。

```sh
cmake --build build-adaptive-production -j2
python "$OUT/paired.py" --name NEW_PAIRED_NAME --count 256
python tests/diagnostics/recal2026/optimization_independent_regression_20261015.py --name NEW_INDEPENDENT_NAME
LCBININT_DIAGNOSTIC_POLAR_PARTITION_SHADOW=1 python tests/diagnostics/recal2026/polar_full_accuracy_audit.py --output "$OUT/NEW_FULL_NAME"
python "$OUT/adjudicate.py" "$OUT/NEW_FULL_NAME"
python tests/diagnostics/recal2026/polar_internal_contract_audit.py --output "$OUT/NEW_CONTRACT_NAME.json"
python tests/diagnostics/recal2026/polar_fresh_error_model_validation.py --name NEW_HOLDOUT_NAME --geometries 16 --seed 202610152001 --q-min 1e-6 --rho-min 1e-5 --references tests/diagnostics/results/recal2026/polar-quadrature-rework/estimator-redesign-v41-new-holdout/validation.json
```

VBM再計時は `measure_selected_ld_vbm.py` のnativeキャンペーンと新規出力先を明示して実行する。凍結ライブラリを使ったpaired runnerの実際のコンパイルコマンドは各 `summary.json` の `compile_commands`。採用版へ戻した後にも契約189と新規holdout144を再実行し、全てのnativeハッシュが元の検証版と一致した。

## 全入力6パネル更新

今回の採用版とCartesianを7,216入力、各2要求で再計時し、VBMも全入力で再計時した。seed構築込み・CPU0/1 thread。各入力・要求でcold3回中央値の速い手法を選び、精度は選択後に評価する。実行時ルーターではない。

図・集計: `../../../../../figures/recal2026/estimator-redesign-fastest-six-panel-20261015/`（リポジトリ直下からの正本は `figures/recal2026/estimator-redesign-fastest-six-panel-20261015/`）。VBM/選択手法の比中央値: Uniform1e-3 **0.2731**、Uniform1e-4 **0.4125**、LD1e-3 **1.3396**、LD1e-4 **4.6332**。全14,432条件で有限の選択結果が得られ、除外0。Polar単独は全件受理・精度達成、偽受理0。選択結果にはCartesian由来の偽受理が各 **44 / 100 / 26 / 82** 件あり、取り除かず図・集計へ含めた。速い方を選ぶ方式全体の精度保証とは主張しない。

Cartesian/Polar計時約900.5秒、VBM約82.5秒。4枚のPNG/PDFを実際に表示し、6軸・カラーバー・ラベルに欠けがないことを確認した。計時完了後のログ表示のみパス例外となったため、その経緯をsix-panel-full/completion-note.jsonに保存し、表示処理を修正した。raw.tsvは86,592行で完全。
